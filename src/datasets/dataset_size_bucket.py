import json
import os
import random
import numpy as np
from PIL import Image, ImageOps
from transformers import CLIPImageProcessor, ViTImageProcessor

import torch
from torch.utils.data import Dataset, Sampler, RandomSampler
from torchvision import transforms

from .utils import get_bucket_size, resize_and_center_crop, get_relative_bbox, mask_dialogs_from_image


def image_transform(pil_image):
    fn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    return fn(pil_image)


class MangaTrainSizeBucketDataset(Dataset):
    def __init__(
        self,
        ann_path,
        image_root,
        size_buckets,
        tokenizer,
        tokenizer_2,
        t_drop_rate=0.05,
        i_drop_rate=0.05,
        c_drop_rate=0.05,
        max_num_ips=4,
        max_num_ip_sources=1,
        max_num_dialogs=8,
        mask_dialog=False,
        load_context_image=False,
        ip_self_condition_rate=0.5,
        ip_flip_rate=0.5,
        min_ip_height=5,
        min_ip_width=5,
    ):
        with open(ann_path, 'r') as f:
            annotations = json.load(f)
        self.annotations = annotations
        self.image_root = image_root
        self.size_buckets = size_buckets
        self.buckets = {}
        self.bucket_size_index = {}
        self.partition_data()
        self.bucket_keys = list(self.buckets.keys())

        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2

        self.t_drop_rate = t_drop_rate
        self.i_drop_rate = i_drop_rate
        self.c_drop_rate = c_drop_rate

        self.max_num_ips = max_num_ips
        self.max_num_ip_sources = max_num_ip_sources
        self.max_num_dialogs = max_num_dialogs

        self.mask_dialog = mask_dialog

        self.load_context_image = load_context_image

        self.ip_self_condition_rate = ip_self_condition_rate
        self.ip_flip_rate = ip_flip_rate

        self.min_ip_height = min_ip_height
        self.min_ip_width = min_ip_width

        self.clip_image_processor = CLIPImageProcessor()
        self.magi_image_processor = ViTImageProcessor()

    def partition_data(self):
        for ann_idx, annotation in enumerate(self.annotations):
            for frame_idx, frame in enumerate(annotation['frames']):
                width = frame['bbox'][2] - frame['bbox'][0]
                height = frame['bbox'][3] - frame['bbox'][1]
                bucket_height, bucket_width, size_index = get_bucket_size(height, width, self.size_buckets)
                bucket_key = (bucket_height, bucket_width)
                
                if bucket_key not in self.buckets:
                    self.buckets[bucket_key] = []
                self.buckets[bucket_key].append({
                    "ann_idx": ann_idx, 
                    "frame_idx": frame_idx
                })
                self.bucket_size_index[bucket_key] = size_index

    def get_support_ip_ids(self, ann):
        support_ip_ids = set()
        for frame in ann["frames"]:
            id_count = {}
            for char in frame["characters"]:
                char_id = char["id"]
                if char_id in id_count:
                    id_count[char_id] += 1
                else:
                    id_count[char_id] = 1

                for char_id, count in id_count.items():
                    if count > 1:
                        support_ip_ids.add(char_id)

        return list(support_ip_ids)

    def sample_condition_characters(self, frame_info, support_ip_ids):
        ids = []
        bbox = []
        page_bbox = []
        ip_type = []
        frame_bbox = frame_info["bbox"]

        for idx in np.random.permutation(len(frame_info["characters"])):
            char = frame_info["characters"][idx]
            char_id = char["id"]
            # Skip if the character ID occurred more than once or should be dropped
            if char_id in support_ip_ids or random.random() < self.i_drop_rate:
                continue
            ids.append(char_id)
            relative_bbox = get_relative_bbox(frame_bbox, char["bbox"])
            bbox.append(relative_bbox)
            page_bbox.append(char["bbox"])
            ip_type.append(char["type"])
            if len(ids) >= self.max_num_ips:
                break
            
        # pad ids and bbox to self.max_num_ips
        while len(ids) < self.max_num_ips:
            ids.append(-1)
            bbox.append([0.0, 0.0, 0.0, 0.0])

        return ids, bbox, page_bbox, ip_type

    def load_ip_images(self, ann, ids, ip_bbox, ip_type, page_image):
        # choose IP image boxes
        ip_boxes = []
        ip_exists = []
        for i, id in enumerate(ids):
            if id != -1:
                if random.random() < self.ip_self_condition_rate:
                    x1, y1, x2, y2 = ip_bbox[i]
                    char_height = y2 - y1
                    char_width = x2 - x1
                    if char_height > self.min_ip_height and char_width > self.min_ip_width:
                        id_boxes = [ip_bbox[i]]
                    else:
                        id_boxes = []
                else:
                    id_boxes = []
                boxes = []
                for frame in ann['frames']:
                    for char in frame['characters']:
                        x1, y1, x2, y2 = char['bbox']
                        char_height = y2 - y1
                        char_width = x2 - x1
                        if char['id'] == id and char_height > self.min_ip_height and char_width > self.min_ip_width and char.get('type', 0) == 0:
                            boxes.append(char['bbox'])
                id_boxes += random.sample(boxes, min(self.max_num_ip_sources - len(id_boxes), len(boxes)))
                ip_exists += [1] * len(id_boxes)
                ip_exists += [0] * (self.max_num_ip_sources - len(id_boxes))
                while len(id_boxes) < self.max_num_ip_sources:
                    id_boxes += [[0.0, 0.0, 0.0, 0.0]]
                ip_boxes += id_boxes
            else:
                ip_exists += [0] * self.max_num_ip_sources
                ip_boxes += [[0.0, 0.0, 0.0, 0.0]] * self.max_num_ip_sources

        # load IP images
        ip_images = []
        for idx, box in enumerate(ip_boxes):
            if ip_exists[idx]:
                x1, y1, x2, y2 = box
                image = page_image.crop([x1, y1, x2, y2])
                if random.random() < self.ip_flip_rate:
                    image = ImageOps.mirror(image)
            else:
                image = Image.new('RGB', (224, 224), (0, 0, 0))
            
            ip_images.append(image)

        try:
            clip_ip_images = self.clip_image_processor(images=ip_images, return_tensors="pt").pixel_values
            magi_ip_images = self.magi_image_processor(images=ip_images, return_tensors="pt").pixel_values
        except Exception as e:
            print(f"preprocess ip images error. ann infomation:")
            print(f"ann: {ann}")
            print(f"ip_bbox: {ip_bbox}")
            for i, ip_image in enumerate(ip_images):
                print(f"ip_image_{i}.size: {ip_image.size}")

            ip_images = []
            for idx, box in enumerate(ip_boxes):
                image = Image.new('RGB', (224, 224), (0, 0, 0))
                ip_images.append(image)

            clip_ip_images = self.clip_image_processor(images=ip_images, return_tensors="pt").pixel_values
            magi_ip_images = self.magi_image_processor(images=ip_images, return_tensors="pt").pixel_values

        return clip_ip_images, magi_ip_images, ip_exists
        
    def __len__(self):
        return sum([len(value) for value in self.buckets.values()])

    def __getitem__(self, idx):
        if idx is None:
            return {
                "is_pseudo_sample": True,
            }
        # Load image and micro-conditions
        bucket_idx, sample_idx = idx
        bucket_key = self.bucket_keys[bucket_idx]
        bucket_height, bucket_width = bucket_key

        ann_idx = self.buckets[bucket_key][sample_idx]["ann_idx"]
        frame_idx = self.buckets[bucket_key][sample_idx]["frame_idx"]
        ann = self.annotations[ann_idx]
        frame_info = ann["frames"][frame_idx]
        image_path = os.path.join(self.image_root, ann["image_path"])

        x1, y1, x2, y2 = frame_info["bbox"]
        width = x2 - x1
        height = y2 - y1

        page_image = Image.open(image_path).convert("RGB") 
        if self.mask_dialog:
            page_image = mask_dialogs_from_image(page_image, ann)   
        image = page_image.crop([x1, y1, x2, y2])
        image, crop_coords_top_left = resize_and_center_crop(image, (bucket_height, bucket_width))
        image = image_transform(image)

        # Tokenize caption
        if random.random() < self.t_drop_rate:
            caption = ""
        else:
            caption = frame_info["caption"]
        text_input_ids = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        text_input_ids_2 = self.tokenizer_2(
            caption,
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        # Get support IP IDs
        support_ip_ids = self.get_support_ip_ids(ann)
        # Load IP images and IP bbox
        ip_ids, ip_bbox, ip_page_bbox, ip_type = self.sample_condition_characters(frame_info, support_ip_ids)
        ip_images, magi_ip_images, ip_exists = self.load_ip_images(ann, ip_ids, ip_page_bbox, ip_type, page_image)

        # Load context image
        if self.load_context_image and len(ann['frames']) > 1 and random.random() >= self.c_drop_rate:
            context_frame_info = random.choice(ann['frames'][:frame_idx] + ann['frames'][frame_idx+1:])
            x1, y1, x2, y2 = context_frame_info["bbox"]
            context_image = page_image.crop([x1, y1, x2, y2])
            drop_context = 0
        else:
            context_image = Image.new('RGB', (224, 224), (0, 0, 0))
            drop_context = 1
        context_image = self.clip_image_processor(images=[context_image], return_tensors="pt").pixel_values
        
        # Load dialog bbox
        dialog_bbox = []
        frame_bbox = frame_info["bbox"]
        for idx in np.random.permutation(len(frame_info["dialogs"])):
            bbox = get_relative_bbox(frame_bbox, frame_info["dialogs"][idx]["bbox"])
            dialog_bbox.append(bbox)
            if len(dialog_bbox) >= self.max_num_dialogs:
                break
        while len(dialog_bbox) < self.max_num_dialogs:
            dialog_bbox.append([0, 0, 0, 0])

        return {
            "image": image,
            "text_input_ids": text_input_ids,
            "text_input_ids_2": text_input_ids_2,
            "ip_exists": torch.Tensor(ip_exists).view(self.max_num_ips, self.max_num_ip_sources),
            "ip_images": ip_images,
            "magi_ip_images": magi_ip_images,
            "ip_bbox": torch.Tensor(ip_bbox),
            "dialog_bbox": torch.Tensor(dialog_bbox),
            "context_image": context_image,
            "drop_context": torch.Tensor(drop_context),
            "original_size": torch.Tensor([height, width]),
            "crop_coords_top_left": torch.Tensor(crop_coords_top_left),
            "target_size": torch.Tensor([bucket_height, bucket_width]),
            "is_pseudo_sample": False,
        }
    

def collate_fn(data):
    data = [example for example in data if example["is_pseudo_sample"] == False]

    images = torch.stack([example["image"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    text_input_ids_2 = torch.cat([example["text_input_ids_2"] for example in data], dim=0)
    ip_exists = torch.stack([example["ip_exists"] for example in data], dim=0)
    ip_images = torch.cat([example["ip_images"] for example in data], dim=0)
    magi_ip_images = torch.cat([example["magi_ip_images"] for example in data], dim=0)
    ip_bbox = torch.stack([example["ip_bbox"] for example in data])
    context_images = torch.cat([example["context_image"] for example in data], dim=0)
    drop_context = torch.cat([example["drop_context"] for example in data], dim=0)
    dialog_bbox = torch.stack([example["dialog_bbox"] for example in data])
    original_size = torch.stack([example["original_size"] for example in data])
    crop_coords_top_left = torch.stack([example["crop_coords_top_left"] for example in data])
    target_size = torch.stack([example["target_size"] for example in data])

    return {
        "images": images,
        "text_input_ids": text_input_ids,
        "text_input_ids_2": text_input_ids_2,
        "ip_exists": ip_exists,
        "ip_images": ip_images,
        "magi_ip_images": magi_ip_images,
        "ip_bbox": ip_bbox,
        "context_images": context_images,
        "drop_context": drop_context,
        "dialog_bbox": dialog_bbox,
        "original_size": original_size,
        "crop_coords_top_left": crop_coords_top_left,
        "target_size": target_size,
    }


class MangaEvaluationDataset(Dataset):
    def __init__(
        self,
        ann_path,
        image_root,
        max_num_ips=4,
        max_num_dialogs=8,
        mask_dialog=False,
        load_context_image=False,
        min_ip_height=0,
        min_ip_width=0,
        min_image_size_step=8,
    ):
        with open(ann_path, 'r') as f:
            annotations = json.load(f)
        self.annotations = annotations
        self.flatten_data()
        self.image_root = image_root

        self.max_num_ips = max_num_ips
        self.max_num_dialogs = max_num_dialogs

        self.mask_dialog = mask_dialog

        self.load_context_image = load_context_image

        self.min_ip_height = min_ip_height
        self.min_ip_width = min_ip_width
        self.min_image_size_step = min_image_size_step

    def flatten_data(self):
        self.ann_plain = []
        for annotation in self.annotations:
            for frame in annotation['frames']:
                frame["image_path"] = annotation["image_path"]
                frame["page_ann"] = annotation
                self.ann_plain.append(frame)

    def get_support_ip_ids(self, ann):
        support_ip_ids = set()
        for frame in ann["frames"]:
            id_count = {}
            for char in frame["characters"]:
                char_id = char["id"]
                if char_id in id_count:
                    id_count[char_id] += 1
                else:
                    id_count[char_id] = 1

                for char_id, count in id_count.items():
                    if count > 1:
                        support_ip_ids.add(char_id)

        return list(support_ip_ids)

    def sample_and_load_ip_images(self, frame_info, support_ip_ids, ann, page_image):
        bbox = []
        target_ip_bboxes = []
        frame_bbox = frame_info["bbox"]
        sorted_characters = sorted(frame_info["characters"], key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]), reverse=True)
        ip_boxes = []
        for char in sorted_characters:
            if char["id"] in support_ip_ids:
                continue
            boxes = []
            for frame in ann['frames']:
                for source_char in frame['characters']:
                    if source_char['id'] == char["id"]:
                        x1, y1, x2, y2 = source_char['bbox']
                        char_height = y2 - y1
                        char_width = x2 - x1
                        if char_height > self.min_ip_height and char_width > self.min_ip_width and source_char.get('type', 0) == 0:
                            boxes.append(source_char['bbox'])
            if len(boxes) > 0:
                ip_boxes.append(random.choice(boxes))
                relative_bbox = get_relative_bbox(frame_bbox, char["bbox"])
                bbox.append(relative_bbox)
                target_ip_bboxes.append(char["bbox"])
            if len(ip_boxes) >= self.max_num_ips:
                break

        ip_images = []
        for box in ip_boxes:
            x1, y1, x2, y2 = box
            image = page_image.crop([x1, y1, x2, y2])
            ip_images.append(image)

        target_ip_images = []
        for box in target_ip_bboxes:
            x1, y1, x2, y2 = box
            image = page_image.crop([x1, y1, x2, y2])
            target_ip_images.append(image)

        return bbox, ip_boxes, ip_images, target_ip_images
        
    def __len__(self):
        return len(self.ann_plain)

    def __getitem__(self, idx):
        ann = self.ann_plain[idx]
        image_path = os.path.join(self.image_root, ann["image_path"])
        caption = ann["caption"]

        x1, y1, x2, y2 = ann["bbox"]
        width = round((x2 - x1) / self.min_image_size_step) * self.min_image_size_step
        height = round((y2 - y1) / self.min_image_size_step) * self.min_image_size_step

        # Load page image
        page_image = Image.open(image_path).convert("RGB")
        if self.mask_dialog:
            page_image = mask_dialogs_from_image(page_image, ann["page_ann"])  

        # Load IP images and IP bbox
        # support_ip_ids = self.get_support_ip_ids(ann["page_ann"])
        support_ip_ids = [] # there are no support ids in mangadex
        ip_bbox, condition_ip_bbox, ip_images, target_ip_images = self.sample_and_load_ip_images(ann, support_ip_ids, ann["page_ann"], page_image)

        # Load context image
        if self.load_context_image and len(ann["page_ann"]['frames']) >= 1:
            context_frame_info = random.choice(ann["page_ann"]['frames'])
            x1, y1, x2, y2 = context_frame_info["bbox"]
            context_image = page_image.crop([x1, y1, x2, y2])
        else:
            context_image = None
        
        # Load dialog bbox
        dialog_bbox = []
        frame_bbox = ann["bbox"]
        for idx in np.random.permutation(len(ann["dialogs"])):
            bbox = get_relative_bbox(frame_bbox, ann["dialogs"][idx]["bbox"])
            dialog_bbox.append(bbox)
            if len(dialog_bbox) >= self.max_num_dialogs:
                break

        return {
            "image_path": image_path,
            "caption": caption,
            "height": height,
            "width": width,
            "ip_images": ip_images,
            "target_ip_images": target_ip_images,
            "context_image": context_image,
            "ip_bbox": ip_bbox,
            "condition_ip_bbox": condition_ip_bbox,
            "dialog_bbox": dialog_bbox,
            "frame_bbox": ann["bbox"],
            "frame_ann": ann,
            "ann": ann["page_ann"],
        }


class BucketBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.buckets = dataset.buckets
        self.bucket_size_index = dataset.bucket_size_index
        self.batch_size = batch_size
        self.bucket_keys = list(self.buckets.keys())
        self.bucket_batches = self.calculate_bucket_batches()

        self.bucket_samplers = [RandomSampler(self.buckets[bucket_key]) for bucket_key in self.bucket_keys]
        # self.bucket_samplers = [SequentialSampler(self.buckets[bucket_key]) for bucket_key in self.bucket_keys]
        # self.bucket_sampler_iters = [iter(sampler) for sampler in self.bucket_samplers]

    def calculate_bucket_batches(self):
        bucket_batches = []
        for bucket_key in self.bucket_keys:
            batch_size = max(1, round(self.batch_size / (2 ** (self.bucket_size_index[bucket_key] * 2))))
            bucket_length = len(self.buckets[bucket_key])
            bucket_batches.append((bucket_length + batch_size - 1) // batch_size)

        # print(f"rank {accelerator.local_process_index}, bucket_batches: {bucket_batches}")
        return bucket_batches
    
    def get_pseudo_full_batch(self, batch):
        return batch + [None] * (self.batch_size - len(batch))

    def __iter__(self):
        bucket_sampler_iters = [iter(sampler) for sampler in self.bucket_samplers]
        
        batch_bucket_indexes = []
        for idx, num_batch in enumerate(self.bucket_batches):
            batch_bucket_indexes += [idx] * num_batch

        random.shuffle(batch_bucket_indexes)

        for bucket_idx in batch_bucket_indexes:
            bucket_key = self.bucket_keys[bucket_idx]
            batch_size = max(1, round(self.batch_size / (2 ** (self.bucket_size_index[bucket_key] * 2))))
            batch = []
            while True:
                try:
                    idx = next(bucket_sampler_iters[bucket_idx])
                    idx = [bucket_idx, idx]
                    batch.append(idx)
                    if len(batch) == batch_size:
                        # Accelerate seems cannot handle batchsampler with varying batch_sizes in multigpu training.
                        # Pad to the largest batch_size.
                        # print(f"rank {accelerator.local_process_index} yield batch, bucket_key: {bucket_key} batch: {batch} batchsize: {batch_size}")
                        yield self.get_pseudo_full_batch(batch)
                        break
                except StopIteration:
                    # print(f"rank {accelerator.local_process_index} StopIteration, bucket_key: {bucket_key} batch: {batch}")
                    if len(batch) > 0:
                        yield self.get_pseudo_full_batch(batch)
                    break

    def __len__(self):
        return sum(self.bucket_batches)
