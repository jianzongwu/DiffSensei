import json
import os
import random
import numpy as np
from PIL import Image, ImageOps
from transformers import CLIPImageProcessor, ViTImageProcessor

import torch
from torch.utils.data import Dataset, Sampler, RandomSampler
from torchvision import transforms

from .utils import get_bucket_size, resize_and_center_crop, get_relative_bbox, mask_dialogs_from_image, character_indices


BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'
BBOX_START_TOKEN = '<box_start>'
BBOX_END_TOKEN = '<box_end>'
LOC_TOKENS = '<loc-{:d}>'


def image_transform(pil_image):
    fn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    return fn(pil_image)


class MangaTrainMLLMDataset(Dataset):
    def __init__(
        self,
        ann_path,
        image_root,
        size_buckets,
        tokenizer,
        tokenizer_2,
        tokenizer_mllm,
        t_drop_rate=0.05,
        i_drop_rate=0.05,
        c_drop_rate=0.05,
        max_num_ips=4,
        max_num_ip_sources=1, # Can only be 1 in this dataset now
        max_num_dialogs=8,
        mask_dialog=False,
        ip_self_condition_rate=0.5,
        ip_flip_rate=0.5,
        min_ip_height=3,
        min_ip_width=3,
        num_img_tokens=64,
        num_loc_tokens=224,
        max_token_length=400,
        max_caption_length=77,
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
        self.tokenizer_mllm = tokenizer_mllm

        self.t_drop_rate = t_drop_rate
        self.i_drop_rate = i_drop_rate
        self.c_drop_rate = c_drop_rate

        self.max_num_ips = max_num_ips
        self.max_num_ip_sources = max_num_ip_sources
        self.max_num_dialogs = max_num_dialogs

        self.mask_dialog = mask_dialog

        self.ip_self_condition_rate = ip_self_condition_rate
        self.ip_flip_rate = ip_flip_rate

        self.min_ip_height = min_ip_height
        self.min_ip_width = min_ip_width

        self.num_img_tokens = num_img_tokens
        self.num_loc_tokens = num_loc_tokens
        self.max_caption_length = max_caption_length
        self.max_token_length = max_token_length

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

        sorted_characters = sorted(frame_info["characters"], key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]), reverse=True)
        for char in sorted_characters:
            char_id = char["id"]
            x1, y1, x2, y2 = char['bbox']
            char_height = y2 - y1
            char_width = x2 - x1
            # Skip if the character ID occurred more than once or should be dropped
            if char_id in support_ip_ids or random.random() < self.i_drop_rate or char_height <= self.min_ip_height or char_width <= self.min_ip_width:
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
        target_ip_bboxes = []
        ip_exists = []
        num_valid_target_ips = 0
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
                target_ip_bboxes.append(ip_bbox[i])
                if len(id_boxes) > 0:
                    num_valid_target_ips += 1
                while len(id_boxes) < self.max_num_ip_sources:
                    id_boxes += [[0.0, 0.0, 0.0, 0.0]]
                ip_boxes += id_boxes
            else:
                ip_exists += [0] * self.max_num_ip_sources
                ip_boxes += [[0.0, 0.0, 0.0, 0.0]] * self.max_num_ip_sources
                target_ip_bboxes.append([0.0, 0.0, 0.0, 0.0])

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

        clip_ip_images = self.clip_image_processor(images=ip_images, return_tensors="pt").pixel_values
        magi_ip_images = self.magi_image_processor(images=ip_images, return_tensors="pt").pixel_values

        # load target IP images
        if len(target_ip_bboxes) > 0:
            target_ip_images = []
            for idx, box in enumerate(target_ip_bboxes):
                x1, y1, x2, y2 = box
                if sum(box) > 0:
                    image = page_image.crop([x1, y1, x2, y2])
                else:
                    image = Image.new('RGB', (224, 224), (0, 0, 0))
                target_ip_images.append(image)

            target_clip_ip_images = self.clip_image_processor(images=target_ip_images, return_tensors="pt").pixel_values
            target_magi_ip_images = self.magi_image_processor(images=target_ip_images, return_tensors="pt").pixel_values
        else:
            target_clip_ip_images = torch.randn([0, 3, self.clip_image_processor.size['shortest_edge'], self.clip_image_processor.size['shortest_edge']])
            target_magi_ip_images = torch.randn([0, 3, self.magi_image_processor.size['height'], self.magi_image_processor.size['width']])

        return clip_ip_images, magi_ip_images, ip_exists, target_clip_ip_images, target_magi_ip_images, num_valid_target_ips
    
    def relative_bbox_to_loc_tokens(self, rel_bbox):
        point_loc_token_idx = []        
        for rel_pos in rel_bbox:
            loc_token_idx = int(max(0, min(rel_pos, 0.99)) * self.num_loc_tokens)
            point_loc_token_idx.append(loc_token_idx)

        x1, y1, x2, y2 = point_loc_token_idx
        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2
        width = x2 - x1
        height = y2 - y1
        loc_tokens = [LOC_TOKENS.format(x_center), LOC_TOKENS.format(y_center), LOC_TOKENS.format(width), LOC_TOKENS.format(height)]

        return BBOX_START_TOKEN + ''.join(loc_tokens) + BBOX_END_TOKEN
    
    def truncate_caption(self, caption):
        tokens = self.tokenizer_mllm.encode(caption, add_special_tokens=False)
        filtered_tokens = tokens[:self.max_caption_length]
        truncated_caption = self.tokenizer_mllm.decode(filtered_tokens)
        return truncated_caption
        
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
        ip_images, magi_ip_images, ip_exists, target_clip_ip_images, target_magi_ip_images, num_valid_target_ips = self.load_ip_images(ann, ip_ids, ip_page_bbox, ip_type, page_image)
        
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

        # Generate MLLM input ids and labels
        instruction = ""
        response = ""
        image_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in range(self.num_img_tokens)]) + EOI_TOKEN
        instruction += self.truncate_caption(frame_info["caption"]) + '\n'
        instruction += image_tokens + '\n'
        response += image_tokens

        input_ids = []
        labels = []
        input_text = ''

        item_ids = self.tokenizer_mllm.encode(instruction, add_special_tokens=False)
        item_labels = [-100] * len(item_ids)
        input_text += instruction
        input_ids.extend(item_ids)
        labels.extend(item_labels)

        item_ids = self.tokenizer_mllm.encode(response, add_special_tokens=False)
        item_labels = item_ids
        input_text += response
        input_ids.extend(item_ids)
        labels.extend(item_labels)

        input_ids = [self.tokenizer_mllm.bos_token_id] + input_ids + [self.tokenizer_mllm.eos_token_id]
        attention_mask = [1] * len(input_ids)
        labels = [-100] + labels + [self.tokenizer_mllm.eos_token_id]

        boi_token_id = self.tokenizer_mllm.encode(BOI_TOKEN, add_special_tokens=False)[1]
        eoi_token_id = self.tokenizer_mllm.encode(EOI_TOKEN, add_special_tokens=False)[1]
        ids_cmp_mask = [False] * len(input_ids)
        ids_gen_mask = [False] * len(input_ids)
        embeds_cmp_mask = [True, False]
        embeds_gen_mask = [False, True]

        # print(f"input text:\n{instruction + response}\n")
        # print(f"len(target_ip_indices): {len(target_ip_indices)} len(input_ids): {len(input_ids)}")
        # print(f"labels: {labels}")
        
        if len(input_ids) >= self.max_token_length:
            # input_ids = input_ids[:self.max_token_length]
            # attention_mask = attention_mask[:self.max_token_length]
            # labels = labels[:self.max_token_length]
            # ids_cmp_mask = ids_cmp_mask[:self.max_token_length]
            # ids_gen_mask = ids_gen_mask[:self.max_token_length]
            print(f'A sample has been removed because of max length. input_text: {input_text}\n len(input_ids): {len(input_ids)}')
            return {"is_pseudo_sample": True}
        else:
            padding_length = self.max_token_length - len(input_ids)
            input_ids = input_ids + [self.tokenizer_mllm.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            labels = labels + [-100] * padding_length
            ids_cmp_mask = ids_cmp_mask + [False] * padding_length
            ids_gen_mask = ids_gen_mask + [False] * padding_length

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        ids_cmp_mask = torch.tensor(ids_cmp_mask, dtype=torch.bool)
        ids_gen_mask = torch.tensor(ids_gen_mask, dtype=torch.bool)
        embeds_cmp_mask = torch.tensor(embeds_cmp_mask)
        embeds_gen_mask = torch.tensor(embeds_gen_mask)

        boi_idx = torch.where(input_ids == boi_token_id)[0].tolist()
        eoi_idx = torch.where(input_ids == eoi_token_id)[0].tolist()

        for i in range(1):
            ids_cmp_mask[boi_idx[i] + 1 : eoi_idx[i]] = True
            
        for i in range(1):
            ids_gen_mask[boi_idx[-i-1] + 1 : eoi_idx[-i-1]] = True
            labels[boi_idx[-i-1] + 1 : eoi_idx[-i-1]] = -100

        return {
            "image": image,
            "text_input_ids": text_input_ids,
            "text_input_ids_2": text_input_ids_2,
            "ip_exists": torch.Tensor(ip_exists).view(self.max_num_ips, self.max_num_ip_sources),
            "ip_images": ip_images,
            "magi_ip_images": magi_ip_images,
            "ip_bbox": torch.Tensor(ip_bbox),
            "dialog_bbox": torch.Tensor(dialog_bbox),
            "original_size": torch.Tensor([height, width]),
            "crop_coords_top_left": torch.Tensor(crop_coords_top_left),
            "target_size": torch.Tensor([bucket_height, bucket_width]),
            # mllm
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'ids_cmp_mask': ids_cmp_mask,
            'ids_gen_mask': ids_gen_mask,
            'embeds_cmp_mask': embeds_cmp_mask,
            'embeds_gen_mask': embeds_gen_mask,
            'target_clip_ip_images': target_clip_ip_images,
            'target_magi_ip_images': target_magi_ip_images,
            'input_text': input_text,
            # all
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
    dialog_bbox = torch.stack([example["dialog_bbox"] for example in data])
    original_size = torch.stack([example["original_size"] for example in data])
    crop_coords_top_left = torch.stack([example["crop_coords_top_left"] for example in data])
    target_size = torch.stack([example["target_size"] for example in data])
    # mllm
    input_ids = torch.stack([item['input_ids'] for item in data], dim=0)
    attention_mask = torch.stack([item['attention_mask'] for item in data], dim=0)
    labels = torch.stack([item['labels'] for item in data], dim=0)
    ids_cmp_mask = torch.stack([item['ids_cmp_mask'] for item in data], dim=0)
    ids_gen_mask = torch.stack([item['ids_gen_mask'] for item in data], dim=0)
    embeds_cmp_mask = torch.cat([item['embeds_cmp_mask'] for item in data], dim=0).to(torch.bool)
    embeds_gen_mask = torch.cat([item['embeds_gen_mask'] for item in data], dim=0).to(torch.bool)
    target_clip_ip_images = torch.cat([item['target_clip_ip_images'] for item in data], dim=0)
    target_magi_ip_images = torch.cat([item['target_magi_ip_images'] for item in data], dim=0)
    input_text = [item['input_text'] for item in data]

    return {
        "images": images,
        "text_input_ids": text_input_ids,
        "text_input_ids_2": text_input_ids_2,
        "ip_exists": ip_exists,
        "ip_images": ip_images,
        "magi_ip_images": magi_ip_images,
        "ip_bbox": ip_bbox,
        "dialog_bbox": dialog_bbox,
        "original_size": original_size,
        "crop_coords_top_left": crop_coords_top_left,
        "target_size": target_size,
        # mllm
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'ids_cmp_mask': ids_cmp_mask,
        'ids_gen_mask': ids_gen_mask,
        'embeds_cmp_mask': embeds_cmp_mask,
        'embeds_gen_mask': embeds_gen_mask,
        'target_clip_ip_images': target_clip_ip_images,
        'target_magi_ip_images': target_magi_ip_images,
        'input_text': input_text,
    }


class MangaEvalMLLMDataset(Dataset):
    def __init__(
        self,
        ann_path,
        image_root,
        tokenizer_mllm,
        max_num_ips=4,
        max_num_dialogs=8,
        num_img_tokens=64,
        num_loc_tokens=224,
        mask_dialog=False,
        min_ip_height=0,
        min_ip_width=0,
        min_image_size_step=8,
        max_caption_length=77,
    ):
        with open(ann_path, 'r') as f:
            annotations = json.load(f)
        self.annotations = annotations
        self.flatten_data()
        self.image_root = image_root

        self.tokenizer_mllm = tokenizer_mllm

        self.max_num_ips = max_num_ips
        self.max_num_dialogs = max_num_dialogs
        self.num_img_tokens = num_img_tokens
        self.num_loc_tokens = num_loc_tokens

        self.mask_dialog = mask_dialog

        self.min_ip_height = min_ip_height
        self.min_ip_width = min_ip_width
        self.min_image_size_step = min_image_size_step

        self.max_caption_length = max_caption_length

        self.clip_image_processor = CLIPImageProcessor()
        self.magi_image_processor = ViTImageProcessor()

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

    def relative_bbox_to_loc_tokens(self, rel_bbox):
        point_loc_token_idx = []        
        for rel_pos in rel_bbox:
            loc_token_idx = int(max(0, min(rel_pos, 0.99)) * self.num_loc_tokens)
            point_loc_token_idx.append(loc_token_idx)

        x1, y1, x2, y2 = point_loc_token_idx
        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2
        width = x2 - x1
        height = y2 - y1
        loc_tokens = [LOC_TOKENS.format(x_center), LOC_TOKENS.format(y_center), LOC_TOKENS.format(width), LOC_TOKENS.format(height)]

        return BBOX_START_TOKEN + ''.join(loc_tokens) + BBOX_END_TOKEN
    
    def truncate_caption(self, caption):
        tokens = self.tokenizer_mllm.encode(caption, add_special_tokens=False)
        filtered_tokens = tokens[:self.max_caption_length]
        truncated_caption = self.tokenizer_mllm.decode(filtered_tokens)
        return truncated_caption
        
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
        
        # Load dialog bbox
        dialog_bbox = []
        frame_bbox = ann["bbox"]
        for idx in np.random.permutation(len(ann["dialogs"])):
            bbox = get_relative_bbox(frame_bbox, ann["dialogs"][idx]["bbox"])
            dialog_bbox.append(bbox)
            if len(dialog_bbox) >= self.max_num_dialogs:
                break

        # Generate MLLM input ids and labels
        instruction = ""
        image_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in range(self.num_img_tokens)]) + EOI_TOKEN
        instruction += self.truncate_caption(caption) + '\n'
        instruction += image_tokens + '\n'

        input_ids = [self.tokenizer_mllm.bos_token_id] + self.tokenizer_mllm.encode(instruction, add_special_tokens=False)

        boi_token_id = self.tokenizer_mllm.encode(BOI_TOKEN, add_special_tokens=False)[1]
        eoi_token_id = self.tokenizer_mllm.encode(EOI_TOKEN, add_special_tokens=False)[1]
        ids_cmp_mask = [False] * len(input_ids)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        ids_cmp_mask = torch.tensor(ids_cmp_mask, dtype=torch.bool)

        boi_idx = torch.where(input_ids == boi_token_id)[0].tolist()
        eoi_idx = torch.where(input_ids == eoi_token_id)[0].tolist()

        for i in range(1):
            ids_cmp_mask[boi_idx[i] + 1 : eoi_idx[i]] = True

        return {
            "image_path": image_path,
            "caption": caption,
            "height": height,
            "width": width,
            "ip_images": ip_images,
            "target_ip_images": target_ip_images,
            "ip_bbox": ip_bbox,
            "condition_ip_bbox": condition_ip_bbox,
            "dialog_bbox": dialog_bbox,
            "frame_bbox": ann["bbox"],
            "frame_ann": ann,
            "ann": ann["page_ann"],
            # mllm
            'input_text': instruction,
            'input_ids': input_ids,
            'ids_cmp_mask': ids_cmp_mask,
        }


class MangaInferenceMLLMDataset(Dataset):
    def __init__(
        self,
        ann_path,
        image_root,
        tokenizer_mllm,
        max_num_ips=4,
        max_num_dialogs=8,
        num_img_tokens=64,
        num_loc_tokens=224,
        mask_dialog=False,
        min_ip_height=0,
        min_ip_width=0,
        min_image_size_step=8,
        max_caption_length=77,
    ):
        with open(ann_path, 'r') as f:
            annotations = json.load(f)
        self.flatten_data(annotations)
        self.page_source_chars = {}

        self.image_root = image_root

        self.tokenizer_mllm = tokenizer_mllm

        self.max_num_ips = max_num_ips
        self.max_num_dialogs = max_num_dialogs
        self.num_img_tokens = num_img_tokens
        self.num_loc_tokens = num_loc_tokens

        self.mask_dialog = mask_dialog

        self.min_ip_height = min_ip_height
        self.min_ip_width = min_ip_width
        self.min_image_size_step = min_image_size_step

        self.max_caption_length = max_caption_length

        self.clip_image_processor = CLIPImageProcessor()
        self.magi_image_processor = ViTImageProcessor()

    def flatten_data(self, annotations):
        self.ann_plain = []
        for annotation in annotations:
            for frame in annotation['frames']:
                frame["image_path"] = annotation["image_path"]
                frame["page_ann"] = annotation
                self.ann_plain.append(frame)

    def sample_source_characters(self, page_ann):
        char_dict = {}
        image_path = page_ann['image_path']
        if self.page_source_chars.get(image_path, None) is not None:
            return self.page_source_chars[image_path]["char_ids"], self.page_source_chars[image_path]["char_bboxes"]

        for frame in page_ann["frames"]:
            for char in frame["characters"]:
                x1, y1, x2, y2 = char['bbox']
                char_height = y2 - y1
                char_width = x2 - x1
                if char_height > self.min_ip_height and char_width > self.min_ip_width and char.get('type', 0) == 0:
                    char_id = char["id"]
                    bbox = [x1, y1, x2, y2]
                    if char_id not in char_dict:
                        char_dict[char_id] = []
                    char_dict[char_id].append(bbox)

        char_ids = list(char_dict.keys())
        char_bboxes = [random.choice(bboxes) for bboxes in char_dict.values()]
        self.page_source_chars[image_path] = {
            "char_ids": char_ids,
            "char_bboxes": char_bboxes
        }

        return char_ids, char_bboxes

    def load_ip_images(self, source_char_ids, source_char_bboxes, frame_ann, page_image):
        frame_bbox = frame_ann["bbox"]
        sorted_characters = sorted(frame_ann["characters"], key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]), reverse=True)
        target_char_rel_bboxes = []
        valid_source_char_bboxes = []
        for char in sorted_characters:
            if char["id"] not in source_char_ids:
                continue
            x1, y1, x2, y2 = char['bbox']
            char_height = y2 - y1
            char_width = x2 - x1
            if char_height <= self.min_ip_height or char_width <= self.min_ip_width:
                continue
            char_idx = source_char_ids.index(char["id"])
            
            valid_source_char_bboxes.append(source_char_bboxes[char_idx])
            relative_bbox = get_relative_bbox(frame_bbox, char["bbox"])
            target_char_rel_bboxes.append(relative_bbox)
            
            if len(target_char_rel_bboxes) >= self.max_num_ips:
                break

        ip_images = []
        for box in valid_source_char_bboxes:
            x1, y1, x2, y2 = box
            image = page_image.crop([x1, y1, x2, y2])
            ip_images.append(image)

        return target_char_rel_bboxes, valid_source_char_bboxes, ip_images
    
    def truncate_caption(self, caption):
        tokens = self.tokenizer_mllm.encode(caption, add_special_tokens=False)
        filtered_tokens = tokens[:self.max_caption_length]
        truncated_caption = self.tokenizer_mllm.decode(filtered_tokens)
        return truncated_caption
        
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

        # Load IP images and IP bbox
        source_char_ids, source_char_bboxes = self.sample_source_characters(ann["page_ann"])
        target_char_rel_bboxes, valid_source_char_bboxes, ip_images = self.load_ip_images(source_char_ids, source_char_bboxes, ann, page_image)
        
        # Load dialog bbox
        dialog_bbox = []
        frame_bbox = ann["bbox"]
        for idx in np.random.permutation(len(ann["dialogs"])):
            bbox = get_relative_bbox(frame_bbox, ann["dialogs"][idx]["bbox"])
            dialog_bbox.append(bbox)
            if len(dialog_bbox) >= self.max_num_dialogs:
                break

        # Generate MLLM input ids and labels
        instruction = ""
        image_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in range(self.num_img_tokens)]) + EOI_TOKEN
        instruction += self.truncate_caption(caption) + '\n'
        instruction += image_tokens + '\n'

        input_ids = [self.tokenizer_mllm.bos_token_id] + self.tokenizer_mllm.encode(instruction, add_special_tokens=False)

        boi_token_id = self.tokenizer_mllm.encode(BOI_TOKEN, add_special_tokens=False)[1]
        eoi_token_id = self.tokenizer_mllm.encode(EOI_TOKEN, add_special_tokens=False)[1]
        ids_cmp_mask = [False] * len(input_ids)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        ids_cmp_mask = torch.tensor(ids_cmp_mask, dtype=torch.bool)

        boi_idx = torch.where(input_ids == boi_token_id)[0].tolist()
        eoi_idx = torch.where(input_ids == eoi_token_id)[0].tolist()

        for i in range(1):
            ids_cmp_mask[boi_idx[i] + 1 : eoi_idx[i]] = True

        return {
            "image_path": image_path,
            "caption": caption,
            "height": height,
            "width": width,
            "ip_images": ip_images,
            "ip_bbox": target_char_rel_bboxes,
            "condition_ip_bbox": valid_source_char_bboxes,
            "dialog_bbox": dialog_bbox,
            "frame_bbox": ann["bbox"],
            "frame_ann": ann,
            "ann": ann["page_ann"],
            # mllm
            'input_text': instruction,
            'input_ids': input_ids,
            'ids_cmp_mask': ids_cmp_mask,
        }


class MangaInferenceCharImageMLLMDataset(Dataset):
    def __init__(
        self,
        ann_path,
        image_root,
        character_image_dir,
        tokenizer_mllm,
        max_num_ips=4,
        max_num_dialogs=8,
        num_img_tokens=64,
        num_loc_tokens=224,
        mask_dialog=False,
        min_ip_height=0,
        min_ip_width=0,
        min_image_size_step=8,
        max_caption_length=77,
    ):
        with open(ann_path, 'r') as f:
            annotations = json.load(f)
        self.flatten_data(annotations)
        self.page_source_chars = {}
        self.source_character_images = self.get_character_images(character_image_dir)

        self.image_root = image_root

        self.tokenizer_mllm = tokenizer_mllm

        self.max_num_ips = max_num_ips
        self.max_num_dialogs = max_num_dialogs
        self.num_img_tokens = num_img_tokens
        self.num_loc_tokens = num_loc_tokens

        self.mask_dialog = mask_dialog

        self.min_ip_height = min_ip_height
        self.min_ip_width = min_ip_width
        self.min_image_size_step = min_image_size_step

        self.max_caption_length = max_caption_length

        self.clip_image_processor = CLIPImageProcessor()
        self.magi_image_processor = ViTImageProcessor()

    def flatten_data(self, annotations):
        self.ann_plain = []
        for annotation in annotations:
            for frame in annotation['frames']:
                frame["image_path"] = annotation["image_path"]
                frame["page_ann"] = annotation
                self.ann_plain.append(frame)

    def get_character_images(self, character_image_dir):
        character_images = []
        character_image_names = os.listdir(character_image_dir)
        for character_image_name in character_image_names:
            character_image_path = os.path.join(character_image_dir, character_image_name)
            character_images.append(Image.open(character_image_path).convert("RGB"))

        return character_images

    def load_ip_images(self, frame_ann):
        frame_bbox = frame_ann["bbox"]
        sorted_characters = sorted(frame_ann["characters"], key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]), reverse=True)
        target_char_rel_bboxes = []
        for char in sorted_characters:
            x1, y1, x2, y2 = char['bbox']
            char_height = y2 - y1
            char_width = x2 - x1
            if char_height <= self.min_ip_height or char_width <= self.min_ip_width:
                continue

            relative_bbox = get_relative_bbox(frame_bbox, char["bbox"])
            target_char_rel_bboxes.append(relative_bbox)
            
            if len(target_char_rel_bboxes) >= min(self.max_num_ips, len(self.source_character_images)):
                break

        source_ip_indices = list(range(len(self.source_character_images)))
        random.shuffle(source_ip_indices)
        ip_images = []
        for idx in source_ip_indices[:len(target_char_rel_bboxes)]:
            ip_images.append(self.source_character_images[idx])

        return target_char_rel_bboxes, source_ip_indices, ip_images
    
    def truncate_caption(self, caption):
        tokens = self.tokenizer_mllm.encode(caption, add_special_tokens=False)
        filtered_tokens = tokens[:self.max_caption_length]
        truncated_caption = self.tokenizer_mllm.decode(filtered_tokens)
        return truncated_caption
        
    def __len__(self):
        return len(self.ann_plain)

    def __getitem__(self, idx):
        ann = self.ann_plain[idx]
        image_path = os.path.join(self.image_root, ann["image_path"])
        caption = ann["caption"]

        x1, y1, x2, y2 = ann["bbox"]
        width = round((x2 - x1) / self.min_image_size_step) * self.min_image_size_step
        height = round((y2 - y1) / self.min_image_size_step) * self.min_image_size_step

        # Load IP images and IP bbox
        target_char_rel_bboxes, source_ip_indices, ip_images = self.load_ip_images(ann)
        
        # Load dialog bbox
        dialog_bbox = []
        frame_bbox = ann["bbox"]
        for idx in np.random.permutation(len(ann["dialogs"])):
            bbox = get_relative_bbox(frame_bbox, ann["dialogs"][idx]["bbox"])
            dialog_bbox.append(bbox)
            if len(dialog_bbox) >= self.max_num_dialogs:
                break

        # Generate MLLM input ids and labels
        instruction = ""
        image_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in range(self.num_img_tokens)]) + EOI_TOKEN
        instruction += self.truncate_caption(caption) + '\n'
        instruction += image_tokens + '\n'

        input_ids = [self.tokenizer_mllm.bos_token_id] + self.tokenizer_mllm.encode(instruction, add_special_tokens=False)

        boi_token_id = self.tokenizer_mllm.encode(BOI_TOKEN, add_special_tokens=False)[1]
        eoi_token_id = self.tokenizer_mllm.encode(EOI_TOKEN, add_special_tokens=False)[1]
        ids_cmp_mask = [False] * len(input_ids)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        ids_cmp_mask = torch.tensor(ids_cmp_mask, dtype=torch.bool)

        boi_idx = torch.where(input_ids == boi_token_id)[0].tolist()
        eoi_idx = torch.where(input_ids == eoi_token_id)[0].tolist()

        for i in range(1):
            ids_cmp_mask[boi_idx[i] + 1 : eoi_idx[i]] = True

        return {
            "image_path": image_path,
            "caption": caption,
            "height": height,
            "width": width,
            "ip_images": ip_images,
            "ip_bbox": target_char_rel_bboxes,
            "source_ip_indices": source_ip_indices,
            "dialog_bbox": dialog_bbox,
            "frame_bbox": ann["bbox"],
            "frame_ann": ann,
            "ann": ann["page_ann"],
            # mllm
            'input_text': instruction,
            'input_ids': input_ids,
            'ids_cmp_mask': ids_cmp_mask,
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
