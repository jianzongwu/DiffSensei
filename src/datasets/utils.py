import numpy as np
from PIL import Image, ImageDraw
import re


size_buckets = [
    {
        "size": 256,
        "buckets": [
            [128, 512, 0.25],
            [128, 496, 0.25806451612903225],
            [136, 480, 0.2833333333333333],
            [144, 464, 0.3103448275862069],
            [144, 448, 0.32142857142857145],
            [152, 432, 0.35185185185185186],
            [160, 416, 0.38461538461538464],
            [160, 400, 0.4],
            [168, 384, 0.4375],
            [176, 368, 0.4782608695652174],
            [184, 352, 0.5227272727272727],
            [192, 336, 0.5714285714285714],
            [208, 320, 0.65],
            [216, 304, 0.7105263157894737],
            [224, 288, 0.7777777777777778],
            [240, 272, 0.8823529411764706],
            [256, 256, 1.0],
            [272, 240, 1.1333333333333333],
            [288, 224, 1.2857142857142858],
            [304, 216, 1.4074074074074074],
            [320, 208, 1.5384615384615385],
            [336, 192, 1.75],
            [352, 184, 1.9130434782608696],
            [368, 176, 2.090909090909091],
            [384, 168, 2.2857142857142856],
            [400, 160, 2.5],
            [416, 160, 2.6],
            [432, 152, 2.8421052631578947],
            [448, 144, 3.111111111111111],
            [464, 144, 3.2222222222222223],
            [480, 136, 3.5294117647058822],
            [496, 128, 3.875],
            [512, 128, 4.0]
        ]
    },
    {
        "size": 512,
        "buckets": [
            [256, 1024, 0.25],
            [256, 992, 0.25806451612903225],
            [272, 960, 0.2833333333333333],
            [288, 928, 0.3103448275862069],
            [288, 896, 0.32142857142857145],
            [304, 864, 0.35185185185185186],
            [320, 832, 0.38461538461538464],
            [320, 800, 0.4],
            [336, 768, 0.4375],
            [352, 736, 0.4782608695652174],
            [368, 704, 0.5227272727272727],
            [384, 672, 0.5714285714285714],
            [416, 640, 0.65],
            [432, 608, 0.7105263157894737],
            [448, 576, 0.7777777777777778],
            [480, 544, 0.8823529411764706],
            [512, 512, 1.0],
            [544, 480, 1.1333333333333333],
            [576, 448, 1.2857142857142858],
            [608, 432, 1.4074074074074074],
            [640, 416, 1.5384615384615385],
            [672, 384, 1.75],
            [704, 368, 1.9130434782608696],
            [736, 352, 2.090909090909091],
            [768, 336, 2.2857142857142856],
            [800, 320, 2.5],
            [832, 320, 2.6],
            [864, 304, 2.8421052631578947],
            [896, 288, 3.111111111111111],
            [928, 288, 3.2222222222222223],
            [960, 272, 3.5294117647058822],
            [992, 256, 3.875],
            [1024, 256, 4.0]
        ]
    },
    {
        "size": 1024,
        "buckets": [
            [512, 2048, 0.25],
            [512, 1984, 0.25806451612903225],
            [544, 1920, 0.2833333333333333],
            [576, 1856, 0.3103448275862069],
            [576, 1792, 0.32142857142857145],
            [608, 1728, 0.35185185185185186],
            [640, 1664, 0.38461538461538464],
            [640, 1600, 0.4],
            [672, 1536, 0.4375],
            [704, 1472, 0.4782608695652174],
            [736, 1408, 0.5227272727272727],
            [768, 1344, 0.5714285714285714],
            [832, 1280, 0.65],
            [864, 1216, 0.7105263157894737],
            [896, 1152, 0.7777777777777778],
            [960, 1088, 0.8823529411764706],
            [1024, 1024, 1.0],
            [1088, 960, 1.1333333333333333],
            [1152, 896, 1.2857142857142858],
            [1216, 864, 1.4074074074074074],
            [1280, 832, 1.5384615384615385],
            [1344, 768, 1.75],
            [1408, 736, 1.9130434782608696],
            [1472, 704, 2.090909090909091],
            [1536, 672, 2.2857142857142856],
            [1600, 640, 2.5],
            [1664, 640, 2.6],
            [1728, 608, 2.8421052631578947],
            [1792, 576, 3.111111111111111],
            [1856, 576, 3.2222222222222223],
            [1920, 544, 3.5294117647058822],
            [1984, 512, 3.875],
            [2048, 512, 4.0]
        ]
    }
]


size_buckets_flux = [
    {
        "size": 256,
        "buckets": [
            [128, 512, 0.25],
            [128, 480, 0.2833333333333333],
            [160, 448, 0.3571428571428571],
            [160, 416, 0.3846153846153846],
            [192, 384, 0.5],
            [192, 352, 0.5454545454545454],
            [224, 320, 0.7],
            [224, 288, 0.7777777777777778],
            [256, 256, 1.0],
            [288, 224, 1.285714285714286],
            [320, 224, 1.428571428571429],
            [352, 192, 1.833333333333333],
            [384, 192, 2.0],
            [416, 160, 2.6],
            [448, 160, 2.8],
            [480, 128, 3.75],
            [512, 128, 4.0],
        ]
    },
    {
        "size": 512,
        "buckets": [
            [256, 1024, 0.25],
            [256, 992, 0.25806451612903225],
            [272, 960, 0.2833333333333333],
            [288, 928, 0.3103448275862069],
            [288, 896, 0.32142857142857145],
            [304, 864, 0.35185185185185186],
            [320, 832, 0.38461538461538464],
            [320, 800, 0.4],
            [336, 768, 0.4375],
            [352, 736, 0.4782608695652174],
            [368, 704, 0.5227272727272727],
            [384, 672, 0.5714285714285714],
            [416, 640, 0.65],
            [432, 608, 0.7105263157894737],
            [448, 576, 0.7777777777777778],
            [480, 544, 0.8823529411764706],
            [512, 512, 1.0],
            [544, 480, 1.1333333333333333],
            [576, 448, 1.2857142857142858],
            [608, 432, 1.4074074074074074],
            [640, 416, 1.5384615384615385],
            [672, 384, 1.75],
            [704, 368, 1.9130434782608696],
            [736, 352, 2.090909090909091],
            [768, 336, 2.2857142857142856],
            [800, 320, 2.5],
            [832, 320, 2.6],
            [864, 304, 2.8421052631578947],
            [896, 288, 3.111111111111111],
            [928, 288, 3.2222222222222223],
            [960, 272, 3.5294117647058822],
            [992, 256, 3.875],
            [1024, 256, 4.0]
        ]
    }
]


character_indices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]


def get_bucket_size(height, width, size_buckets):
    image_size = height * width
    image_ratio = height / width

    sizes = [sb["size"]**2 for sb in size_buckets]
    size_diffs = [abs(image_size - size) for size in sizes]
    closest_index_size = size_diffs.index(min(size_diffs))

    buckets = np.array(size_buckets[closest_index_size]["buckets"])
    aspect_ratios = buckets[:, -1]
    aspect_ratio_diffs = [abs(image_ratio - ratio) for ratio in aspect_ratios]
    closest_index = aspect_ratio_diffs.index(min(aspect_ratio_diffs))

    bucket = buckets[closest_index]

    return int(bucket[0]), int(bucket[1]), closest_index_size


def resize_and_center_crop(image, bucket_size):
    wA, hA = image.size
    hB, wB = bucket_size

    if hA / wA >= hB / wB:
        new_hA = int(hA * wB / wA)
        new_wA = wB
    else:
        new_hA = hB
        new_wA = int(wA * hB / hA)

    resized_image = image.resize((new_wA, new_hA), Image.BICUBIC)

    left = (new_wA - wB) // 2
    top = (new_hA - hB) // 2
    right = left + wB
    bottom = top + hB

    cropped_image = resized_image.crop((left, top, right, bottom))

    return cropped_image, (top, left)


def resize_and_pad(image, target_size=1024):
    # Resize the image, keeping the aspect ratio, so that the longer edge is 1024 pixels
    image.thumbnail((target_size, target_size), Image.BICUBIC)
    
    # Calculate padding to make the image square
    width, height = image.size
    pad_width = (target_size - width) // 2 if width < target_size else 0
    pad_height = (target_size - height) // 2 if height < target_size else 0
    
    # Create a new image with white background
    new_image = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    
    # Paste the resized image onto the white background
    new_image.paste(image, (pad_width, pad_height))
    
    return new_image


def center_crop_and_resize(image, original_width, original_height):
    width, height = image.size
    aspect_ratio = original_width / original_height

    if original_width > original_height:
        new_width = width
        new_height = int(new_width / aspect_ratio)
        padding = (height - new_height) // 2
        cropped_image = image.crop((0, padding, width, height - padding))
    else:
        new_height = height
        new_width = int(new_height * aspect_ratio)
        padding = (width - new_width) // 2
        cropped_image = image.crop((padding, 0, width - padding, height))

    resized_image = cropped_image.resize((original_width, original_height), Image.BICUBIC)

    return resized_image


def get_relative_bbox(bbox_bg, bbox_fg):
    bg_x1, bg_y1, bg_x2, bg_y2 = bbox_bg
    fg_x1, fg_y1, fg_x2, fg_y2 = bbox_fg

    bg_width = bg_x2 - bg_x1
    bg_height = bg_y2 - bg_y1

    # Calculate the relative coordinates of bbox_fg within bbox_bg
    rel_x1 = (fg_x1 - bg_x1) / bg_width
    rel_y1 = (fg_y1 - bg_y1) / bg_height
    rel_x2 = (fg_x2 - bg_x1) / bg_width
    rel_y2 = (fg_y2 - bg_y1) / bg_height

    return [rel_x1, rel_y1, rel_x2, rel_y2]


def get_page_bbox(frame_bbox, frame_info):
    x1, y1, x2, y2 = frame_bbox
    x1_frame, y1_frame, _, _ = frame_info["bbox"]
    
    return [x1 + x1_frame, y1 + y1_frame, x2 + x1_frame, y2 + y1_frame]


def get_page_bbox_from_rel_bbox(rel_bbox, frame_bbox):
    x1, y1, x2, y2 = frame_bbox
    x1_rel, y1_rel, x2_rel, y2_rel = rel_bbox
    frame_width = x2 - x1
    frame_height = y2 - y1
    
    return [round(x1 + x1_rel * frame_width), round(y1 + y1_rel * frame_height), round(x1 + x2_rel * frame_width), round(y1 + y2_rel * frame_height)]


def get_cropped_ip_images_from_relative_bbox(image, relative_bbox):
    """
    Args:
        image: PIL.Image
        relative_bbox: List[[x1, y1, x2, y2]].
    Return:
        List[PIL.Image]. The ip_images cropped from the relative bbox
    """
    image_width, image_height = image.size
    cropped_images = []
    
    for rel_bbox in relative_bbox:
        rel_x1, rel_y1, rel_x2, rel_y2 = rel_bbox
        
        abs_x1 = int(rel_x1 * image_width)
        abs_y1 = int(rel_y1 * image_height)
        abs_x2 = int(rel_x2 * image_width)
        abs_y2 = int(rel_y2 * image_height)

        abs_x1 = max(0, min(abs_x1, image_width))
        abs_y1 = max(0, min(abs_y1, image_height))
        abs_x2 = max(0, min(abs_x2, image_width))
        abs_y2 = max(0, min(abs_y2, image_height))
        
        cropped_image = image.crop((abs_x1, abs_y1, abs_x2, abs_y2))
        cropped_images.append(cropped_image)

    return cropped_images


def mask_dialogs_from_image(image, ann):
    draw = ImageDraw.Draw(image)
    
    for frame_info in ann["frames"]:
        for dialog in frame_info["dialogs"]:
            bbox = dialog["bbox"] # x1, y1, x2, y2
            x1, y1, x2, y2 = bbox
            
            # Draw a white rectangle over the bounding box area
            draw.rectangle([x1, y1, x2, y2], fill="white")

    return image


def sort_manga_panels(ann, width, threshold=100):
    """
    Args:
        ann: annotations of a manga page. A page contains multiple panels, in ann["frames"], 
             each frame has a "bbox" attribute to indicate its position in the page.
        width: Page width.
        threshold: The maximum difference in y_min to consider panels as being on the same line.
    Returns:
        The sorted panels in manga reading order.
    """
    frames = ann["frames"]
    
    # Split frames into left half and right half
    left_half = []
    right_half = []
    
    for frame in frames:
        x_min = frame["bbox"][0]  # x_min coordinate of the bounding box
        if x_min < width / 2 - threshold:
            left_half.append(frame)
        else:
            right_half.append(frame)
    
    # Sorting function for reading order: from right to left, top to bottom
    def sort_key(frame):
        x_min, y_min, x_max, y_max = frame["bbox"]
        return (round(y_min / threshold), -x_min)

    # Sort each half using the defined reading order with "soft" y-min handling
    left_half_sorted = sorted(left_half, key=sort_key)
    right_half_sorted = sorted(right_half, key=sort_key)

    # Combine the sorted halves: left half first, then right half
    sorted_panels = left_half_sorted + right_half_sorted

    return sorted_panels
