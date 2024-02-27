
import os
import re
import cv2
import PIL
import time
import json
import torch
import imghdr
import random
import openai
import inflect
import argparse

import numpy as np
import pandas as pd
import os.path as osp
import plotext as plotext
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from tqdm import tqdm
from pathlib import Path
from einops import rearrange
from termcolor import colored
from PIL import Image, ImageDraw
from rich.console import Console
from rich.markdown import Markdown
from torchvision.ops import box_iou
from collections import defaultdict
from typing import Optional, Union
from matplotlib import pyplot as plt
from bounding_box import bounding_box as bb

from llm_guide.bkg_template import bkg_template
from llm_guide.box_template import bbox_template, raw_template, given_prompt
from llm_guide.ratio_template import ratio_template
from llm_guide.coco_data_bbox_retrieval import CocoDataset

# ===== User Specified Regions (start) =====
OPENAI_API_BASE = "https://api.openai.com/v1/" # NOTE: [User specified]
OPENAI_API_KEY = "sk-************************************************" # NOTE: [User specified] For safety reason, I put the placeholder here.
OPENAI_MODEL = 'gpt-3.5-turbo-1106' # NOTE: [User specified]
# OPENAI_MODEL = 'gpt-4-0613' # NOTE: [User specified]: Can use GPT-4 for better compositions
OPENAI_MODEL_FOR_RATIOS = 'gpt-4-0613' # NOTE: [User specified]: Can use GPT-4 for better compositions
# ====== User Specified Regions (end) ======


VALID_IMAGE_EXTENSION = (
    'rgb', 'gif', 'pbm', 'pgm', 'ppm', 'tiff', 'rast',
    'xbm', 'jpeg', 'jpg', 'bmp', 'png', 'webp', 'ext')

color = ['navy', 'blue', 'aqua', 'teal', 'olive', 'green', 'lime', 'yellow',
         'orange', 'red', 'maroon', 'fuchsia', 'purple', 'black', 'gray' ,'silver',
        'navy', 'blue', 'aqua', 'teal', 'olive', 'green', 'lime', 'yellow',
         'orange', 'red', 'maroon', 'fuchsia', 'purple', 'black', 'gray' ,'silver']

def show_report(report_list: list=['> Report']):
    console = Console()

    markdown = Markdown('\n'.join(report_list))
    console.print(markdown)

def get_objects(d):
    # Use the 'max' and 'min' functions to find the key corresponding to the maximum and minimum values in the dictionary.
    # The 'key' argument specifies that the key should be determined based on the values using 'd.get'.
    return max(d, key=d.get), min(d, key=d.get)

def parse_args():
    parser = argparse.ArgumentParser(description='Samples Creator')
    parser.add_argument('--src-dir', default='', help='source directory')
    parser.add_argument('--src-imgs', default='', nargs='+', help='source images')
    parser.add_argument('--dest', default='results', help='dest root')
    parser.add_argument('-img-h', '--img-height', type=int, default=1024, help='background resolution')
    parser.add_argument('-img-w', '--img-width', type=int, default=1024, help='background resolution')
    parser.add_argument('-n', '--num-samples', type=int, default=30, help='number of samples to create')
    parser.add_argument('-i', '--num-iters', type=int, default=1,
        help='number of iterations to sample different objects')
    parser.add_argument('--coco-dir', default='', help='source directory')
    parser.add_argument(
        '-cate', '--categories',
        type=str, default=None, nargs='+')
    parser.add_argument(
        '-nipc', '--num_images_per_category',
        type=int, default=10)
    parser.add_argument('-ci', '--composition-index', type=int, default=1,
        help='Index for naming compositions')
    parser.add_argument('-simg', '--start-img-id', type=int, default=None,
        help='starting image id')
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="+",
        default="cat playing with a ball",
        help="the prompt to render"
    )
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('-mnobj', '--min-num-objects', type=int, default=2,
        help='minmum number of object compositions')
    parser.add_argument('--rotate', action='store_true', help='w/ or w/o rotation')
    parser.add_argument('--hflip', action='store_true', help='w/ or w/o horizontal flip')
    parser.add_argument('--vshift', action='store_true', help='w/ or w/o vertical shift')
    parser.add_argument('--hshift', action='store_true', help='w/ or w/o horizontal shift')
    parser.add_argument('-obj-ratio', '--with-object-ratio', action='store_true', help='Get object ratio or not')
    parser.add_argument(
        "--objects",
        type=str,
        nargs="+",
        default=None,
        help="manual designed objects"
    )
    args = parser.parse_args()
    return args

class Text2Box(object):
    def __init__(
            self,
            img_height: int = 1024,
            img_width: int = 1024,
            bboxes_examples: Optional[list] = None,
        ):
        super(Text2Box, self).__init__()

        self.init_api_settings() # initialize OpenAI API
        self.get_template_resolution() # define image resolution for LLM-Guided template

        self.bboxes_examples = bboxes_examples
        self.img_height, self.img_width = img_height, img_width # output resolution

    def get_template_resolution(self):
        self.tmpl_img_height = 512 # image height for LLM-Guided template
        self.tmpl_img_width = 512 # image width for LLM-Guided template

    def init_api_settings(self):
        openai.api_base = OPENAI_API_BASE
        openai.api_key = OPENAI_API_KEY

    def get_response(self, prompt, model="test", temperature: float=0.):
        """
        Temperature is a number between 0 and 2, with a default value of 1 or 0.7 depending on the model you choose
        """

        messages = [
            {"role": "user", "content": f"""{prompt}"""}
        ]
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=temperature, # this is the degree of randomness of the model's output
            # temperature=0, # this is the degree of randomness of the model's output
        )
        return response.choices[0].message["content"]

    def get_background_description(self, prompt, temperature: float=0.):

        prompt = bkg_template.format(prompt=prompt)
        res = self.get_response(prompt=prompt, temperature=temperature)

        return res

    def get_objects_size_relations(self, prompt, temperature: float=0.):

        prompt = ratio_template.format(prompt=prompt)

        messages = [
            {"role": "user", "content": f"""{prompt}"""}
        ]
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL_FOR_RATIOS,
            messages=messages,
            temperature=temperature, # this is the degree of randomness of the model's output
            # temperature=0, # this is the degree of randomness of the model's output
        )
        res = response.choices[0].message["content"]

        return res

    def get_bounding_boxes(self, prompt, temperature: float=0.):

        if bool(self.bboxes_examples):
            template = raw_template + ''.join(self.bboxes_examples) + given_prompt
            bkg_prompt = ''
        else:
            template = bbox_template

        prompt = template.format(prompt=prompt)

        messages = [
            {"role": "user", "content": f"""{prompt}"""}
        ]
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=temperature, # this is the degree of randomness of the model's output
            # temperature=0, # this is the degree of randomness of the model's output
        )
        res = response.choices[0].message["content"]

        if 'background' in res.lower():
            res = res.splitlines()[0]
            # bkg_prompt = bkg_prompt.replace('Background prompt:', '').strip()
            # bboxes = res

        try:
            bboxes = eval(res)
        except:
            return []
        # res = self.get_response(prompt=prompt, temperature=temperature)
        # try:
        #     if bool(self.bboxes_examples):
        #         bboxes = res
        #     else:
        #         # bboxes = res.splitlines()
        #         bboxes, bkg_prompt = res.splitlines()
        #         bkg_prompt = bkg_prompt.replace('Background prompt:', '').strip()
        #     bboxes = eval(bboxes)
        # except:
        #     return []
        # print(bboxes)

        try:
            H, W = float(self.tmpl_img_height), float(self.tmpl_img_width)

            out_bboxes = list()
            for (class_name, coord) in bboxes:
                x1, y1, box_width, box_height = coord

                coord = np.array(coord).astype(np.float32) # x1, y1, box_width, box_height
                coord[2] += coord[0] # get bottom-right x coordinate
                coord[3] += coord[1] # get bottom-right y coordinate

                # Normalized coordiante by image resolution of template (default: H=W=512)
                coord[[0, 2]] = np.divide(coord[[0, 2]], W)
                coord[[1, 3]] = np.divide(coord[[1, 3]], H)

                # Scale coordinate 
                coord[[0, 2]] *= self.img_width
                coord[[1, 3]] *= self.img_height

                coord = coord.astype(np.int32) # (x1, y1, x2, y2)

                out_bboxes.append((class_name, coord.tolist()))
            return out_bboxes, bkg_prompt
        except:
            return []

def create_compositions(args: argparse.Namespace,
                        src_dir: str,
                        text2box: object,
                        min_num_objects: int = 2):

    # Get output image resolution (default: 1024x1024)
    H, W = args.img_height, args.img_width

    # Get image resolution used for LLM-Guided template (default: 512x512)
    tmpl_H, tmpl_W = text2box.tmpl_img_height, text2box.tmpl_img_width

    # Create output directory if not exists
    Path(args.dest).mkdir(exist_ok=True, parents=True)

    # Initialize object (concept) directionary
    object_dict = defaultdict(dict)

    # Get segmented foreground images with their masks
    for root, dirs, files in os.walk(src_dir, topdown=False):

        # Only keep files with valid image extension
        files = sorted([f for f in files if f.lower().endswith(VALID_IMAGE_EXTENSION)])

        if not bool(files): continue # skip empty folders

        file_dict = defaultdict(list)
        dir_name = Path(root).stem

        for name in files:
            if 'mask' in name and imghdr.what(osp.join(root, name)): # if is mask and valid image format

                object_name = dir_name

                # Remove number
                # e.g., we have the argument `--objects cat1 dog2 houseplant4` 
                # It will become cat dog housepant as the `object_name`
                object_name = re.sub(r'[0-9]+', '', object_name)

                object_name = ' '.join(object_name.split('_')).strip() # also remove beginning/ending whitespaces

                img_name = name.replace('mask', 'foreground') # use mask image as input

                img_path = osp.join(root, img_name)
                mask_path = osp.join(root, name)

                file_dict['object_name'] = object_name
                file_dict['item_paths'].append(dict(img=img_path, mask=mask_path))

        object_dict[dir_name] = file_dict

    out_dict = dict()

    candidates = np.array(args.objects)

    reports, aug_params = list(), list()

    # NOTE object position prior
    pos_prior = [
        'from left to right',
    ]

    # Create output path
    out_dir_name: str = '+'.join(sorted([
        object_dict[c]['object_name'] for c in candidates]))
    out_dir_name: str = out_dir_name.replace(' ', '-')
    out_path: str = osp.join(args.dest, f'{args.composition_index}_{out_dir_name}')
    Path(out_path).mkdir(exist_ok=True, parents=True)

    num_images: list = [i
        for d in os.listdir(args.dest)
            if osp.isdir(osp.join(args.dest, d))
                for i in os.listdir(osp.join(args.dest, d))
                    if 'mask' in i
    ]
    num_images = len(num_images)

    if bool(args.start_img_id):
        img_cnt = args.start_img_id
    else:
        img_cnt = num_images

    out_dict[Path(out_path).stem] = sorted(candidates.tolist())

    # Turn number to words
    num_to_words_engine = inflect.engine()

    # =====================
    # Get background prompt
    # e.g., a background prompt can be: one cat, one dog and one houseplant
    # --------------------- 
    object_names = [object_dict[c]['object_name'] for c in candidates]

    uni_object_names, uni_counts = np.unique(object_names, return_counts=True)
    obj_prompt = ''
    for uni_obj_idx, uni_obj in enumerate(uni_object_names):
        suffix_num = 's' if uni_counts[uni_obj_idx] > 1 else ''

        num_words = num_to_words_engine.number_to_words(uni_counts[uni_obj_idx])
        if uni_obj_idx == len(uni_object_names) - 1:
            obj_prompt += f' and {num_words} {uni_obj}{suffix_num}'
        elif uni_obj_idx == 0:
            obj_prompt += f'{num_words} {uni_obj}{suffix_num}'
        else:
            obj_prompt += f', {num_words} {uni_obj}{suffix_num}'

    bkg_prompt = text2box.get_background_description(
            obj_prompt,
            1.)

    bkg_prompt = bkg_prompt.split('\n') # e.g., ['sky, park, beach', '    Background: in the sky, at the park, at the beach']
    scenes = list(map(lambda string: string.lstrip(), bkg_prompt[0].split(',')))
    bkg_prompt = list(map(lambda string: string.lstrip(),
                          bkg_prompt[1].split(':')[-1].split(',')))

    # print(f"Suggested scenes: {','.join(scenes)}\nThe background prompt: {','.join(bkg_prompt)}")
    report_list = [
        f"> 🛎️  Suggested Scenes",
    ]
    for s_idx, s in enumerate(scenes):
        report_list += [
            f"- {s} ({bkg_prompt[s_idx]})",
        ]
    show_report(report_list)

    # ============================
    # (Optional) Get Object Ratios
    # ---------------------------- 
    if args.with_object_ratio:
        object_ratios = text2box.get_objects_size_relations(
                ', '.join(uni_object_names),
                1.)
        object_ratios = list(map(lambda string: string.lstrip(), object_ratios.split(',')))
        object_relations = {obj_name: float(object_ratios[u_idx])
            for u_idx, obj_name in enumerate(uni_object_names)}

        report_list = [
            f"> ⚖️  Suggested Object Ratios",
        ]
        for u_idx, obj_name in enumerate(uni_object_names):
            report_list += [
                f"- {obj_name} with ratio of {object_ratios[u_idx]}",
            ]
        show_report(report_list)
        biggest_obj_name, _ = get_objects(object_relations)


    # ================
    # Get bounding box
    # ----------------
    for n_sample in tqdm(range(args.num_samples),
                         desc=colored("Perform Object Compositions...", 'blue', attrs=['bold',]),
                         ncols=100, position=1, ascii="░▒█"):

        if n_sample != 0 and n_sample % 20 == 0:
            time.sleep(70) # sleep t seconds (avoid request OpenAI too frequent)

        # Use black background
        img_canvas = Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8))
        mask_canvas = Image.fromarray(np.zeros((H, W), dtype=np.uint8))

        # NOTE: Temperature is a number between 0 and 2,
        # with a default value of 1 or 0.7 depending on the model you choose
        temperature: float = np.random.uniform(0, 1)

        prompt: str = '{}'.format(np.random.choice(pos_prior))

        # Randomly select one background prompt
        prompt: str = obj_prompt + f' {np.random.choice(bkg_prompt)}'

        # Get Bounding Boxes
        res = text2box.get_bounding_boxes(
                prompt, temperature) # box in (x_min, y_min, x_max, y_max)

        # Generate next composition if returning an empty result
        if not bool(res):
            print(f'⚠️  Empty bounding box returned from LLM model w/ returned result: {res}')
            continue
        else: bboxes, _ = res

        classes: list = object_names
        bboxes: list = [b for (_, b) in bboxes]

        # At least return >= min_num_objects
        if len(bboxes) < min_num_objects:
            print('❗ Number of returned boxes not match to the number of given objects')
            continue

        # Skip the generated boxes that outside the range of image resolution
        out_of_region_box = np.concatenate([
            np.where(np.array(bboxes)[:, [0, 2]].ravel() > W)[0],
            np.where(np.array(bboxes)[:, [1, 3]].ravel() > H)[0]])
        if out_of_region_box.size != 0: continue

        # Skip the generated boxes having all zeros coordinate
        zero_coord = any([
            np.any(~np.any(np.array(bboxes)[:, [0, 2]], axis=1)), # (x1, x2) :Nonzero numbers are considered True, and 0 is considered False.
            np.any(~np.any(np.array(bboxes)[:, [1, 3]], axis=1)), # (y1, y2) :Nonzero numbers are considered True, and 0 is considered False.
        ])
        if zero_coord: continue

        # Paste largest object first, and then paste other smaller objects
        ious = box_iou(torch.Tensor(bboxes), torch.Tensor(bboxes))
        ious = ious * (1 - torch.eye(*ious.shape))
        ious = ious.sum(dim=-1)
        paste_order = torch.argsort(ious, descending=True).numpy()

        invalid = False

        # Preprocessing - Make sure box size is larger than minimum requirement
        all_boxes_area: list = list()
        for obj_idx in paste_order:
            try:
                class_name, region, key = classes[obj_idx], bboxes[obj_idx], candidates[obj_idx]
            except:
                invalid = True
                break

            # Get bounding boxes upper-left and lower-right coordinates from LLM model
            llm_box_x1, llm_box_y1, llm_box_x2, llm_box_y2 = region

            # Get bounding boxes centers, where boxes are generated by LLM model
            llm_box_cent_x = llm_box_x1 + (llm_box_x2 - llm_box_x1) / 2.
            llm_box_cent_y = llm_box_y1 + (llm_box_y2 - llm_box_y1) / 2.

            # Scale geenerated bounding boxes to match the real object ratio in the world
            if args.with_object_ratio:
                biggest_obj_idx = classes.index(biggest_obj_name)
                biggest_obj_bbox = bboxes[biggest_obj_idx]
                llm_box_big_x1, llm_box_big_y1, llm_box_big_x2, llm_box_big_y2 = biggest_obj_bbox

                llm_box_big_width = (llm_box_big_x2 - llm_box_big_x1) * object_relations[class_name]
                llm_box_big_height = (llm_box_big_y2 - llm_box_big_y1) * object_relations[class_name]

                llm_box_x1 = int(llm_box_cent_x - llm_box_big_width / 2.)
                llm_box_x2 = int(llm_box_cent_x + llm_box_big_width / 2.)
                llm_box_y1 = int(llm_box_cent_y - llm_box_big_height / 2.)
                llm_box_y2 = int(llm_box_cent_y + llm_box_big_height / 2.)

            # Adjust x coordinates
            if llm_box_x1 < 0:
                llm_box_x2 += llm_box_x1  # Adjust llm_box_x2 left by the amount llm_box_x1 is out of bounds
                llm_box_x1 = 0
            if llm_box_x2 > W - 1:
                llm_box_x1 = max(llm_box_x1 - (llm_box_x2 - (W - 1)), 0)  # Adjust x1 without going out of bounds
                llm_box_x2 = W - 1

            # Adjust y coordinates
            if llm_box_y1 < 0:
                llm_box_y2 += llm_box_y1  # Adjust llm_box_y2 up by the amount llm_box_y1 is out of bounds
                llm_box_y1 = 0
            if llm_box_y2 > H - 1:
                llm_box_y1 = max(llm_box_y1 - (llm_box_y2 - (H - 1)), 0)  # Adjust y1 without going out of bounds
                llm_box_y2 = H - 1

            llm_box_width, llm_box_height = llm_box_x2 - llm_box_x1, llm_box_y2 - llm_box_y1

            all_boxes_area.append(llm_box_width * llm_box_height)

        if invalid: continue

        min_box_area = np.min(all_boxes_area)
        min_box_area_in_canvas: float = (H / 4) * (W / 4) if len(classes) <= 3 else \
                                    (H / 6) * (W / 6) if len(classes) == 4 else (H / 8) * (W / 8)
        if min_box_area < min_box_area_in_canvas:
            box_canvas_ratio: float = min_box_area_in_canvas / min_box_area
        else:
            box_canvas_ratio: float = 1.

        box_canvas_ratio: list = [box_canvas_ratio] * len(classes) # expand

        invalid = False
        final_bbox = list()
        for obj_idx in paste_order:

            # =====================================
            #  Obtain bounding boxes from LLM model
            # =====================================

            try:
                class_name, region, key = classes[obj_idx], bboxes[obj_idx], candidates[obj_idx]
            except:
                invalid = True
                print(f'[Invalid] Retrieved numbers of boxes are {len(bboxes)} while given number of objects are {len(classes)}')
                break

            # Get bounding boxes upper-left and lower-right coordinates from LLM model
            llm_box_x1, llm_box_y1, llm_box_x2, llm_box_y2 = region

            # Get bounding boxes centers, where boxes are generated by LLM model
            llm_box_cent_x = llm_box_x1 + (llm_box_x2 - llm_box_x1) / 2.
            llm_box_cent_y = llm_box_y1 + (llm_box_y2 - llm_box_y1) / 2.

            # Scale geenerated bounding boxes to match the real object ratio in the world
            if args.with_object_ratio:
                biggest_obj_idx: int = classes.index(biggest_obj_name)
                biggest_obj_bbox = bboxes[biggest_obj_idx]
                llm_box_big_x1, llm_box_big_y1, llm_box_big_x2, llm_box_big_y2 = biggest_obj_bbox

                llm_box_big_width = (llm_box_big_x2 - llm_box_big_x1) * object_relations[class_name]
                llm_box_big_height = (llm_box_big_y2 - llm_box_big_y1) * object_relations[class_name]

                llm_box_x1 = int(llm_box_cent_x - llm_box_big_width / 2.)
                llm_box_x2 = int(llm_box_cent_x + llm_box_big_width / 2.)
                llm_box_y1 = int(llm_box_cent_y - llm_box_big_height / 2.)
                llm_box_y2 = int(llm_box_cent_y + llm_box_big_height / 2.)

            # Adjust x coordinates
            if llm_box_x1 < 0:
                llm_box_x2 += llm_box_x1  # Adjust llm_box_x2 right by the amount llm_box_x1 is out of bounds
                llm_box_x1 = 0
            if llm_box_x2 > W - 1:
                llm_box_x1 = max(llm_box_x1 - (llm_box_x2 - (W - 1)), 0)  # Adjust x1 without going out of bounds
                llm_box_x2 = W - 1

            # Adjust y coordinates
            if llm_box_y1 < 0:
                llm_box_y2 += llm_box_y1  # Adjust llm_box_y2 down by the amount llm_box_y1 is out of bounds
                llm_box_y1 = 0
            if llm_box_y2 > H - 1:
                llm_box_y1 = max(llm_box_y1 - (llm_box_y2 - (H - 1)), 0)  # Adjust y1 without going out of bounds
                llm_box_y2 = H - 1

            llm_box_width, llm_box_height = llm_box_x2 - llm_box_x1, llm_box_y2 - llm_box_y1

            llm_box_width *= box_canvas_ratio[obj_idx]
            llm_box_height *= box_canvas_ratio[obj_idx]

            print(llm_box_width, llm_box_height)

            llm_box_width, llm_box_height = int(llm_box_width), int(llm_box_height)

            object_candidates = np.arange(len(object_dict[key]['item_paths']))
            sel_idx: int = np.random.choice(object_candidates)
            item = object_dict[key]['item_paths'][sel_idx]

            img = Image.open(item['img']).convert('RGB')
            mask = Image.open(item['mask'])

            # Create augmentation probabilities
            shift_vp, shift_hp, rotate_p, flip_p = np.random.rand(4)

            # * Augmentation - Rotation (default: no rotation)
            if rotate_p > 0.5 and args.rotate:
                # apply a functional transform with the same parameters to multiple images
                # ref: https://pytorch.org/vision/0.15/transforms.html
                # angle = np.random.randint(-15, 15)
                angle = np.random.randint(-5, 5)
                # print(key, class_name, angle)
                img = TF.rotate(img, angle) # in PIL.Image.Image format
                mask = TF.rotate(mask, angle) # in PIL.Image.Image format
            else:
                angle = 0 # for recording augmentation parameters

            # * Augmentation - Horizontal flip (defulat: no horizonatal flip)
            if flip_p > 0.5 and args.hflip:
                img = TF.hflip(img) # in PIL.Image.Image format
                mask = TF.hflip(mask) # in PIL.Image.Image format
                h_flip = 1
            else:
                h_flip = 0 # for recording augmentation parameters

            # =========================================
            #  Obtain bounding boxes from source images
            # =========================================

            # Obtain boudning boxes from mask of source images
            ret, binary_mask = cv2.threshold(np.array(mask), 70, 255, 0) # the channel of mask = 1

            white_pixels = np.where(binary_mask == 255)
            y_min, y_max = np.min(white_pixels[0]), np.max(white_pixels[0])
            x_min, x_max = np.min(white_pixels[1]), np.max(white_pixels[1])

            # Object bounding box size from source images
            # NOTE: src_obj_height and src_obj_width mean box size from source images
            # NOTE: llm_box_witdh and llm_box_height mean box size generated from LLM model
            src_obj_height, src_obj_width = y_max - y_min, x_max - x_min

            # Ignore those generated boxes with zero llm_box_width and llm_box_height
            if (llm_box_width == 0) or (llm_box_height == 0) or (src_obj_width == 0) or (src_obj_height == 0):
                invalid = True
                print('[Invalid] since width or height = 0')
                print(f'\t LLM box width: {llm_box_width} / LLM box height: {llm_box_height}')
                print(f'\t Source object width: {src_obj_width} / Source object height: {src_obj_height}')
                break

            # Maintain the original aspect ratio
            # src_obj_width, src_obj_height are the dimensions of the source object
            # llm_box_width, llm_box_height are the initial dimensions of the bounding box

            # Adjust bounding box dimensions based on the aspect ratio of the source object
            if src_obj_width >= src_obj_height:
                new_width: int = min(llm_box_width, llm_box_height)

                wpercent: float = new_width / float(src_obj_width)
                hsize = int((float(src_obj_height) * float(wpercent)))

                # Scale the height of the bounding box to maintain the source object's aspect ratio
                llm_box_width, llm_box_height = new_width, hsize
            else:  # Source object is taller
                new_height: int = min(llm_box_width, llm_box_height)

                # hpercent = llm_box_height / float(src_obj_height)
                hpercent: float = new_height / float(src_obj_height)
                wsize = int((float(src_obj_width) * float(hpercent)))

                # Scale the width of the bounding box to maintain the source object's aspect ratio
                llm_box_width, llm_box_height = wsize, new_height

            llm_box_cent_x: int = llm_box_x1 + llm_box_width // 2
            llm_box_cent_y: int = llm_box_y1 + llm_box_height // 2

            llm_box_x1: int = llm_box_cent_x - llm_box_width // 2
            llm_box_x2: int = llm_box_cent_x + llm_box_width // 2
            llm_box_y1: int = llm_box_cent_y - llm_box_height // 2
            llm_box_y2: int = llm_box_cent_y + llm_box_height // 2

            # Adjust x coordinates
            if llm_box_x1 < 0:
                llm_box_x2 += llm_box_x1  # Adjust llm_box_x2 right by the amount llm_box_x1 is out of bounds
                llm_box_x1 = 0
            if llm_box_x2 > W - 1:
                llm_box_x1 = max(llm_box_x1 - (llm_box_x2 - (W - 1)), 0)  # Adjust x1 without going out of bounds
                llm_box_x2 = W - 1

            # Adjust y coordinates
            if llm_box_y1 < 0:
                llm_box_y2 += llm_box_y1  # Adjust llm_box_y2 down by the amount llm_box_y1 is out of bounds
                llm_box_y1 = 0
            if llm_box_y2 > H - 1:
                llm_box_y1 = max(llm_box_y1 - (llm_box_y2 - (H - 1)), 0)  # Adjust y1 without going out of bounds
                llm_box_y2 = H - 1

            if llm_box_width == 0 and llm_box_height == 0:
                invalid = True
                print('[Invalid] since both width and height from LLM generated box = 0')
                break

            # * Augmentation: Shift vertically (default: no vertical shift)
            if shift_vp and args.vshift: # shift vertically
                v_shift: int = np.random.randint(-H // 100, H // 100)

                bak_box_y1, bak_box_y2 = llm_box_y1, llm_box_y2

                llm_box_y1, llm_box_y2 = np.clip(llm_box_y1 + v_shift, 0, H), np.clip(llm_box_y2 + v_shift, 0, H)

                if llm_box_y1 == llm_box_y2:
                    llm_box_y1 = bak_box_y1
                    llm_box_y2 = bak_box_y2
            else:
                v_shift = 0 # for recording augmentation parameters

            # * Augmentation: Shift horizontally (default: no horizontal shift)
            if shift_hp and args.hshift: # shift horizontally
                h_shift: int = np.random.randint(-W // 100, W // 100)

                bak_box_x1, bak_box_x2 = llm_box_x1, llm_box_x2

                llm_box_x1, llm_box_x2 = np.clip(llm_box_x1 + h_shift, 0, W), np.clip(llm_box_x2 + h_shift, 0, W)

                if llm_box_x1 == llm_box_x2:
                    llm_box_x1 = bak_box_x1
                    llm_box_x2 = bak_box_x2
            else:
                h_shift = 0 # for recording

            # Obtain source object and mask
            img: PIL.Image = img.crop((x_min, y_min, x_max, y_max))
            mask: PIL.Image = mask.crop((x_min, y_min, x_max, y_max))

            try:
                img: PIL.Image = img.resize((llm_box_x2 - llm_box_x1, llm_box_y2 - llm_box_y1)) # (llm_box_width, llm_box_height)
                mask: PIL.Image = mask.resize((llm_box_x2 - llm_box_x1, llm_box_y2 - llm_box_y1)) # (llm_box_width, llm_box_height)

                # Paste object onto canvas
                img_canvas.paste(img, (llm_box_x1, llm_box_y1), mask) # x and y coords in upper left (x1, y1)
                mask_canvas.paste(mask, (llm_box_x1, llm_box_y1), mask) # x and y coords in upper left (x1, y1)

                final_bbox.append([{f'{class_name}': dict(x1=int(llm_box_x1),
                                                          y1=int(llm_box_y1),
                                                          x2=int(llm_box_x2),
                                                          y2=int(llm_box_y2))}])
            except:
                invalid = True
                print(f'[Invalid] Wrong bounding box coordinates: x1: {llm_box_x1}, y1: {llm_box_y1}, x2: {llm_box_x2}, y2: {llm_box_y2}')
                break

        if invalid: continue

        out_name = f'{img_cnt+1:06d}.png'
        img_canvas.save(osp.join(out_path, f'img_{out_name}'))
        mask_canvas.save(osp.join(out_path, f'mask_{out_name}'))

        reports.append(['', Path(out_path).stem, f'img_{out_name}',
                        ','.join(scenes),
                        ','.join(bkg_prompt)])
        aug_params.append([
            {f"img_{out_name}": dict(
                temperature=temperature,
                vertial_shift=v_shift,
                horizontal_shift=h_shift,
                rotation_angle=angle,
                is_horizontal_flip=h_flip,
                bbox=final_bbox,
            )}
        ])
        img_cnt += 1

    df = pd.DataFrame(reports,
                      columns=['check', 'directory', 'image_id', 'background_scenes', 'background_prompt'])
    df.to_csv(osp.join(args.dest, f'{Path(out_path).stem}_report.csv'), index=False)

    with open(osp.join(args.dest, f'{Path(out_path).stem}_params.json'), 'w') as fp:
        json.dump(aug_params, fp)

    with open(osp.join(args.dest, f'{Path(out_path).stem}_directory_classes.json'), 'w') as fp:
        json.dump(out_dict, fp)

def main():
    args = parse_args()

    # Fix random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # If using COCO annotations or not
    if bool(args.coco_dir):
        coco_val_cfg = dict(
            image_dir='images/val2017',
            instances_json='annotations/instances_val2017.json',
            caption_json='annotations/captions_val2017.json',
            deprecated_stuff_ids_txt='',
            specific_image_ids=[ ],
        )
        coco_dataset = CocoDataset(
            image_dir=osp.join(args.coco_dir, coco_val_cfg['image_dir']),
            instances_json=osp.join(args.coco_dir, coco_val_cfg['instances_json']),
            caption_json=osp.join(args.coco_dir, coco_val_cfg['caption_json']),
            query_classes=args.categories,
        )
        query_classes_db = coco_dataset.query_classes_db

        bboxes_examples = list()

        space = ''.join([' '] * 4) # 4 spaces

        # Use all categories from COCO dataset
        for cls in list(query_classes_db.keys()):
            img_ids = list(query_classes_db[cls].keys())
            if not bool(img_ids): continue
            num_selected_samples = min(len(img_ids), args.num_images_per_category)
            img_ids = np.random.choice(img_ids, num_selected_samples, replace=False)

            for idx in img_ids:
                cls_dict = query_classes_db[cls][idx]
                caption = cls_dict['caption']

                # Turn bboxes to new height and width
                bbox = [b for b in cls_dict['bboxes']]

                bbox = json.dumps(bbox) # turn list to string

                example = f'\n{space}Caption: {caption}\n{space}Objects: {bbox}\n'
                bboxes_examples.append(example)

        bboxes_examples = np.random.choice(bboxes_examples, 50, replace=False)
        bboxes_examples = bboxes_examples.tolist()
    else:
        bboxes_examples = None

    print()
    console = Console()
    console.rule("[bold blue]2️⃣  Step 2: LLM-Guided Object Composition")

    text2box = Text2Box(img_height=args.img_height,
                        img_width=args.img_width,
                        bboxes_examples=bboxes_examples)

    create_compositions(args, args.src_dir, text2box, min_num_objects=args.min_num_objects)

if __name__ == '__main__':
    main()

