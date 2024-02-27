
import os
import cv2
import yaml
import json
import numpy as np
from pycocotools.coco import COCO
from terminaltables import AsciiTable, DoubleTable
from typing import Optional
from collections import defaultdict
from tqdm import tqdm
from PIL import Image

class CocoDataset(object):
    def __init__(
            self,
            image_dir: str,
            instances_json: str,
            stuff_json: str=None,
            caption_json: str=None,
            keypoint_json: str=None,
            mode: str='val',
            query_classes: Optional[list]=None,
        ):
        """
        A PyTorch Dataset for loading Coco and Coco-Stuff annotations and converting
        them to scene graphs on the fly.

        Inputs:
        - image_dir: Path to a directory where images are held
        - instances_json: Path to a JSON file giving COCO annotations
        - stuff_json: (optional) Path to a JSON file giving COCO-Stuff annotations
        """
        super(CocoDataset, self).__init__()

        # Load COCO captions
        caption_data = None
        if caption_json is not None and bool(caption_json):
            with open(caption_json, 'r') as f:
                caption_data = json.load(f)
        image_id_to_captions = defaultdict(list)
        for ann in caption_data['annotations']:
            image_id_to_captions[ann['image_id']].append(ann['caption'])

        # reference: https://leimao.github.io/blog/Inspecting-COCO-Dataset-Using-COCO-API/
        coco_annotation = COCO(annotation_file=instances_json)
        # coco_annotation = COCO(annotation_file=keypoint_json)
        self.coco_annotation = coco_annotation

        # Category IDs.
        cat_ids = coco_annotation.getCatIds()
        # print(f"Number of Unique Categories: {len(cat_ids)}")
        # print("Category IDs:")
        # print(cat_ids)  # The IDs are not necessarily consecutive.

        # All categories.
        cats = coco_annotation.loadCats(cat_ids)
        cat_names = [cat["name"] for cat in cats]
        # print("Categories Names:")
        # print(cat_names)

        cat_ids_to_name = {cat_ids[i]: cat_names[i] for i in range(len(cat_ids))}
        # print(yaml.dump(cat_ids_to_name))

        reports = list()
        report_title = ['Query Class', 'Query Class ID', '#Images', '#Original Images']
        self.report_title = report_title
        reports.append(report_title)

        self.query_classes_db = defaultdict(lambda: defaultdict(dict))

        if query_classes is None: query_classes = cat_names

        for query_name in query_classes:
            # Category Name -> Category ID.
            query_id = coco_annotation.getCatIds(catNms=[query_name])[0]
            # print()
            # print(f"Category Name: {query_name}, Category ID: {query_id}")

            # Get the ID of all the images containing the object of the category.
            image_ids = coco_annotation.getImgIds(catIds=[query_id])
            # print(f"Number of Images Containing {query_name}: {len(image_ids)}")

            # customize progress bar style: https://stackoverflow.com/questions/68211844/python-change-tqdm-bar-style
            # for img_id in tqdm(image_ids, desc='create database', ncols=100, ascii="░▒█"):
            for img_id in image_ids:
                captions = image_id_to_captions[img_id]

                img = coco_annotation.imgs[img_id]
                img_height, img_width = img['height'], img['width']

                ann_ids = coco_annotation.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
                ann = coco_annotation.loadAnns(ann_ids)

                # Get the keypoints for all annotated people in the selected image

                # Draw keypoints on the image for all annotated people
                # a number of keypoints is specified in sets of 3, (x, y, v)
                # x and y indicate pixel positions in the image.
                # v indicates visibility— v=0: not labeled (in which case x=y=0), v=1: labeled but not visible, and v=2: labeled and visible 
                # reference https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch

                bboxes = list()
                for a in ann:
                    obj_name = cat_ids_to_name[a['category_id']]
                    bbox = a['bbox']
                    x1, y1, w, h = list(map(float, bbox))

                    ratio = (w * h) / (img_width * img_height)
                    if ratio < 0.03 or ratio > 0.25: continue

                    # normalize
                    x1, w = x1 / img_width, w / img_width
                    y1, h = y1 / img_height, h / img_height

                    # scaling
                    x1, w = x1 * 512, w * 512
                    y1, h = y1 * 512, h * 512

                    bboxes.append(
                        (obj_name, list(map(int, bbox))) # (object_name, [x1, y1, width, height])
                    )
                if not (bool(bboxes)): continue

                self.query_classes_db[query_name][img_id] = dict(
                    caption=captions[0], # use the first caption
                    bboxes=bboxes,
                )
            # for Summary
            reports.append([query_name, query_id, len(self.query_classes_db[query_name]),
                len(image_ids)])

        table = AsciiTable(reports)
        table.justify_columns[1] = 'center'
        table.justify_columns[2] = 'center'
        table.justify_columns[3] = 'center'
        table.title = f' COCO {mode} set '
        print()
        # print(table.table)

