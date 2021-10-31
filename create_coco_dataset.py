import sys
import os
import cv2
import json
import datetime
import argparse
import pycocotools
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from collections import Counter
from pycocotools.coco import COCO

################################################################################

parser = argparse.ArgumentParser()

parser.add_argument('--config_dir', type=str)
parser.add_argument('--dataset_name', type=str)
parser.add_argument('--image_dir', type=Path)


args = parser.parse_args()


################################################################################

class COCOJsonConverter:
    def __init__(self, image_dir):
        self._category_to_id = {'whistler' : 1}
        self.image_dir = image_dir
        self.info = {
            "year": 2020,
            "version": "1.0",
            "description": "COCO-like dataset for whistler detection",
            "url": "",
            "date_created": str(datetime.datetime.now().year)
        }
        self.licenses = []
        self.images = []
        self.annotations = []
        self.categories = [{"supercategory" : 'wh',
                            "id" : 1,
                            "name" : 'whistler'}]
        
        self.imagenames = list(Path(image_dir).glob('*_x*'))
        self.imagenames = [i.as_posix() for i in self.imagenames]

    def _build_images(self):
        for file_name in self.imagenames:
            im = cv2.imread(file_name)
            image_id = file_name.split('/')[-1]
            height, width, _ = im.shape
                
            image = {
                "license": 1,
                "file_name": file_name,
                "coco_url": "",
                "height": height,
                "width": width,
                "date_captured": "2020",
                "flickr_url": "",
                "id": image_id
            }
            self.images.append(image)

    def _build_annotations(self):
        annotation_id = 0
        for idx, file_name in enumerate(self.imagenames):
            image_id = file_name.split('/')[-1]
            
            annot_names = list(Path(self.image_dir).glob(f"{image_id.replace('x.png', '')}*y*"))
            #print(annot_names)
            for annot_name in annot_names:
                #annot_name = file_name.replace('_x.', '_y.')
                img = cv2.imread(annot_name.as_posix())
                img = (np.asarray(img[:,:,0], order="F") != 0).astype(np.uint8)

                if img.sum() == 0:
                    continue
                    
                seg = pycocotools.mask.encode(img)
                seg['counts'] =  seg['counts'].decode('utf-8')

                row = (img.sum(0) != 0)
                xmin = list(row).index(True)
                xmax = len(row) - list(row)[::-1].index(True) - 1

                col = (img.sum(1) != 0)
                ymin = list(col).index(True)
                ymax = len(col) - list(col)[::-1].index(True) - 1

                anno = {
                    "segmentation": seg,
                    "area": int((ymax - ymin)) * int((xmax - xmin)),
                    "image_id": image_id,
                    "bbox": [xmin, 
                             ymin, 
                             (xmax-xmin), 
                             (ymax-ymin)],
                    "bbox_mode": 1,
                    "category_id": 1,
                    "id": annotation_id
                }
                annotation_id += 1
                self.annotations.append(anno)

    def create_coco_json(self):
        self._build_images()
        self._build_annotations()
        coco_output = {
            "info": self.info,
            "licenses": self.licenses,
            "categories": self.categories,
            "images": self.images,
            "annotations": self.annotations
        }
        return coco_output

################################################################################

converter_train = COCOJsonConverter(image_dir=args.image_dir.as_posix())
jsondata_train  = converter_train.create_coco_json()

with open(f'{args.config_dir}/{args.dataset_name}.json', 'w') as f:
    json.dump(jsondata_train, f)