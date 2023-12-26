import base64
import numpy as np
import csv
import sys
import os
import json
import argparse
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ['img_id', 'img_h', 'img_w', 'objects_id', 'objects_conf', 'attrs_id', 'attrs_conf', 'num_boxes', 'boxes',
              'features']

parser = argparse.ArgumentParser(description="Extracting bottom-up features")
parser.add_argument("--input", type=str, default='//data/mmc_wangbing/vg_gqa_obj36.tsv',
                    help="path to bottom-up features")
parser.add_argument("--output", type=str, default='data/extracted_features/',
                    help="path to saving the extracted features")
args = parser.parse_args()

if __name__ == '__main__':
    if not os.path.exists(args.output + 'features/'):
        os.mkdir(args.output + 'features')
    if not os.path.exists(args.output + 'box/'):
        os.mkdir(args.output + 'box')

    obj_info = dict()

    with open(args.input) as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in tqdm(reader):
            img_id = item['img_id']
            obj_info[img_id] = dict()
            num_box, img_h, img_w = int(item['num_boxes']), int(item['img_h']), int(item['img_w'])
            cur_data = np.frombuffer(base64.b64decode(item['features']), dtype=np.float32).reshape((num_box, -1))
            cur_box = np.frombuffer(base64.b64decode(item['boxes']), dtype=np.float32).reshape((num_box, -1)).copy()
            #cur_box[:, 0] /= img_w
            #cur_box[:, 2] /= img_w
            #cur_box[:, 1] /= img_h
            #cur_box[:, 3] /= img_h
            cur_objs = np.frombuffer(base64.b64decode(item['objects_id']), dtype=np.int64).reshape((num_box))
            cur_attrs = np.frombuffer(base64.b64decode(item['attrs_id']), dtype=np.int64).reshape((num_box))
            obj_info[img_id]['objs'] = cur_objs.tolist()
            obj_info[img_id]['attrs'] = cur_attrs.tolist()
            obj_info[img_id]['img_h'] = img_h
            obj_info[img_id]['img_w'] = img_w
            np.save(os.path.join(args.output, 'features', str(img_id)), cur_data)
            np.save(os.path.join(args.output, 'box', str(img_id)), cur_box)

    with open('obj_info.json', 'w') as f:
        json.dump(obj_info, f)
