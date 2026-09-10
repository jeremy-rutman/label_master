#!/usr/bin/env python

import json
import os

PATH_TO_GT_FILES: str | None = None
OUT_PATH: str | None = None
CURRENT_IMAGE_NAME_TEMPLATE = "{img_id}.jpg"


def _require_config_value(name: str, configured_value: str | None) -> str:
    if configured_value:
        return configured_value
    env_value = os.environ.get(name)
    if env_value:
        return env_value
    raise RuntimeError(f"Set {name} before running this script.")


def _current_image_name(img_id: int) -> str:
    template = os.environ.get("CURRENT_IMAGE_NAME_TEMPLATE", CURRENT_IMAGE_NAME_TEMPLATE)
    return template.format(img_id=img_id)

if __name__ == '__main__':

    gt_dir = _require_config_value('PATH_TO_GT_FILES', PATH_TO_GT_FILES)
    if not os.path.isdir(gt_dir):
        raise RuntimeError(f'Ground-truth directory does not exist: {gt_dir}')
    gt_list = os.listdir(gt_dir)
    
    out_dir = _require_config_value('OUT_PATH', OUT_PATH)
    os.makedirs(out_dir, exist_ok=True)
    
    for gt in gt_list:
        if gt.endswith('.txt'):
            gt_file = os.path.join(gt_dir, gt)
            
            out_name = gt.replace('.txt','.json')
            out_file = os.path.join(out_dir, out_name)
            
            out_data = {'categories': [],
                        'images': [],
                        'annotations': []}
            
            cat = dict(id=1, name='drone')
            out_data['categories'].append(cat)
            
            width = 1920
            height = 1080
            if 'C000' in gt:
                height = 3840
                width = 1920
            elif 'two_distant' in gt:
                height = 1280
                width = 720
            elif 'custom' in gt or 'swarm' in gt or 'matrice_600' in gt or 'two_parrot' in gt:
                height = 720
                width = 576  
                
            ann_cnt = 0         
                        
            with open(gt_file) as ann:
                line = ann.readline()  
                while line:
                    params = line.split(' ')
                    img_id = int(params[0])
                    obj_cnt = int(params[1])

                    img_info = dict()
                    img_info['id'] = img_id
                    img_info['width'] = width
                    img_info['height'] = height
                    img_info['file_name'] = _current_image_name(img_id)
                    out_data['images'].append(img_info)
                
                    for idx in range(obj_cnt):
                        x_left = int(params[idx*5 + 2])
                        y_top = int(params[idx*5 + 3])
                        w = int(params[idx*5 + 4])
                        h = int(params[idx*5 + 5])
                        cls = params[idx*5 +6]

                        ann_info = dict()
                        ann_info['id'] = ann_cnt
                        ann_info['iscrowd'] = 0
                        ann_info['image_id'] = img_id
                        ann_info['bbox'] = [x_left, y_top, w, h]
                        ann_info['area'] = w*h
                        ann_info['category_id'] = 1
                        out_data['annotations'].append(ann_info)
                
                        ann_cnt += 1
                
                    line = ann.readline()
                            
                                     
            #save out json 
            with open(out_file, 'w') as outfile:
                json.dump(out_data, outfile)




