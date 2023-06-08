import argparse

import os
import os.path as osp
import sys
#sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
import time
import cv2
import torch
import glob
import json
import mmcv
import csv
import pandas as pd

from mmdet.apis import inference_detector, init_detector, show_result


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('input_img_dir', type=str, help='the dir of input images')
    parser.add_argument('output_dir', type=str, help='the dir for result images')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--mean_teacher', action='store_true', help='test the mean teacher pth')
    args = parser.parse_args()
    return args

def parse_results(result):
    # Initialize an empty list to store all bounding boxes and scores
    bboxs = result[0]

    bboxes_coords = []
    # For each class's result
    for bbox in bboxs:
        # from an array of five elements, extract the first four
        # which are the bounding box coordinates
        bbox = bbox[:4]

        # append the bounding box coordinates to the list
        bboxes_coords.append(bbox)

    return bboxes_coords




def mock_detector(model, image_name, output_dir):
    image = cv2.imread(image_name)
    results = inference_detector(model, image)
    basename = os.path.basename(image_name).split('.')[0]
    result_name = basename + "_result.jpg"
    result_name = os.path.join(output_dir, result_name)
    show_result(image, results, model.CLASSES, out_file=result_name)

    return {image_name: results}

def create_base_dir(dest):
    basedir = os.path.dirname(dest)
    if not os.path.exists(basedir):
        os.makedirs(basedir)

def run_detector_on_dataset():
    args = parse_args()
    input_dir = args.input_img_dir
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(input_dir)
    eval_imgs = glob.glob(os.path.join(input_dir, '*.png'))
    print(eval_imgs)

    model = init_detector(
        args.config, args.checkpoint, device=torch.device('cuda:0'))
    
    bbox_dic = {}
    prog_bar = mmcv.ProgressBar(len(eval_imgs))
    for im in eval_imgs:
        result = mock_detector(model, im, output_dir)
        result = parse_results(result)
        bbox_dic[im] = result
        prog_bar.update()
    
    
    # Flatten the dictionary into a list of tuples
    data = [(k, v) for k, v in bbox_dic.items()]

    # Convert the list into a DataFrame
    df = pd.DataFrame(data, columns=["image_name", "bboxs_coords"])

    # Write the DataFrame to a CSV file
    df.to_csv(os.path.join(output_dir, 'results.csv'), index=False)

    

if __name__ == '__main__':
    run_detector_on_dataset()