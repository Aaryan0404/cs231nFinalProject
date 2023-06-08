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

def parse_results(results):
    # Initialize an empty list to store all bounding boxes and scores
    bboxes_scores = []

    # For each class's result
    for class_result in results:
        # For each bounding box
        for bbox in class_result:
            print(f"bbox = {bbox}")  # Debug print
            # The last element is the score, the others are the coordinates
            score = float(bbox[-1])
            bbox = tuple(map(int, bbox[:-1]))

            # Append to the list as a tuple
            bboxes_scores.append((bbox, score))

    return bboxes_scores




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
    
    results_dict = {}
    prog_bar = mmcv.ProgressBar(len(eval_imgs))
    for im in eval_imgs:
        result = mock_detector(model, im, output_dir)
        parsed_results = parse_results(result)
        results_dict[im] = parsed_results
        prog_bar.update()
    
    # Flatten the dictionary into a list of tuples
    data = [(img_name, *res) for img_name, results in results_dict.items() for res in results]

    # Convert the list into a DataFrame
    df = pd.DataFrame(data, columns=["image_name", "bbox", "score"])

    # Write the DataFrame to a CSV file
    df.to_csv(os.path.join(output_dir, 'results.csv'), index=False)

    

if __name__ == '__main__':
    run_detector_on_dataset()