import re, fileinput, math
import numpy as np
import sys
import caffe
from PIL import Image
import os
import random
import scipy.ndimage
import math
import cv2
from tqdm import tqdm

sys.path.append('../utils');
from evaluation_utils import *

data_path = './data/KITTI/kitti_raw_data/'

gt16bit_dir = './data/KITTI/gt16bit/'

if not os.path.isdir(gt16bit_dir):
    os.makedirs(gt16bit_dir)

max_depth = 80;
img_width = 621;
img_height = 188;

img_files = [];
img_labels = [];
#get the list of rgb images
fid = open('../utils/filenames/eigen_train_files.txt', 'r')
img_lines = fid.readlines();
fid1 = open('../utils/filenames/eigen_train_pairs.txt', 'w')

for in_idx in tqdm(range(len(img_lines))):
    #print('Processing %d-th image ...' % in_idx)
    img_lines0 = img_lines[in_idx].split(' ')[0]
    index_f = str(in_idx+1);
    img_name = index_f.zfill(5) + '.png';

    #load image and depth
    gt_file, gt_calib, im_size, im_file, cams = read_file_data_new(img_lines[in_idx], data_path);
    camera_id = cams[0];
    depth = generate_depth_map(gt_calib[0], gt_file[0], im_size[0], camera_id, False, True);

    im_depth_16 = (depth * 100).astype(np.uint16);
    filename2 = os.path.join(gt16bit_dir, img_name);
    file_line = os.path.join('kitti_raw_data', img_lines0) + ' ' + os.path.join('gt16bit', img_name) + '\n'
    fid1.write(file_line)
    cv2.imwrite(filename2, im_depth_16)
fid.close()
fid1.close()
