# usr/bin/bash -tt
import numpy as np
import matplotlib.pyplot as plt
import os.path
import sys
import scipy
import argparse
import math
import pylab
import string
from PIL import Image
import matplotlib.colors
import caffe
import lmdb
import cv2
from tqdm import tqdm
sys.path.insert(0, './utils')
from fill_depth_hole import fill_depth_colorization

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_def', type=str, default='path/to/model/definition/prototxt');
parser.add_argument('--weights', type=str, default='path/to/trained/caffe/model');
parser.add_argument('--pred_file', type=str, default='path/to/save/predicted/depth/npy/file, e.g. ./output/depth_all.npy');
parser.add_argument('--data_root', type=str, default='path/to/kitti/raw/data');
parser.add_argument('--prediction_blob', type=str, default='name of the final prediction layer output blob, e.g., final_output')
#parser.add_argument('--save_depth_output', type=str, default="False")
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--img_width', type=int, default=621)
parser.add_argument('--img_height', type=int, default=188)


args = parser.parse_args()

caffe.set_mode_gpu()
caffe.set_device(args.gpu)

net = caffe.Net(args.model_def, args.weights, caffe.TEST)

min_depth = 1e-3;
max_depth = 80.0;
num_samples = 697;

fp = open('./utils/filenames/eigen_test_files.txt');
lines = fp.readlines();
image_size = 0;
image_data_dir = '../';
img_mean=[103.939, 116.779, 123.68]
pred_all = np.zeros((num_samples, args.img_height, args.img_width))

for i in tqdm(range(len(lines))):
#for i in range(len(lines)):
    #read image file
    img_ori = cv2.imread(os.path.join(args.data_root, lines[i].split(' ')[0]))
    img_ori = cv2.resize(img_ori, (args.img_width, args.img_height))
    #print img.shape
    img = np.float32(img_ori)
    img = img[:, :, ::-1]  # change to BGR
    img -= img_mean
    img = img.transpose((2, 0, 1))
    net.blobs['data'].reshape(1, 3, img.shape[1], img.shape[2])
    net.blobs['data'].data[0, ...] = img

    net.forward();
    #pred_depth = net.blobs['upscore-map4c'].data
    pred_depth = net.blobs[args.prediction_blob].data
    pred_depth = np.squeeze(pred_depth[0,:,:,:])

    index_f = str(i+1)
    file_name = index_f.zfill(4) + '.png'
    #print('The min and max depth are %d, %d.' % (pred_depth.min(), pred_depth.max())
    pred_depth[pred_depth < min_depth] = min_depth
    pred_depth[pred_depth > max_depth] = max_depth

    #save all depth predicitons to .npy file for evaluation...
    pred_all[i, :, :] = pred_depth      
np.save(args.pred_file, pred_all)
print('Success!')
