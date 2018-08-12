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
parser.add_argument('--save_depth_output', type=str, default="False")
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
       
    if args.save_depth_output == "True":
        #fill the upper part of the depth map
        #pred_depth[0:60, :] = 0
        #img_ori = np.float32(img_ori) / float(255.0)
        #pred_depth_painted = fill_depth_colorization(img_ori, pred_depth)
        #save depth predictions to single files
        reconstruction_results_dir = './output'
        if not os.path.isdir(reconstruction_results_dir):
            os.mkdir(reconstruction_results_dir)
        colormap_dir = './output/colormap/'
        if not os.path.isdir(colormap_dir):
            os.mkdir(colormap_dir)
        graymap_dir = './output/graymap/'
        if not os.path.isdir(graymap_dir):
            os.mkdir(graymap_dir)
        pred_depth_painted = (pred_depth - min_depth) / (max_depth - min_depth);
        #pred_depth_painted = pred_depth_painted - pred_depth_painted.min()
        #pred_depth_painted = pred_depth_painted / pred_depth_painted.max()
        #save gray maps
        #scipy.misc.toimage(255*pred_depth_painted, cmin=0, cmax=255).save(graymap_dir + str(i).zfill(3)+'.png')      
        plt.imsave(graymap_dir + str(i).zfill(3) + '.png', pred_depth_painted, cmap='gray')
        #save color maps
        plt.imsave(colormap_dir + str(i).zfill(3) + '.png', pred_depth_painted, cmap='plasma')
np.save(args.pred_file, pred_all)
print('Success!')
