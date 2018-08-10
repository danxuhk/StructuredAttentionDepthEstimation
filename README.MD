## Structured Attention Guided Convolutional Neural Fields for Monocular Depth Estimation in CVPR 2018  <br>
The code for the above paper which can be downloaded form the link: https://arxiv.org/abs/1803.11029 <br>
By Dan Xu, Wei Wang, Hao Tang, Hong Liu, Nicu Sebe, Elisa Ricci
<p align="center">
  <img src="figures/framework.jpg" width="760"/>
</p>

### Experimental Setup <br>
The code is implemented based on the Caffe framework. Please first download and install the modified caffe version. The code is tested on CUDA 8.0, cudnn 5.1, and python 2.7. The installation can follow the following instructions: <br>
First clone the repository: 
<pre>
git clone https://github.com/danxuhk/StructuredAttentionDepthEstimation.git 
</pre>
Then build caffe and pycaffe:
<pre>
cd $StructuredAttentionDepthEstimation_ROOT
cp Makefile.config.example Makefile.config
vim Makefile.config %%%% changing neccessary lines
sh install.sh
</pre>

### Data Preparation <br>
First download KITTI raw data from the official website http://www.cvlibs.net/datasets/kitti/. Then go to ./data_preparation, and run save_16bitpng_gt.py (changing the path of saving the kitti raw data and the path for saving the generated 16bit depth data), which will generate a training pair text files under ./utils/filenames for use in the training phase. For testing, the eigen split of 697 images is used.
### Testing and Evaluation
### Training
### Citation 
Please consider citing the following paper if the code is helpful in your research work:
<pre>
@inproceedings{xu2018structured,
  title={Structured Attention Guided Convolutional Neural Fields for Monocular Depth Estimation},
  author={Xu, Dan and Wang, Wei and Tang, Hao and Liu, Hong and Sebe, Nicu and Ricci, Elisa},
  booktitle={CVPR},
  year={2018}
}
</pre>

The details are coming soon ...

