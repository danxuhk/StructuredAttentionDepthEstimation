#!/usr/bin/env python
'''
Structured Attention Network with Caffe NetSpec generation
Base Network: ResNet50.
'''
import sys
import caffe
from caffe import layers as L, params as P


def _conv_bn_scale(bottom, nout, bias_term=False, **kwargs):
    '''Helper to build a conv -> BN -> relu block.
    '''
    conv = L.Convolution(bottom, num_output=nout, bias_term=bias_term,
                         **kwargs)
    bn = L.BatchNorm(conv, use_global_stats=True, in_place=True)
    scale = L.Scale(bn, bias_term=True, in_place=True)
    return conv, bn, scale


def _resnet_block(name, n, bottom, nout, branch1=False, initial_stride=2):
    '''Basic ResNet block.
    '''
    if branch1:
        res_b1 = 'res{}_branch1'.format(name)
        bn_b1 = 'bn{}_branch1'.format(name)
        scale_b1 = 'scale{}_branch1'.format(name)
        n[res_b1], n[bn_b1], n[scale_b1] = _conv_bn_scale(
            bottom, 4*nout, kernel_size=1, stride=initial_stride, pad=0)
    else:
        initial_stride = 1

    res = 'res{}_branch2a'.format(name)
    bn = 'bn{}_branch2a'.format(name)
    scale = 'scale{}_branch2a'.format(name)
    n[res], n[bn], n[scale] = _conv_bn_scale(
        bottom, nout, kernel_size=1, stride=initial_stride, pad=0)
    relu2a = 'res{}_branch2a_relu'.format(name)
    n[relu2a] = L.ReLU(n[scale], in_place=True)
    res = 'res{}_branch2b'.format(name)
    bn = 'bn{}_branch2b'.format(name)
    scale = 'scale{}_branch2b'.format(name)
    n[res], n[bn], n[scale] = _conv_bn_scale(
        n[relu2a], nout, kernel_size=3, stride=1, pad=1)
    relu2b = 'res{}_branch2b_relu'.format(name)
    n[relu2b] = L.ReLU(n[scale], in_place=True)
    res = 'res{}_branch2c'.format(name)
    bn = 'bn{}_branch2c'.format(name)
    scale = 'scale{}_branch2c'.format(name)
    n[res], n[bn], n[scale] = _conv_bn_scale(
        n[relu2b], 4*nout, kernel_size=1, stride=1, pad=0)
    res = 'res{}'.format(name)
    if branch1:
        n[res] = L.Eltwise(n[scale_b1], n[scale])
    else:
        n[res] = L.Eltwise(bottom, n[scale])
    relu = 'res{}_relu'.format(name)
    n[relu] = L.ReLU(n[res], in_place=True)

def resnet50(n, bottom):
    '''ResNet 50 layers.
    '''
    n.conv1, n.bn_conv1, n.scale_conv1 = _conv_bn_scale(
        bottom, 64, bias_term=True, kernel_size=7, pad=3, stride=2)
    n.conv1_relu = L.ReLU(n.scale_conv1)
    n.pool1 = L.Pooling(
        n.conv1_relu, kernel_size=3, stride=2, pool=P.Pooling.MAX)

    _resnet_block('2a', n, n.pool1, 64, branch1=True, initial_stride=1)
    _resnet_block('2b', n, n.res2a_relu, 64)
    _resnet_block('2c', n, n.res2b_relu, 64)

    _resnet_block('3a', n, n.res2c_relu, 128, branch1=True)
    _resnet_block('3b', n, n.res3a_relu, 128)
    _resnet_block('3c', n, n.res3b_relu, 128)
    _resnet_block('3d', n, n.res3c_relu, 128)

    _resnet_block('4a', n, n.res3d_relu, 256, branch1=True)
    _resnet_block('4b', n, n.res4a_relu, 256)
    _resnet_block('4c', n, n.res4b_relu, 256)
    _resnet_block('4d', n, n.res4c_relu, 256)
    _resnet_block('4e', n, n.res4d_relu, 256)
    _resnet_block('4f', n, n.res4e_relu, 256)

    _resnet_block('5a', n, n.res4f_relu, 512, branch1=True)
    _resnet_block('5b', n, n.res5a_relu, 512)
    _resnet_block('5c', n, n.res5b_relu, 512)

    #n.pool5 = L.Pooling(
    #    n.res5c_relu, kernel_size=7, stride=1, pool=P.Pooling.AVE)
        
def MeanFieldUpdate(n, bottom_send, bottom_receive, feat_ind, mf_iter, feat_num):
    '''
    Meanfield updating for the features and the attention for one pair of features.
    bottom_list is a list of observation features derived from the backbone CNN.
    '''
    #generating an attention map
    concat_f = 'concat_f{}_mf{}'.format(feat_ind, mf_iter)
    conv_f = 'conv_f{}_mf{}'.format(feat_ind, mf_iter)
    atten_f = 'atten_f{}_mf{}'.format(feat_ind, mf_iter)
    norm_atten_f = 'norm_atten_f{}_mf{}'.format(feat_ind, mf_iter)
    message_f = 'message_f{}_mf{}'.format(feat_ind, mf_iter)
    filter_message_f = 'filter_message_f{}_mf{}'.format(feat_ind, mf_iter)
    message_scaled = 'message_scaled_f{}_mf{}'.format(feat_ind, mf_iter)
    updated_f = 'updated_f{}_mf{}'.format(feat_ind, mf_iter)

    n[concat_f] = L.Concat(bottom_send, bottom_receive)
    #specify parameter names to make them share between different meanfield updating
    n[atten_f] = L.Convolution(n[concat_f], num_output=feat_num, kernel_size=3, stride=1, pad=1, param=[dict(name='atten_f{}_w'.format(feat_ind), lr_mult=1, decay_mult=1), dict(name='atten_f{}_b'.format(feat_ind), lr_mult=2, decay_mult=0)])
    n[norm_atten_f] = L.Sigmoid(n[atten_f])
    n[message_f] = L.Convolution(bottom_send, num_output=feat_num, kernel_size=3, stride=1, pad=1, param=[dict(name='message_f{}_w'.format(feat_ind), lr_mult=1, decay_mult=1), dict(name='message_f{}_b'.format(feat_ind), lr_mult=2, decay_mult=0)])
    n[filter_message_f] = L.Eltwise(n[message_f], n[norm_atten_f], operation=P.Eltwise.PROD)
    #scale the messages before adding 
    n[message_scaled] = L.Scale(n[filter_message_f], bias_term=True, in_place=True)
    n[updated_f] = L.Eltwise(bottom_receive, n[message_scaled], operation=P.Eltwise.SUM)  

                   
def SAN(n, bottom, feat_num, feat_width, feat_height):
    '''
    Network definition for the structured attention guided learning module.
    The meanfield updating is carried out three times for the features.
    '''
    #pass data to Backbone ResNet50
    resnet50(n, bottom)
    
    #generating multi-scale features with the same dimension
    n.res4f_dec_1 = L.Deconvolution(n.res4f_relu, convolution_param=dict(num_output=feat_num, kernel_size=4, stride=2, pad=1, 
        weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type= 'constant',value=0)), 
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    n.res4f_dec_1_relu = L.ReLU(n.res4f_dec_1, in_place=True)

    n.res5c_dec_1 = L.Deconvolution(n.res5c_relu, convolution_param=dict(num_output=feat_num, kernel_size=8, stride=4, pad=2, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type= 'constant',value=0)), param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    n.res5c_dec_1_relu = L.ReLU(n.res5c_dec_1, in_place=True)

    n.res4f_dec = L.Interp(n.res4f_dec_1_relu, interp_param=dict(height=feat_height, width=feat_width))
    n.res3d_dec = L.Interp(n.res3d_relu,  interp_param=dict(height=feat_height, width=feat_width))
    n.res5c_dec = L.Interp(n.res5c_dec_1_relu, interp_param=dict(height=feat_height, width=feat_width))
    
    
    ##add deep supervision for three semantic layers
    n.prediction_3d = L.Convolution(n.res3d_dec, num_output=1, kernel_size=3, stride=1, pad=1)
    n.prediction_4f = L.Convolution(n.res4f_dec, num_output=1, kernel_size=3, stride=1, pad=1)
    n.prediction_5c = L.Convolution(n.res5c_dec, num_output=1, kernel_size=3, stride=1, pad=1)
    
    #the first meanfield updating
    MeanFieldUpdate(n, n.res3d_dec, n.res5c_dec, 1, 1, feat_num)
    MeanFieldUpdate(n, n.res4f_dec, n.updated_f1_mf1, 2, 1, feat_num)
    MeanFieldUpdate(n, n.res5c_dec, n.updated_f2_mf1, 3, 1, feat_num)
    #the second meanfield updating
    MeanFieldUpdate(n, n.res3d_dec, n.updated_f3_mf1, 1, 2, feat_num)
    MeanFieldUpdate(n, n.res4f_dec, n.updated_f1_mf2, 2, 2, feat_num)
    MeanFieldUpdate(n, n.res5c_dec, n.updated_f2_mf2, 3, 2, feat_num)
    #the third meanfield updating
    MeanFieldUpdate(n, n.res3d_dec, n.updated_f3_mf2, 1, 3, feat_num)
    MeanFieldUpdate(n, n.res4f_dec, n.updated_f1_mf3, 2, 3, feat_num)
    MeanFieldUpdate(n, n.res5c_dec, n.updated_f2_mf3, 3, 3, feat_num)
    #the fourth meanfield updating
    MeanFieldUpdate(n, n.res3d_dec, n.updated_f3_mf3, 1, 4, feat_num)
    MeanFieldUpdate(n, n.res4f_dec, n.updated_f1_mf4, 2, 4, feat_num)
    MeanFieldUpdate(n, n.res5c_dec, n.updated_f2_mf4, 3, 4, feat_num)
    #the fifth meanfield updating
    MeanFieldUpdate(n, n.res3d_dec, n.updated_f3_mf4, 1, 5, feat_num)
    MeanFieldUpdate(n, n.res4f_dec, n.updated_f1_mf5, 2, 5, feat_num)
    MeanFieldUpdate(n, n.res5c_dec, n.updated_f2_mf5, 3, 5, feat_num)

    #using a concatanation instead of meanfield updating
    #n.concat_all = L.Concat(n.res3d_dec, n.res4f_dec, n.res5c_dec_relu)
    #n.dropout_l = L.Dropout(n.concat_all, in_place=True, dropout_ratio=0.3)
    
    #produce the output 
    n.prediction_map_1 = L.Deconvolution(n.updated_f3_mf5, convolution_param=dict(num_output=feat_num/2, kernel_size=4, stride=2, pad=1, 
        weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0)), 
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    n.prediction_map_1_relu = L.ReLU(n.prediction_map_1, in_place=True)
    n.prediction_map_2 = L.Deconvolution(n.prediction_map_1_relu, convolution_param=dict(num_output=feat_num/4, kernel_size=4, stride=2, pad=1, 
        weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0)), 
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    n.prediction_map_2_relu = L.ReLU(n.prediction_map_2, in_place=True)
    n.prediction_map = L.Convolution(n.prediction_map_2_relu, num_output=1, kernel_size=3, stride=1, pad=1)

    #n.prediction_map_ori_resolution = L.Interp(n.prediction_map, interp_param=dict(height=, width=feat_width))
    #n.prediction_map_ori_resolution = L.Deconvolution(n.prediction_map, convolution_param=dict(num_output=1, kernel_size=8, stride=4, pad=2, bias_term=False), param=[dict(lr_mult=0)])
    
if __name__ == '__main__':
    net = caffe.NetSpec()
    #net.data = L.Data(source='./lmdb/train_data_kitti_80_lmdb', backend=P.Data.LMDB, batch_size=4, ntop=1, transform_param=dict(mean_value=[103.939, 116.779, 123.68]))
    #net.label = L.Data(source='./lmdb/train_label_kitti_80_lmdb', backend=P.Data.LMDB, batch_size=4, ntop=1)
    net.data, net.label = L.Python(python_param=dict(module='Pixel_Data_Layer', layer='PixelDataLayer',
        param_str='{ "batch_size": 4, "data_root_dir": "/scratch/local/ssd/danxu/KITTI", "list_file": "/home/danxu/projects/StructuredAttentionDepth/utils/filenames/eigen_train_pairs.txt", "scale_factors": [1], "mean_values": [103.939, 116.779, 123.68], "mirror": True, "shuffle": True, "split": "train" }'), ntop=2)
    
    SAN(net, net.data, feat_num=512, feat_width=80, feat_height=24)
    net.prediction_3d_output = L.Interp(net.prediction_3d,  interp_param=dict(height=188, width=621))
    net.prediction_4f_output = L.Interp(net.prediction_4f,  interp_param=dict(height=188, width=621))
    net.prediction_5c_output = L.Interp(net.prediction_5c,  interp_param=dict(height=188, width=621))
    net.final_output = L.Interp(net.prediction_map,  interp_param=dict(height=188, width=621))
    #loss
    net.loss_3d = L.EuclideanMaskLoss(net.prediction_3d_output, net.label, loss_weight=0.8)    
    net.loss_4f = L.EuclideanMaskLoss(net.prediction_4f_output, net.label, loss_weight=0.8)  
    net.loss_5c = L.EuclideanMaskLoss(net.prediction_5c_output, net.label, loss_weight=0.8) 
    net.loss_final = L.EuclideanMaskLoss(net.final_output, net.label, loss_weight=1)
    with open('train_SAN.prototxt', 'w') as f:
        f.write(str(net.to_proto()))
