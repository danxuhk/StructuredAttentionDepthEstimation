import numpy as np
import random
import scipy.ndimage as ndimage


class DataTransformer:

    """
    DataTransformer is a simple class for preprocessing and deprocessing
    images for caffe.
    """

    def __init__(self, mean=[103.939, 116.779, 123.68]):
        self.mean = np.array(mean, dtype=np.float32)
        self.scale = 1.0
    
    def scale_img(self, im, is_depth):
        if not is_depth:
            # scale input image
            return cv2.resize(im, (im.shape[0] * float(self.scale), im.shape[1] * float(self.scale), im.shape[2]).astype(np.uint8));
        else:
            return (cv2.resize(im, (im.shape[0] * float(self.scale), im.shape[1] * float(self.scale), im.shape[2]).astype(np.uint8))) / float(self.scale);       
                     
    def preprocess(self, im, depth_label):
        """
        preprocess() emulate the pre-processing occurring in the vgg16 caffe
        prototxt.
        """
        ##process input image
        assert im.dtype == np.uint8;
        im = np.float32(im)
        im = im[:, :, ::-1]  # change to BGR
        im -= self.mean
        # resize
        #im = scale_img(im, False);
        
        ##process depth map
        assert depth_label.dtype == np.uint16;
        depth_label = depth_label / float(100);
        #depth_label = scale_img(depth_label, True);
               
        im = im.transpose((2, 0, 1));
        depth_label = depth_label.reshape(1, im.shape[1], im.shape[2]);

        return im, depth_label