"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""

import os
import glob
import h5py
import random
import matplotlib.pyplot as plt

from PIL import Image  # for loading images as YCbCr format
import scipy.misc
import scipy.ndimage
import numpy as np
import cv2



def read_h5data(path):
    """
    Read h5 format data file
    
    Args:
    path: file path of desired file
    data: '.h5' file format that contains train data values
    label: '.h5' file format that contains train label values
    """
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
    return data, label


def preprocess(path, args):
    """
    Preprocess single image file
    (1) Read original image as YCbCr format (and grayscale as default)
    (2) Normalize
    (3) Apply image file with bicubic interpolation
    
    Args:
    path: file path of desired file
    input_: image applied bicubic interpolation (low-resolution)
    label_: image with original resolution (high-resolution)
    """
    image = plt.imread(path)
    label_ = modcrop(image, args.scale)
    
    # Must be normalized
    label_ = label_ / 255.


    if args.is_train != "inference":
        input_ = scipy.ndimage.interpolation.zoom(label_, [(1./args.scale),(1./args.scale),1], prefilter=False)

    else :
        input_ = label_

    input_ = scipy.ndimage.interpolation.zoom(input_, [(args.scale/1.),(args.scale/1.),1] , prefilter=False)
    return input_, label_


def prepare_data(sess, args):
    """
    Args:
    dataset: choose train dataset or test dataset
    
    For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
    """
    
    if args.is_train == "train":
        data_dir = os.path.join(os.getcwd(), args.is_train, args.train_subdir)
        data = glob.glob(os.path.join(data_dir, "*"))
        print(data)
    elif args.is_train == "test":
        data = os.path.join(os.getcwd(),args.is_train,args.test_imgpath)
    elif args.is_train == "inference":
        data = os.path.join(os.getcwd(), args.is_train,args.infer_imgpath)
    return data



def make_data(sess, data, label, args):
    """
    Make input data as h5 file format
    Depending on 'is_train' (flag value), savepath would be changed.
    """
    if args.is_train == "train":
        savepath = os.path.join(os.getcwd(), 'checkpoint/train.h5')
    elif args.is_train == "test":
        savepath = os.path.join(os.getcwd(), 'checkpoint/test.h5')
    elif args.is_train == "inference":
        savepath = os.path.join(os.getcwd(), 'checkpoint/inference.h5')


    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('data', data=data)
        hf.create_dataset('label', data=label)



def modcrop(image, scale=3):
  """
  To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
  
  We need to find modulo of height (and width) and scale factor.
  Then, subtract the modulo from height (and width) of original image size.
  There would be no remainder even after scaling operation.
  """
  if len(image.shape) == 3:
    h, w, _ = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w, :]
  else:
    h, w = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w]
  return image



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def input_setup(sess, args):
    """
    Read image files and make their sub-images and saved them as a h5 file format.
    """
#===========================================================
# [input setup] / Load data path
#===========================================================
    data = prepare_data(sess, args=args)

#===========================================================
# [input setup] / split image
#===========================================================
    sub_input_sequence = []
    sub_label_sequence = []
    padding = abs(args.image_size - args.label_size) / 2 # 6
    nx = ny = 0

    #----------------------------------------------------------------
    # [input setup] / split image - for train
    #----------------------------------------------------------------
    if args.is_train == "train":
        for i in range(len(data)):
            input_, label_ = preprocess(data[i],args)

            if len(input_.shape) == 3:
                h, w, _ = input_.shape
                
            for x in range(0, h-args.image_size+1, args.stride):
                if i == 0 :
                    nx += 1
                for y in range(0, w-args.image_size+1, args.stride):
                    if i == 0 and nx == 1 :
                        ny += 1

                    sub_input = input_[x:x+args.image_size, y:y+args.image_size, :] # [33 x 33 x 3]
                    sub_label = label_[x+int(padding):x+int(padding)+args.label_size, y+int(padding):y+int(padding)+args.label_size, :] # [21 x 21 x 3]
    
                    # Make channel value
                    sub_input = sub_input.reshape([args.image_size, args.image_size, args.c_dim])
                    sub_label = sub_label.reshape([args.label_size, args.label_size, args.c_dim])
        
                    sub_input_sequence.append(sub_input)
                    sub_label_sequence.append(sub_label)



    #----------------------------------------------------------------
    # [input setup] / split image - for test
    #----------------------------------------------------------------
    elif args.is_train == "test":
        #input_, label_ = preprocess(data[2], args.scale)
        input_, label_ = preprocess(data, args)
        h, w, _ = input_.shape

        # Numbers of sub-images in height and width of image are needed to compute merge operation.
        for x in range(0, h-args.image_size+1, args.label_size):
            nx += 1
            ny = 0
            for y in range(0, w-args.image_size+1, args.label_size):
                ny += 1
                sub_input = input_[x:x+args.image_size, y:y+args.image_size,:] # [33 x 33 x 3]
                sub_label = label_[x+int(padding):x+int(padding)+args.label_size, y+int(padding):y+int(padding)+args.label_size,:] # [21 x 21 x 3]
                
                sub_input = sub_input.reshape([args.image_size, args.image_size, args.c_dim])
                sub_label = sub_label.reshape([args.label_size, args.label_size, args.c_dim])
        
                sub_input_sequence.append(sub_input)
                sub_label_sequence.append(sub_label)


    #----------------------------------------------------------------
    # [input setup] / split image - for inference
    #----------------------------------------------------------------
    else:
        input_, _ = preprocess(data, args)
        h, w, _ = input_.shape
    
        # Numbers of sub-images in height and width of image are needed to compute merge operation.
        for x in range(0, h - args.image_size + 1, args.label_size):
            nx += 1
            ny = 0
            for y in range(0, w - args.image_size + 1, args.label_size):
                ny += 1
                sub_input = input_[x:x + args.image_size, y:y + args.image_size, :3]  # [33 x 33 x3]
                sub_input = sub_input.reshape([args.image_size, args.image_size,  args.c_dim])
                sub_input_sequence.append(sub_input)
    """
    len(sub_input_sequence) : the number of sub_input (33 x 33 x ch) in one image
    (sub_input_sequence[0]).shape : (33, 33, 1)
    """
    
#===========================================================
# [input setup] / save h5 data
#===========================================================
    # Make list to numpy array. With this transform
    arrdata = np.asarray(sub_input_sequence) # [?, 33, 33, 1]
    arrlabel = np.asarray(sub_label_sequence) # [?, 21, 21, 1]
    make_data(sess, arrdata, arrlabel,args=args)

    return nx, ny
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




def imsave(image, path):
    #image = image - np.min(image)
    #image = image / np.max(image)
    image = np.clip(image,0,1)
    return plt.imsave(path,image)
    #return scipy.misc.imsave(path, image) #why different?

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros([h*size[0], w*size[1], 3])
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img
