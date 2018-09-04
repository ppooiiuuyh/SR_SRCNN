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




def preprocess(path, args):
    image = plt.imread(path)
    image_croped = modcrop(image, args.scale)
    
    # Must be normalized
    image_croped = image_croped / 255.

    if args.mode == "train" or args.mode == "test":
        label_ = image_croped
        input_ = scipy.ndimage.interpolation.zoom(image_croped, [(1./args.scale),(1./args.scale),1], prefilter=False)
        input_ = scipy.ndimage.interpolation.zoom(input_, [(args.scale/1.),(args.scale/1.),1] , prefilter=False)
        return input_,label_

    elif args.mode == "inference":
        input_ = scipy.ndimage.interpolation.zoom(image_croped, [(args.scale / 1.), (args.scale / 1.), 1], prefilter=False)
        return input_



def prepare_data(sess, args, mode):
    if args.mode == "train":
        data_dir = os.path.join(os.getcwd(), args.mode, args.train_subdir)
        data = glob.glob(os.path.join(data_dir, "*"))
 
    elif args.mode == "test":
        data_dir = os.path.join(os.getcwd(), args.mode, args.test_subdir)
        data = glob.glob(os.path.join(data_dir, "*"))
 
    elif args.mode == "inference":
        data_dir = os.path.join(os.getcwd(), args.mode, args.infer_subdir)
        data = glob.glob(os.path.join(data_dir, args.infer_imgpath))
    return data




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



def augumentation(img_sequence):
    augmented_sequence = []
    for img in img_sequence:
        for _ in range(4):
            rot_img = np.rot90(img)
            augmented_sequence.append(rot_img)
            
        flipped_img = np.fliplr(img)
        
        for _ in range(4):
            rot_flipped_img = np.rot90(flipped_img)
            augmented_sequence.append(rot_flipped_img)
            

    img_sequence.extend(augmented_sequence)
    return img_sequence


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def input_setup(sess, args, mode):
#===========================================================
# [input setup] / split image
#===========================================================
    sub_input_sequence = []
    sub_label_sequence = []
    padding = abs(args.image_size - args.label_size) / 2 # 6


#----------------------------------------------------------------
# [input setup] / split image - for trainset and testset
#----------------------------------------------------------------
    if mode == "train" or mode == "test":
        data = prepare_data(sess, args=args, mode=mode)
        for i in range(len(data)):
            input_, label_ = preprocess(data[i],args) #normalized full-size image
            h, w, _ = input_.shape #only for R,G,B image
            
            for x in range(0, h-args.image_size+1, args.stride):
                for y in range(0, w-args.image_size+1, args.stride):
                    sub_input = input_[x:x+args.image_size, y:y+args.image_size, :] # [33 x 33 x 3]
                    sub_label = label_[x+int(padding):x+int(padding)+args.label_size, y+int(padding):y+int(padding)+args.label_size, :] # [21 x 21 x 3]
    
                    # Make channel value
                    sub_input = sub_input.reshape([args.image_size, args.image_size, args.c_dim])
                    sub_label = sub_label.reshape([args.label_size, args.label_size, args.c_dim])
        
                    sub_input_sequence.append(sub_input)
                    sub_label_sequence.append(sub_label)
                    
        return sub_input_sequence, sub_label_sequence

#----------------------------------------------------------------
# [input setup] / split image - for inference
#----------------------------------------------------------------
    elif mode == "inference": #for 1 image
        data = prepare_data(sess, args=args, mode=mode)
        input_ = preprocess(data[0], args)
        h, w, _ = input_.shape
        nx = ny = 0
        for x in range(0, h - args.image_size + 1, args.label_size):
            nx += 1
            ny = 0
            for y in range(0, w - args.image_size + 1, args.label_size):
                ny += 1
                sub_input = input_[x:x + args.image_size, y:y + args.image_size, :3]  # [33 x 33 x3]
                sub_input = sub_input.reshape([args.image_size, args.image_size,  args.c_dim])
                sub_input_sequence.append(sub_input)
            
    return nx, ny, sub_input_sequence, sub_label_sequence
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



'''
def make_data(sess, data, label, args):
    """
    Make input data as h5 file format
    Depending on 'mode' (flag value), savepath would be changed.
    """
    if args.mode == "train":
        savepath = os.path.join(os.getcwd(), 'checkpoint/train.h5')
    elif args.mode == "test":
        savepath = os.path.join(os.getcwd(), 'checkpoint/test.h5')
    elif args.mode == "inference":
        savepath = os.path.join(os.getcwd(), 'checkpoint/inference.h5')


    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('data', data=data)
        hf.create_dataset('label', data=label)
'''

'''
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
'''
'''

#===========================================================
# [input setup] / save h5 data
#===========================================================
    # Make list to numpy array. With this transform
    arrdata = np.asarray(sub_input_sequence) # [?, 33, 33, 1]
    arrlabel = np.asarray(sub_label_sequence) # [?, 21, 21, 1]
    make_data(sess, arrdata, arrlabel,args=args)

'''