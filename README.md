# SRCNN-Tensorflow
Tensorflow implementation of Convolutional Neural Networks for super-resolution.  The original Matlab and Caffe from official website can be found [here](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html).

## Prerequisites
 * python 3.x
 * Tensorflow > 1.5
 * Scipy version > 0.18 ('mode' option from scipy.misc.imread function)
 * matplotlib
 * argparse

## Properties (what's different from reference code)
 * This code requires Tensorflow. This code was fully implemented based on Python 3 differently from the original.
 * This code supports only RGB color images.
 * This code use Adam optimizer instead of GradienDecententOptimizer differently from the original.
 * This code supports tensorboard summarization
 * This code supports data augmentation (rotation and mirror flip)
 * This code supports custom dataset


## Usage
```
usage: main_srcnn.py [-h] [--epoch EPOCH] [--batch_size BATCH_SIZE]
                     [--image_size IMAGE_SIZE] [--label_size LABEL_SIZE]
                     [--lr LR] [--c_dim C_DIM] [--scale SCALE]
                     [--stride STRIDE] [--checkpoint_dir CHECKPOINT_DIR]
                     [--cpkt_itr CPKT_ITR] [--result_dir RESULT_DIR]
                     [--train_subdir TRAIN_SUBDIR] [--test_subdir TEST_SUBDIR]
                     [--infer_subdir INFER_SUBDIR]
                     [--infer_imgpath INFER_IMGPATH]
                     [--mode {train,test,inference}]
                     [--save_extension {jpg,png}]

optional arguments:
  -h, --help            show this help message and exit
  --epoch EPOCH
  --batch_size BATCH_SIZE
  --image_size IMAGE_SIZE
  --label_size LABEL_SIZE
  --lr LR
  --c_dim C_DIM
  --scale SCALE
  --stride STRIDE
  --checkpoint_dir CHECKPOINT_DIR
  --cpkt_itr CPKT_ITR
  --result_dir RESULT_DIR
  --train_subdir TRAIN_SUBDIR
  --test_subdir TEST_SUBDIR
  --infer_subdir INFER_SUBDIR
  --infer_imgpath INFER_IMGPATH
  --mode {train,test,inference}
  --save_extension {jpg,png}
```

For training, `python main.py --mode train --check_itr 0` [set 0 for training from scratch, -1 for latest]
For testing, `python main.py --mode test`
For inference with cumstom dataset, `python main.py --mode inference --infer_imgpath 3.bmp` [result will be generated in ./result/inference]
For running tensorboard, `tensorboard --logdir=./board` then access localhost:6006 with your browser

## Result
![orig](https://github.com/tegg89/SRCNN-Tensorflow/blob/master/result/orig.png)<br>
![bicubic](https://github.com/tegg89/SRCNN-Tensorflow/blob/master/result/bicubic.png)<br>
![srcnn](https://github.com/tegg89/SRCNN-Tensorflow/blob/master/result/srcnn.png)

## References
* [tegg89/SRCNN-Tensorflow](https://github.com/tegg89/SRCNN-Tensorflow) : reference source code
* [SRCNN](https://arxiv.org/abs/1501.00092) : reference paper


## Author
Dohyun Kim

