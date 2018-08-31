import argparse
import os
import pprint

import tensorflow as tf

from model import SRCNN

if __name__ == '__main__':
#=======================================================
# [global variables]
#=======================================================
    pp = pprint.PrettyPrinter()
    args = None


#=======================================================
# [add parser]
#=======================================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=300000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=33)
    parser.add_argument("--label_size", type=int, default=21)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--c_dim", type=int, default=3)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--stride", type=int, default=14)#14 for training, lable_size for inference
    parser.add_argument("--checkpoint_dir", default="checkpoint")
    parser.add_argument("--cpkt_itr", default=26500)#set -1 for training from scratch, None for latest
    parser.add_argument("--result_dir", default="result")
    parser.add_argument("--train_subdir", default="mytrainset")
    parser.add_argument("--test_imgpath", default="Set5/bird_GT.bmp")
    parser.add_argument("--infer_imgpath", default="0.jpg")
    parser.add_argument("--is_train", default="inference", choices = ["train", "test", "inference"] )
    parser.add_argument("--save_extension", default=".jpg", choices = ["jpg", "png"])

    args = parser.parse_args()
    pp.pprint(args)

#=======================================================
# [make directory]
#=======================================================
    if not os.path.exists(os.path.join(os.getcwd(),args.checkpoint_dir)):
        os.makedirs(os.path.join(os.getcwd(),args.checkpoint_dir))
    if not os.path.exists(os.path.join(os.getcwd(),args.result_dir)):
        os.makedirs(os.path.join(os.getcwd(),args.result_dir))
    if not os.path.exists(os.path.join(os.getcwd(),args.result_dir,args.is_train)):
        os.makedirs(os.path.join(os.getcwd(),args.result_dir,args.is_train))
    
    
#=======================================================
# [Main]
#=======================================================
    g = tf.Graph()
    g.as_default()
    with tf.Session(graph=g) as sess:
    #-----------------------------------
    # build model
    #-----------------------------------
        srcnn = SRCNN(sess,
                      image_size=args.image_size,
                      label_size=args.label_size,
                      batch_size=args.batch_size,
                      c_dim=args.c_dim,
                      checkpoint_dir=args.checkpoint_dir,
                      result_dir=args.result_dir)

    #-----------------------------------
    # train, test, inferecnce
    #-----------------------------------
        srcnn.train(args)
