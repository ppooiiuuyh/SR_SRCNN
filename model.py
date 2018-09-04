import os
import time
import tensorflow as tf
from utils import input_setup, imsave, merge, augumentation
import matplotlib.pyplot as plt
import pprint
import math
import numpy as np

class SRCNN(object):
#==========================================================
# class initializer
#==========================================================
    def __init__(self, sess, args):
        self.sess = sess
        self.args = args
        self.model()
        self.init_model()
        self.preprocess()



#==========================================================
# build model
#==========================================================
    def model(self):
        self.images = tf.placeholder(tf.float32, [None, self.args.image_size, self.args.image_size, self.args.c_dim], name='images')
        self.labels = tf.placeholder(tf.float32, [None, self.args.label_size, self.args.label_size, self.args.c_dim], name='labels')
        self.inner_model()
        self.other_tensors()
    

    def inner_model(self):
        with tf.variable_scope("block1") as scope:
            conv1 = tf.layers.conv2d(self.images, 64, [9, 9], strides=[1, 1], padding='VALID',
                                     kernel_initializer=tf.random_normal_initializer(0, 0.001),activation=tf.nn.relu)
            conv2 = tf.layers.conv2d(conv1, 32, [1, 1], strides=[1, 1], padding='VALID',
                                     kernel_initializer=tf.random_normal_initializer(0, 0.001),activation=tf.nn.relu)
        with tf.variable_scope("block2") as scope:
            #conv3 = tf.nn.conv2d(conv2, self.weights['w3'], strides=[1, 1, 1, 1], padding='VALID') + self.biases['b3']
            conv3 = tf.layers.conv2d(conv2, 3, [5, 5], strides=[1,1], padding='VALID',
                                     kernel_initializer=tf.random_normal_initializer(0,0.001))
        self.recon_img = conv3
        self.pred = conv3


    def other_tensors(self):
        self.saver = tf.train.Saver(max_to_keep=0)

        self.global_step_tensor = tf.Variable(0, trainable=False, name = "global_step")
        self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
        self.mse_placeholder = tf.placeholder(tf.float32)
        self.psnr = 10 * (tf.log(1 / self.mse_placeholder)) / math.log(10)
        self.train_op = tf.train.GradientDescentOptimizer(self.args.lr).minimize(self.loss)
        var_list1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='block1')
        var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='block2')
        train_op1 = tf.train.AdamOptimizer(0.0001).minimize(self.loss, var_list=var_list1)
        train_op2 = tf.train.AdamOptimizer(0.00001).minimize(self.loss, var_list=var_list2, global_step=self.global_step_tensor)
        self.train_op = tf.group(train_op1, train_op2)

        summary_tensor_loss = tf.summary.scalar("loss", self.loss)
        summary_tensor_psnr = tf.summary.scalar("psnr", self.psnr)
        summary_tensor_input_patches = tf.summary.image("input patches",self.images, max_outputs= 10)
        summary_tensor_output_patches = tf.summary.image("output patches", self.recon_img, max_outputs= 10)
        summary_tensor_label_patches = tf.summary.image("label patches", self.labels, max_outputs= 10)
        #extend vs append vs '+'. 'append' and 'extend' do not generate new list
        self.merged_train = tf.summary.merge([summary_tensor_loss]+[tf.summary.histogram(var.name,var) for var in tf.global_variables()])
        self.merged_test = tf.summary.merge([summary_tensor_input_patches, summary_tensor_output_patches, summary_tensor_label_patches])
        self.merged_psnr = tf.summary.merge([summary_tensor_psnr])
        self.summary_writer = tf.summary.FileWriter("./board", self.sess.graph)
        
    def init_model(self):
        self.sess.run(tf.global_variables_initializer())
        if self.cpkt_load(self.args.checkpoint_dir, self.args.cpkt_itr):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")



#==========================================================
# preprocessing
#==========================================================
    def preprocess(self):
        if self.args.mode == "train":
            self.train_data, self.train_label = input_setup(self.sess, self.args, mode="train")
            self.test_data , self.test_label  = input_setup(self.sess, self.args, mode="test")
            self.train_data = augumentation(self.train_data)
            self.train_label = augumentation(self.train_label)
            
            
            '''
            print(train_data.shape)
            print(train_data[0, :, :, 0].shape)

            sample_size = [10, 10]
            data = train_data
            fig, ax = plt.subplots(nrows=sample_size[0], ncols=sample_size[1], figsize=sample_size)
            for r in range(sample_size[0]):
                for c in range(sample_size[1]):
                    ax[r, c].set_axis_off()
                    ax[r, c].imshow(data[r * sample_size[1] + c, :, :, :])

            plt.show()
            '''

        elif self.args.mode == "test":
            self.train_data, self.train_label = input_setup(self.sess, self.args, mode = "test")


        elif self.args.mode == "inference":
            self.nx, self.ny, self.train_data, self.train_label = input_setup(self.sess, self.args, mode = "inference")


        else:
            assert ("invalid augments. must be in train, test, inference")
            



#==========================================================
# train
#==========================================================
    def train(self):
        print("Training...")
        start_time = time.time()
        for ep in range(self.args.epoch):
            # Run by batch images
            batch_idxs = len(self.train_data) // self.args.batch_size
            for idx in range(0, batch_idxs):
                batch_images = self.train_data[idx*self.args.batch_size : (idx+1)*self.args.batch_size]
                batch_labels = self.train_label[idx*self.args.batch_size : (idx+1)*self.args.batch_size]
                feed_dict = {self.images: batch_images, self.labels: batch_labels}
                result, err, summary_train = self.sess.run([self.train_op, self.loss, self.merged_train],feed_dict=feed_dict)
                self.summary_writer.add_summary(summary_train,global_step=self.global_step_tensor.eval(self.sess))

            if ep % 10 == 0:
                print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
                         % ((ep+1), self.global_step_tensor.eval(self.sess), time.time()-start_time, err))
                self.test(self.args)
            

            if ep % 100 == 0:
                self.cpkt_save(self.args.checkpoint_dir, ep)

#------------------------------------------------------
# test
#------------------------------------------------------
    def test(self, args):
        print("Testing...")

        errs = []
        psnrs = []

        batch_idxs = len(self.test_data) // self.args.batch_size
        for idx in range(0, batch_idxs):
            batch_images_test = self.test_data[idx * self.args.batch_size: (idx + 1) * self.args.batch_size]
            batch_labels_test = self.test_label[idx * self.args.batch_size: (idx + 1) * self.args.batch_size]
            feed_dict = {self.images: batch_images_test,  self.labels: batch_labels_test}

            #image summary requires too much storage
            result, err, input_imgs, output_imgs, label_imgs = \
                self.sess.run([self.train_op, self.loss, self.recon_img, self.images, self.labels], feed_dict=feed_dict)
            #result, err, input_imgs, output_imgs, label_imgs, summary_test = \
            #    self.sess.run([self.train_op, self.loss, self.recon_img, self.images, self.labels, self.merged_test], feed_dict=feed_dict)
            # self.summary_writer.add_summary(summary_test,global_step=self.global_step_tensor.eval(self.sess))

            errs.append(err)
            
        mse = np.mean(err)
        psnr,summary_psnr = self.sess.run([self.psnr,self.merged_psnr],feed_dict={self.mse_placeholder:mse})
        self.summary_writer.add_summary(summary_psnr,global_step=self.global_step_tensor.eval(self.sess))
        print(str(round(np.mean(psnr),3)) + "dB")


#------------------------------------------------------
# inference
#------------------------------------------------------
    def inference(self, infer_imgpath):
        print("inferring...")
        
        start_time = time.time()
        result = self.sess.run(self.pred,feed_dict={self.images: self.train_data})
        print("Elapse:", time.time() - start_time)

        result = merge(result, [self.nx, self.ny])
        result = result.squeeze()
        image_path = os.path.join(os.getcwd(), self.args.result_dir)
        image_path = os.path.join(image_path,self.args.mode)
        image_path = os.path.join(image_path,infer_imgpath + str(self.args.cpkt_itr) + self.args.save_extension)
        imsave(result, image_path)
        plt.imshow(result)
        plt.show()
    



#------------------------------------------------------
# functions
#------------------------------------------------------
    def cpkt_save(self, checkpoint_dir, step):
        model_name = "SRCNN.model"
        model_dir = "srcnn"
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)



    def cpkt_load(self, checkpoint_dir, checkpoint_itr):
        print(" [*] Reading checkpoints...")
        model_dir = "srcnn"
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if checkpoint_itr == 0:
            print("train from scratch")
            return True
        
        elif checkpoint_dir == -1:
            ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    
        else:
            ckpt = os.path.join(checkpoint_dir,"SRCNN.model-"+str(checkpoint_itr))

        print(ckpt)
        if ckpt:
            self.saver.restore(self.sess, ckpt)
            return True
        else:
            return False
        
        
#==========================================================
# others
#==========================================================


