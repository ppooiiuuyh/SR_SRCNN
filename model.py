import os
import time

import tensorflow as tf

from utils import read_h5data, input_setup, imsave, merge
import matplotlib.pyplot as plt


class SRCNN(object):
#==========================================================
# class initializer
#==========================================================
    def __init__(self,
               sess,
               image_size=33,
               label_size=21,
               batch_size=128,
               c_dim=3,
               checkpoint_dir=None,
               result_dir=None):

        self.sess = sess
        self.is_grayscale = (c_dim == 1)
        self.image_size = image_size
        self.label_size = label_size
        self.batch_size = batch_size

        self.c_dim = c_dim

        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.build_model()




#==========================================================
# build model
#==========================================================
    def build_model(self):
    #------------------------------------------------------
    # [build model] / set placeholder
    #------------------------------------------------------
        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images')
        self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels')


    #------------------------------------------------------
    # [build model] / build inner network
    #------------------------------------------------------
        self.pred = self.inner_model()



    def inner_model(self):
    #------------------------------------------------------
    # [inner model] / prepare weights
    #------------------------------------------------------
        self.weights = {
            'w1': tf.Variable(tf.random_normal([9, 9, 3, 64], stddev=1e-3), name='w1'),
            'w2': tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3), name='w2'),
            'w3': tf.Variable(tf.random_normal([5, 5, 32, 3], stddev=1e-3), name='w3')
        }

        self.biases = {
            'b1': tf.Variable(tf.zeros([64]), name='b1'),
            'b2': tf.Variable(tf.zeros([32]), name='b2'),
            'b3': tf.Variable(tf.zeros([3]), name='b3')
        }

    #------------------------------------------------------
    # [inner model] / build inner model
    #------------------------------------------------------
        with tf.variable_scope("block1") as scope:
            conv1 = tf.layers.conv2d(self.images, 64, [9, 9], strides=[1, 1], padding='VALID', kernel_initializer=tf.random_normal_initializer(0, 0.001),activation=tf.nn.relu)
            conv2 = tf.layers.conv2d(conv1, 32, [1, 1], strides=[1, 1], padding='VALID', kernel_initializer=tf.random_normal_initializer(0, 0.001),activation=tf.nn.relu)
        with tf.variable_scope("block2") as scope:
            #conv3 = tf.nn.conv2d(conv2, self.weights['w3'], strides=[1, 1, 1, 1], padding='VALID') + self.biases['b3']
            conv3 = tf.layers.conv2d(conv2, 3, [5, 5], strides=[1,1], padding='VALID',kernel_initializer=tf.random_normal_initializer(0,0.001))
        return conv3


#==========================================================
# train and test
#==========================================================
    def train(self, args):
    #------------------------------------------------------
    # [train] / setup
    #------------------------------------------------------
        nx, ny = input_setup(self.sess, args)
        data_dir = os.path.join('./{}'.format(args.checkpoint_dir), args.is_train+".h5")
        train_data, train_label = read_h5data(data_dir)


        """
        print(train_data.shape)
        print(train_data[0, :, :, 0].shape)

        sample_size = [10,10]
        data = train_label
        fig, ax = plt.subplots(nrows=sample_size[0], ncols=sample_size[1], figsize=sample_size)
        for r in range(sample_size[0]):
            for c in range(sample_size[1]):
                ax[r, c].set_axis_off()
                ax[r, c].imshow(data[r*sample_size[1]+c,:,:,0],cmap='gray')

        data = train_data
        fig2, ax2 = plt.subplots(nrows=sample_size[0], ncols=sample_size[1], figsize=sample_size)
        for r in range(sample_size[0]):
            for c in range(sample_size[1]):
                ax2[r, c].set_axis_off()
                ax2[r, c].imshow(data[r * sample_size[1] + c, :, :, 0], cmap='gray')

        plt.show()
        """

    #------------------------------------------------------
    # [train] / set train tensors
    #------------------------------------------------------
        self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
        self.train_op = tf.train.GradientDescentOptimizer(args.lr).minimize(self.loss)
        var_list1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='block1')
        var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='block2')
        print(var_list2)
        train_op1 = tf.train.AdamOptimizer(0.0001).minimize(self.loss, var_list=var_list1)
        train_op2 = tf.train.AdamOptimizer(0.00001).minimize(self.loss, var_list=var_list2)
        self.train_op = tf.group(train_op1, train_op2)
    
        self.saver = tf.train.Saver(max_to_keep=200)

    #------------------------------------------------------
    # [train] / init and load models
    #------------------------------------------------------
        tf.initialize_all_variables().run()

        if self.load(self.checkpoint_dir, args.cpkt_itr):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")


    #------------------------------------------------------
    # [train] / train
    #------------------------------------------------------
        if args.is_train == "train":
            print("Training...")

            start_time = time.time()
            counter = 0
            for ep in range(args.epoch):
                # Run by batch images
                batch_idxs = len(train_data) // args.batch_size
                for idx in range(0, batch_idxs):
                    batch_images = train_data[idx*args.batch_size : (idx+1)*args.batch_size]
                    batch_labels = train_label[idx*args.batch_size : (idx+1)*args.batch_size]
                    counter += 1
                    result, err = self.sess.run([self.train_op, self.loss], feed_dict={self.images: batch_images, self.labels: batch_labels})

                if ep % 10 == 0:
                    print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
                  % ((ep+1), counter, time.time()-start_time, err))

                if ep % 500 == 0:
                    self.save(args.checkpoint_dir, ep)
                    result = self.sess.run(self.pred,feed_dict={self.images: train_data[0:nx*ny].reshape(-1,33,33,3) } )
                    result = merge(result, [nx, ny])
                    result = result.squeeze()
                    #plt.imshow(result)
                    #plt.show()


    #------------------------------------------------------
    # [train] / test
    #------------------------------------------------------
        elif args.is_train == "test":
            print("Testing...")

            result,loss = sess.run([self.pred,loss],feed_dict = {self.images: train_data, self.labels: train_label})
            result = merge(result, [nx, ny])
            result = result.squeeze()
            image_path = os.path.join(os.getcwd(), args.result_dir)
            image_path = os.path.join(image_path, "test_image.png")
            imsave(result, image_path)

    #------------------------------------------------------
    # [train] / inference
    #------------------------------------------------------
        elif args.is_train == "inference":
            print("inferring...")
            start_time = time.time()
            result = self.pred.eval({self.images: train_data})
            print("Elapse:", time.time() - start_time)
    
            result = merge(result, [nx, ny])
            result = result.squeeze()
            image_path = os.path.join(os.getcwd(), args.result_dir)
            image_path = os.path.join(image_path,args.is_train)
            image_path = os.path.join(image_path,args.infer_imgpath + str(args.cpkt_itr) + args.save_extension)
            imsave(result, image_path)
            plt.imshow(result)
            plt.show()
    



    #------------------------------------------------------
    # [train] / functions
    #------------------------------------------------------
    def save(self, checkpoint_dir, step):
        model_name = "SRCNN.model"
        model_dir = "%s_%s" % ("srcnn", self.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)


    def load(self, checkpoint_dir, checkpoint_itr):
        print(" [*] Reading checkpoints...")
        model_dir = "%s_%s" % ("srcnn", self.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if checkpoint_itr == -1:
            print("train from scratch")
            return True
        
        else :
            if checkpoint_itr == None:
                ckpt = tf.train.latest_checkpoint(checkpoint_dir)
            else :
                ckpt = os.path.join(checkpoint_dir,"SRCNN.model-"+str(checkpoint_itr))
            print(ckpt)
    
            if ckpt:
                self.saver.restore(self.sess, ckpt)
                return True
            else:
                return False
        
        return False
        
        """
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print(ckpt_name)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
        """

#==========================================================
# others
#==========================================================


