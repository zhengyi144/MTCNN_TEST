import numpy as np
import tensorflow as tf
slim=tf.contrib.slim 

def prelu(inputs):
    alphas=tf.get_variable("alphas",shape=inputs.get_shape()[-1],dtype=tf.float32, initializer=tf.constant_initializer(0.25))
    pos=tf.nn.relu(inputs)
    neg=alphas*(inputs-abs(inputs))*0.5
    return pos+neg

def P_Net(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        padding='valid'):
        print(inputs.get_shape())
        
        #input=(batch,12,12,3)
        conv1=slim.conv2d(inputs,10,3,stride=1,scope='conv1')  #(batch,10,10,10)
        print(conv1.get_shape())
        pool1=slim.max_pool2d(conv1,kernel_size=[2,2],stride=2,scope='pool1',padding='SAME') #(batch,6,6,10)
        print(pool1.get_shape())
        conv2=slim.conv2d(pool1,16,3,stride=1,scope='conv2') #(batch,4,4,16)
        print(conv2.get_shape())
        conv3=slim.conv2d(conv2,32,3,stride=1,scope='conv3') #(batch,2,2,32)



