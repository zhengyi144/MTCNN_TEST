import numpy as np
import tensorflow as tf
slim=tf.contrib.slim 

num_keep_radio = 0.7

def prelu(inputs):
    alphas=tf.get_variable("alphas",shape=inputs.get_shape()[-1],dtype=tf.float32, initializer=tf.constant_initializer(0.25))
    pos=tf.nn.relu(inputs)
    neg=alphas*(inputs-abs(inputs))*0.5
    return pos+neg

def classification_loss(cls_prob,label):
    """
    cls_prob: (batch,2),每张图像划分为两类：1和0
    label: (batch)
    classification loss:(cross-entropy loss)
    这里只取positive，negative的label为1，0，只计算正样本还是负样本的loss，其他类别不计算loss
    """
    zeros=tf.zeros_like(label)
    #除了人脸label为1，其他都为0了
    label_filter_invalid=tf.where(tf.less(label,0),zeros,label)   #tf.less(label,0)大等于0的位置为false，反之为true,tf.where将false的位置由label的值替换
    #size=batch*2
    num_cls_prob=tf.size(cls_prob)    
    #reshape相当于flaten    
    cls_prob_reshape=tf.reshape(cls_prob,[num_cls_prob,-1])
    label_int=tf.cast(label_filter_invalid,tf.int32)    #正样本为1，负样本为0
    #get row number
    num_row=tf.to_int32(cls_prob.get_shape()[0])
    #row=[0,2,4,....]
    row=tf.range(num_row)*2 
    indices=row+label_int      #正样本取奇数值，负样本取偶数值      
    label_prob=tf.squeeze(tf.gather(cls_prob_reshape,indices))
    loss=-tf.log(label_prob+1e-10)
    zeros = tf.zeros_like(label_prob, dtype=tf.float32)
    ones = tf.ones_like(label_prob,dtype=tf.float32)
    # set pos and neg to be 1, rest to be 0
    valid_inds = tf.where(label < 0,zeros,ones)  #与label_filter_invalid是一样的
    num_valid=tf.reduce_sum(valid_inds)       #累计positive和negative有效样本的个数     
    
    keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)

    loss=loss*valid_inds
    loss,_=tf.nn.top_k(loss,k=keep_num)     #剔除一部分loss
    return tf.reduce_mean(loss)

def boundingbox_loss(bbox_pred,bbox_target,label):
    """
    bounding box regression:loss=||pred-target||^2
    """
    zeros=tf.zeros_like(label,dtype=tf.float32)
    ones=tf.ones_like(label,dtype=tf.float32)
    # keep pos and part examples
    valid_index= tf.where(tf.equal(tf.abs(label), 1),ones,zeros) 
    square_error=tf.square(bbox_pred-bbox_target)
    square_error=tf.reduce_mean(square_error,axis=1)   
    num_valid=tf.reduce_sum(valid_index)
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    square_error = square_error*valid_index   #剔除非pos,part的样本
    # keep top k examples, k equals to the number of positive examples
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)

    return tf.reduce_mean(square_error)

def landmark_loss(landmark_pred,landmark_target,label):
    """
    landmark_loss:||landmark_pred-landmark_target||^2
    """
    ones=tf.ones_like(label,dtype=tf.float32)
    zeros=tf.zeros_like(label,dtype=tf.float32)
    valid_index=tf.where(tf.equal(label,-2),ones,zeros)
    square_error=tf.square(landmark_pred-landmark_target)
    square_error=tf.reduce_sum(square_error,axis=1)
    num_valid = tf.reduce_sum(valid_index)
    #keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    #剔除非landmark的样本
    square_error = square_error*valid_index   
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)




def P_Net(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005), 
                        padding='valid'):
        print(inputs.get_shape())
        
        #input=(batch,12,12,3)
        conv1=slim.conv2d(inputs,10,3,stride=1,scope='conv1')  #(batch,10,10,10)
        print(conv1.get_shape())
        pool1=slim.max_pool2d(conv1,kernel_size=[2,2],stride=2,scope='pool1',padding='SAME') #(batch,5,5,10)
        print(pool1.get_shape())
        conv2=slim.conv2d(pool1,16,3,stride=1,scope='conv2') #(batch,3,3,16)
        print(conv2.get_shape())
        conv3=slim.conv2d(conv2,32,3,stride=1,scope='conv3') #(batch,1,1,32)
        #(batch,1,1,2)
        conv4=slim.conv2d(conv3,2,1,stride=1,scope='conv4_1',activation_fn=tf.nn.softmax)   #这里用softmax是防止计算交叉熵出现nan
        print(conv4.get_shape())
        
        #(batch,1,1,4)
        bbox_pred=slim.conv2d(conv3,4,1,stride=1,scope='conv4_2',activation_fn=None)
        print(bbox_pred.get_shape())
        
        #(batch,1,1,10)
        landmark_pred=slim.conv2d(conv3,10,1,stride=1,scope='conv4_3',activation_fn=None)

        if training:
            #classification loss
            cls_prob=tf.squeeze(conv4,[1,2],name='cls_prob')    #squeeze删除维度为1的维，否则报错，相当于降维，(batch,1,1,2)-->(batch,2)
            cls_loss=classification_loss(cls_prob,label)

            #bounding box loss
            bbox_pred=tf.squeeze(bbox_pred,[1,2],name='bbox_pred')
            bbox_loss=boundingbox_loss(bbox_pred,bbox_target,label)

            #landmark_loss
            landmark_pred=tf.squeeze(landmark_pred,[1,2],name='landmark_pred')
            lm_loss=landmark_loss(landmark_pred,landmark_target,label)

            accuracy=cal_accuracy(cls_prob,label)







