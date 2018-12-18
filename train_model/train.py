import os
import numpy as np
import tensorflow as tf
import sys
import random
import cv2
import mtcnn_config as config
from train_model import P_Net

sys.path.append("d:\\VSCodeWorkspace\\MTCNN\\prepare_data")
print(sys.path)
from read_tfrecord_v2 import read_multi_tfrecords,read_single_tfrecord

def image_color_distort(inputs):
    inputs = tf.image.random_contrast(inputs, lower=0.5, upper=1.5)
    inputs = tf.image.random_brightness(inputs, max_delta=0.2)
    inputs = tf.image.random_hue(inputs,max_delta= 0.2)
    inputs = tf.image.random_saturation(inputs,lower = 0.5, upper= 1.5)
    return inputs

def train_model(base_lr,loss,data_num):
    lr_factor = 0.1
    global_step = tf.Variable(0, trainable=False)
    boundaries = [int(epoch * data_num / config.BATCH_SIZE) for epoch in config.LR_EPOCH]
    #lr_values[0.01,0.001,0.0001,0.00001]
    lr_values = [base_lr * (lr_factor ** x) for x in range(0, len(config.LR_EPOCH) + 1)]
    #control learning rate
    lr_op = tf.train.piecewise_constant(global_step, boundaries, lr_values)
    optimizer = tf.train.MomentumOptimizer(lr_op, 0.9)
    train_op = optimizer.minimize(loss, global_step)
    return train_op, lr_op

def random_flip_images(image_batch,label_batch,landmark_batch):
    #mirror
    if random.choice([0,1]) > 0:
        num_images = image_batch.shape[0]
        fliplandmarkindexes = np.where(label_batch==-2)[0]
        flipposindexes = np.where(label_batch==1)[0]
        #only flip
        flipindexes = np.concatenate((fliplandmarkindexes,flipposindexes))
        #random flip    
        for i in flipindexes:
            cv2.flip(image_batch[i],1,image_batch[i])        
        
        #pay attention: flip landmark    
        for i in fliplandmarkindexes:
            landmark_ = landmark_batch[i].reshape((-1,2))
            landmark_ = np.asarray([(1-x, y) for (x, y) in landmark_])
            landmark_[[0, 1]] = landmark_[[1, 0]]#left eye<->right eye
            landmark_[[3, 4]] = landmark_[[4, 3]]#left mouth<->right mouth        
            landmark_batch[i] = landmark_.ravel()
        
    return image_batch,landmark_batch


def train(net_factory,prefix,end_epoch,base_dir,base_lr=0.01,net='PNet'):
    """
    train PNet/RNet/ONet
    :param net_factory:
    :param prefix: model path
    :param end_epoch:
    :param dataset:
    :param display:
    :param base_lr:
    :return:
    """
    net='PNet'
    label_file=os.path.join(base_dir,'train_%s_landmark.txt'%net)
    print(label_file)

    f=open(label_file,'r')
    num=len(f.readlines())
    print("Total size of the dataset is: ", num)
    print(prefix)

    if net=='PNet':
        dataset_dir = os.path.join(base_dir,'train_%s_landmark.tfrecord' % net)
        print('dataset dir is:',dataset_dir)
        image_batch,label_batch,bbox_batch,landmark_batch=read_single_tfrecord(dataset_dir,config.BATCH_SIZE,net)
    
    if net=='PNet':
        image_size=12
        radio_cls_loss=1.0; radio_bbox_loss=0.5; radio_landmark_loss=0.5
    
    input_image = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, image_size, image_size, 3], name='input_image')
    label = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE], name='label')
    bbox_target = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, 4], name='bbox_target')
    landmark_target = tf.placeholder(tf.float32,shape=[config.BATCH_SIZE,10],name='landmark_target')
    #图像增强
    input_image=image_color_distort(input_image)
    cls_loss_op,bbox_loss_op,landmark_loss_op,l2_loss_op,accuracy_op=net_factory(input_image, label, bbox_target,landmark_target,training=True)
    total_loss_op  = radio_cls_loss*cls_loss_op + radio_bbox_loss*bbox_loss_op + radio_landmark_loss*landmark_loss_op + l2_loss_op

    train_op,lr_op=train_model(base_lr,total_loss_op,num)
    init=tf.global_variables_initializer()
    with tf.Session() as sess:
        saver=tf.train.Saver()
        sess.run(init)

        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(sess=sess,coord=coord)
        i=0
        MAX_STEP = int(num / config.BATCH_SIZE + 1) * end_epoch
        epoch = 0

        try:
            for step in range(MAX_STEP):
                i+=1
                image_batch_array, label_batch_array, bbox_batch_array,landmark_batch_array = sess.run([image_batch, label_batch, bbox_batch,landmark_batch])
                #random flip
                
                image_batch_array,landmark_batch_array = random_flip_images(image_batch_array,label_batch_array,landmark_batch_array)
                print(label_batch_array)
                """print(image_batch_array.shape)
                print(label_batch_array.shape)
                print(bbox_batch_array.shape)
                print(landmark_batch_array.shape)"""
                _,_ = sess.run([train_op,lr_op], feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array,landmark_target:landmark_batch_array})
                
                if (step)%50==0:
                    cls_loss, bbox_loss,landmark_loss,l2_loss,lr,acc = sess.run([cls_loss_op, bbox_loss_op,landmark_loss_op,l2_loss_op,lr_op,accuracy_op],
                                                             feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array, landmark_target: landmark_batch_array})
                    
                    total_loss = radio_cls_loss*cls_loss + radio_bbox_loss*bbox_loss + radio_landmark_loss*landmark_loss + l2_loss
                    # landmark loss: %4f,
                    print("Step: %d/%d, accuracy: %3f, cls loss: %4f, bbox loss: %4f,Landmark loss :%4f,L2 loss: %4f, Total Loss: %4f ,lr:%f " % (
                                       step+1,MAX_STEP, acc, cls_loss, bbox_loss,landmark_loss, l2_loss,total_loss, lr))
                #save model
                if i*config.BATCH_SIZE>num:
                    i=0
                    epoch+=1
                    path_prefix =saver.save(sess,prefix,global_step=epoch)
                    print(path_prefix)
        except tf.errors.OutOfRangeError:
            pass
        finally:
            coord.request_stop()
        coord.join(threads) 
        sess.close()    


if __name__=="__main__":
    base_dir=r'D:\DeepLearning\FACE_DATASET\WIDER_DATASET\DATA\imglists\PNet'
    model_name='PNet'
    model_path=r'D:\DeepLearning\FACE_DATASET\WIDER_DATASET\DATA\imglists\PNet'
    end_epoch=3
    lr=0.005
    train(P_Net,model_path,end_epoch,base_dir,base_lr=lr)

