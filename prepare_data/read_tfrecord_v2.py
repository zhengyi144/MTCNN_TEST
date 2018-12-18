import tensorflow as tf
import numpy as np
import cv2
import os

def read_single_tfrecord(tfrecord_file, batch_size, net):
    # generate a input queue
    # each epoch shuffle
    filename_queue = tf.train.string_input_producer([tfrecord_file],shuffle=True)
    # read tfrecord
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    image_features = tf.parse_single_example(
        serialized_example,
        features={
            'image/encoded': tf.FixedLenFeature([], tf.string),#one image  one record
            'image/label': tf.FixedLenFeature([], tf.int64),
            'image/roi': tf.FixedLenFeature([4], tf.float32),
            'image/landmark': tf.FixedLenFeature([10],tf.float32)
        }
    )
    if net == 'PNet':
        image_size = 12
    elif net == 'RNet':
        image_size = 24
    else:
        image_size = 48
    image = tf.decode_raw(image_features['image/encoded'], tf.uint8)
    image = tf.reshape(image, [image_size, image_size, 3])
    image = (tf.cast(image, tf.float32)-127.5) / 128
    
    # image = tf.image.per_image_standardization(image)
    label = tf.cast(image_features['image/label'], tf.float32)
    roi = tf.cast(image_features['image/roi'],tf.float32)
    landmark = tf.cast(image_features['image/landmark'],tf.float32)
    image, label,roi,landmark = tf.train.batch(
        [image, label,roi,landmark],
        batch_size=batch_size,
        num_threads=2,
        capacity=1 * batch_size
    )
    label = tf.reshape(label, [batch_size])
    roi = tf.reshape(roi,[batch_size,4])
    landmark = tf.reshape(landmark,[batch_size,10])
    return image, label, roi,landmark

def read_multi_tfrecords(tfrecord_files, batch_sizes, net):
    pos_dir,part_dir,neg_dir,landmark_dir = tfrecord_files
    pos_batch_size,part_batch_size,neg_batch_size,landmark_batch_size = batch_sizes
    #assert net=='RNet' or net=='ONet', "only for RNet and ONet"
    pos_image,pos_label,pos_roi,pos_landmark = read_single_tfrecord(pos_dir, pos_batch_size, net)
    print(pos_image.get_shape())
    part_image,part_label,part_roi,part_landmark = read_single_tfrecord(part_dir, part_batch_size, net)
    print(part_image.get_shape())
    neg_image,neg_label,neg_roi,neg_landmark = read_single_tfrecord(neg_dir, neg_batch_size, net)
    print(neg_image.get_shape())
    landmark_image,landmark_label,landmark_roi,landmark_landmark = read_single_tfrecord(landmark_dir, landmark_batch_size, net)
    print(landmark_image.get_shape())

    images = tf.concat([pos_image,part_image,neg_image,landmark_image], 0, name="concat/image")
    print(images.get_shape())
    labels = tf.concat([pos_label,part_label,neg_label,landmark_label],0,name="concat/label")
    print
    assert isinstance(labels, object)
    labels.get_shape()
    rois = tf.concat([pos_roi,part_roi,neg_roi,landmark_roi],0,name="concat/roi")
    print( rois.get_shape())
    landmarks = tf.concat([pos_landmark,part_landmark,neg_landmark,landmark_landmark],0,name="concat/landmark")
    return images,labels,rois,landmarks
    
