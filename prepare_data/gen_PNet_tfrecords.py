import sys
import time
import os
import random
import tensorflow as tf
from utils import convert_to_example_simple,process_image_withoutcoder


def convertToTFRecord(data_dir,net,output_dir,name='MTCNN',shuffling=False):
    tf_filename='%s/train_PNet_landmark.tfrecord'%(output_dir)
    if tf.gfile.Exists(tf_filename):
        print('tf_record files already exist!')
        return

    dataset=get_dataset(data_dir)

    if shuffling:
        random.shuffle(dataset)
    
    with tf.python_io.TFRecordWriter(tf_filename) as writer:
        for i,image_example in enumerate(dataset):
            filename=image_example["filename"]
            addToTFRecord(filename,image_example,writer)
    
    writer.close()
    print("finished convert dataset to tfrecord!")

def addToTFRecord(filename,image_example,writer):
    image_data,height,width=process_image_withoutcoder(filename)
    example=convert_to_example_simple(image_example,image_data)
    writer.write(example.SerializeToString())

def get_dataset(data_dir,net='PNet'):
    file='imglists/PNet/train_%s_landmark.txt' % net
    file_path=os.path.join(data_dir,file)
    reader=open(file_path,'r')

    dataset=[]
    for line in reader.readlines():
        info=line.strip().split(' ')
        data_example=dict()
        bbox=dict()
        data_example['filename']=info[0]
        #print(data_example['filename'])
        data_example['label']=int(info[1])
        bbox['xmin'] = 0
        bbox['ymin'] = 0
        bbox['xmax'] = 0
        bbox['ymax'] = 0
        bbox['xlefteye'] = 0
        bbox['ylefteye'] = 0
        bbox['xrighteye'] = 0
        bbox['yrighteye'] = 0
        bbox['xnose'] = 0
        bbox['ynose'] = 0
        bbox['xleftmouth'] = 0
        bbox['yleftmouth'] = 0
        bbox['xrightmouth'] = 0
        bbox['yrightmouth'] = 0        
        if len(info) == 6:
            bbox['xmin'] = float(info[2])
            bbox['ymin'] = float(info[3])
            bbox['xmax'] = float(info[4])
            bbox['ymax'] = float(info[5])
        if len(info) == 12:
            bbox['xlefteye'] = float(info[2])
            bbox['ylefteye'] = float(info[3])
            bbox['xrighteye'] = float(info[4])
            bbox['yrighteye'] = float(info[5])
            bbox['xnose'] = float(info[6])
            bbox['ynose'] = float(info[7])
            bbox['xleftmouth'] = float(info[8])
            bbox['yleftmouth'] = float(info[9])
            bbox['xrightmouth'] = float(info[10])
            bbox['yrightmouth'] = float(info[11])
            
        data_example['bbox'] = bbox
        dataset.append(data_example)
    return dataset


if __name__=='__main__':
    data_dir=r'D:\DeepLearning\FACE_DATASET\WIDER_DATASET\DATA'
    net='PNet'
    output_dir=r'D:\DeepLearning\FACE_DATASET\WIDER_DATASET\DATA\imglists\PNet'
    convertToTFRecord(data_dir,net,output_dir)
