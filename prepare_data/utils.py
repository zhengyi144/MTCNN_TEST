import numpy as np
import os
import cv2
import tensorflow as tf

def IoU(box,boxes):
    """
    compute IoU between detect box and gt boxes
    box=shape(5,):x1,y1,x2,y2,score
    gt boxes=shape(n,4):x1,y1,x2,y2

    return
    ovr: shape(n,1) IOU
    """
    box_area=(box[2]-box[0]+1)*(box[3]-box[1]+1)
    area= (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    w=np.maximum(0,xx2-xx1+1)
    h=np.maximum(0,yy2-yy1+1)

    inter=w*h
    ovr=inter/(box_area+area-inter)
    return ovr

def getDataFromTxt(txt,data_path,with_landmark=True):
    """
      generate data from txt file
      return [(img_path, bbox, landmark)]
            bbox: [left,  top, right,bottom]
            landmark: [(x1, y1), (x2, y2), ...]
    """
    with open(txt,'r') as reader:
        lines=reader.readlines()
    
    result=[]
    for line in lines:
        line=line.strip()
        components=line.split(' ')
        image_path=os.path.join(data_path,components[0]).replace("\\",'/')

        # bounding box, (left,  top, right,bottom)
        bbox=(components[1],components[3],components[2],components[4])
        bbox=[float(i) for i in bbox]
        bbox=list(map(int,bbox))

        if not with_landmark:
            result.append((image_path,BBox(bbox)))
            continue
        
        landmark=np.zeros((5,2))
        for index in range(0,5):
            rv = (float(components[5+2*index]), float(components[5+2*index+1]))
            landmark[index] = rv
        
        result.append((image_path,BBox(bbox),landmark))
    return result
        

class BBox(object):
    """
       bounding box of face
    """
    def __init__(self,bbox):
        self.left=bbox[0]
        self.top=bbox[1]
        self.right=bbox[2]
        self.bottom=bbox[3]

        self.x=bbox[0]
        self.y=bbox[1]
        self.w=self.right-self.left
        self.h=self.bottom-self.top
    
    #扩展bbox
    def expand(self,scale=0.05):
        bbox=[self.left, self.right, self.top, self.bottom]
        bbox[0] -= int(self.w * scale)
        bbox[1] += int(self.w * scale)
        bbox[2] -= int(self.h * scale)
        bbox[3] += int(self.h * scale)
        return BBox(bbox)
    
    #absolute position(image (left,top))
    def reproject(self,point):
        x=self.x+self.w*point[0]
        y=self.y+self.h*point[1]
        return np.asarray([x,y])
    
    def reprojectLandmark(self,landmark):
        #len(landmark)=5
        p=np.zeros((len(landmark),2))
        for i in range(len(landmark)):
            p[i]=self.reproject(landmark[i])
        return p
    
    #landmark offset（归一化）
    def project(self,point):
        x=(point[0]-self.x)/self.w
        y=(point[1]-self.y)/self.h
        return np.asarray([x,y])
    
    def projectLandmark(self,landmark):
        p=np.zeros((len(landmark),2))
        for i in range(len(landmark)):
            p[i]=self.project(landmark[i])
        return p

#翻转
def flip(face,landmark):
    face_flipped_by_x=cv2.flip(face,1)  #沿水平方向翻转180
    #mirror
    _landmark=np.asarray([(1-x,y) for (x,y) in landmark])  #landmark归一化了
    #调换下存储顺序
    _landmark[[0,1]]=_landmark[[1,0]] #left eye<->right eye
    _landmark[[3,4]]=_landmark[[4,3]] #left mouth<->right mouth
    return (face_flipped_by_x,_landmark)

#旋转
def rotate(img,bbox,landmark,alpha):
    '''
    img:original image
    bbox:original face bounding box
    landmark: original face landmark
    alpha: 旋转角度
    ''' 
    center=((bbox.left+bbox.right)/2,(bbox.top+bbox.bottom)/2)
    rot_mat=cv2.getRotationMatrix2D(center,alpha,1)
    img_rotated=cv2.warpAffine(img,rot_mat,(img.shape[1],img.shape[0]))
    #转换landmark的坐标
    _landmark=np.asarray([(rot_mat[0][0]*x+rot_mat[0][1]*y+rot_mat[0][2],\
                 rot_mat[1][0]*x+rot_mat[1][1]*y+rot_mat[1][2]) for (x, y) in landmark])
    face=img_rotated[bbox.top:bbox.bottom+1,bbox.left:bbox.right+1]
    return(face,_landmark)

#将图像数据转为string的类型
def process_image_withoutcoder(filename):
    image=cv2.imread(filename)
    image_data=image.tostring()
    assert len(image.shape)==3
    height=image.shape[0]
    width=image.shape[1]
    assert image.shape[2]==3
    return image_data,height,width 

def _bytes_feature(value):
    if not isinstance(value,list):
        value=[value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _int64_feature(value):
    """Wrapper for insert int64 feature into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    """Wrapper for insert float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def convert_to_example_simple(image_example,image_buffer):
    """
    convert to tfrecord file
    """
    class_label = image_example['label']
    bbox = image_example['bbox']
    roi = [bbox['xmin'],bbox['ymin'],bbox['xmax'],bbox['ymax']]
    landmark = [bbox['xlefteye'],bbox['ylefteye'],bbox['xrighteye'],bbox['yrighteye'],bbox['xnose'],bbox['ynose'],
                bbox['xleftmouth'],bbox['yleftmouth'],bbox['xrightmouth'],bbox['yrightmouth']]
                
      
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(image_buffer),
        'image/label': _int64_feature(class_label),
        'image/roi': _float_feature(roi),
        'image/landmark': _float_feature(landmark)
    }))
    return example


