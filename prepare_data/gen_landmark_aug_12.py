import os
import random
import cv2
import numpy as np

from utils import IoU,BBox,getDataFromTxt,flip,rotate

output=r'D:\DeepLearning\FACE_DATASET\WIDER_DATASET\DATA\12'
dstdir=r'D:\DeepLearning\FACE_DATASET\WIDER_DATASET\DATA\12\train_PNet_landmark_aug'

def generateData(ftxt,data_path,net,augmentation=True):
    if net == "PNet":
        size = 12
    elif net == "RNet":
        size = 24
    elif net == "ONet":
        size = 48
    else:
        print('Net type error')
        return
    image_id = 0

    f=open(os.path.join(output,"landmark_%s_aug.txt"%(size)),'w')
    # get image path , bounding box, and landmarks from file 'ftxt'
    data=getDataFromTxt(ftxt,data_path=data_path)
    idx=0
    
    for (imagePath,bbox,landmarkGt) in data:
        images=[]
        landmarks=[]
        img=cv2.imread(imagePath)
        img_h,img_w,img_c=img.shape
        #原图所在的坐标
        if  bbox.right<bbox.left or bbox.bottom<bbox.top:
            continue
        gt_box = np.array([bbox.left,bbox.top,bbox.right,bbox.bottom])
        f_face = img[bbox.top:bbox.bottom+1,bbox.left:bbox.right+1]
        #cv2.imshow("face",f_face)
        #cv2.waitKey(0)
        f_face=cv2.resize(f_face,(size,size))
        landmark=np.zeros((5,2))

        #normalize land mark by dividing the width and height of the ground truth bounding box
        # landmakrGt is a list of tuples
        for index,one in enumerate(landmarkGt):
            # (( x - bbox.left)/ width of bounding box, (y - bbox.top)/ height of bounding box
            rv = ((one[0]-gt_box[0])/(gt_box[2]-gt_box[0]), (one[1]-gt_box[1])/(gt_box[3]-gt_box[1]))
            landmark[index]=rv
        images.append(f_face)
        landmarks.append(landmark.reshape(10))

        landmark=np.zeros((5,2))
        if augmentation:
            idx+=1
            if idx%100==0:
                print(idx,"images done")
            x1,y1,x2,y2=gt_box
            gt_w=x2-x1+1
            gt_h=y2-y1+1
            
            #长宽太小得人脸就不做变换了
            if max(gt_w,gt_h)<40 or x1<0 or y1<0:
                continue
            
            #random shift
            for i in range(10):
                bbox_size=random.randint(int(min(gt_w, gt_h) * 0.8), np.ceil(1.25 * max(gt_w, gt_h)))
                delta_x = random.randint(int(-gt_w * 0.2), int(gt_w * 0.2))
                delta_y = random.randint(int(-gt_h * 0.2), int(gt_h * 0.2))
                nx1 = int(max(x1+gt_w/2-bbox_size/2+delta_x,0))
                ny1 = int(max(y1+gt_h/2-bbox_size/2+delta_y,0))

                nx2 = nx1 + bbox_size
                ny2 = ny1 + bbox_size
                if nx2 > img_w or ny2 > img_h:
                    continue
                crop_box = np.array([nx1,ny1,nx2,ny2])
                cropped_im = img[ny1:ny2+1,nx1:nx2+1,:]
                resized_im = cv2.resize(cropped_im, (size, size))
                #cal iou
                iou = IoU(crop_box, np.expand_dims(gt_box,0))
                if iou>0.65:
                    images.append(resized_im)

                    #normalize
                    for index, one in enumerate(landmarkGt):
                        rv = ((one[0]-nx1)/bbox_size, (one[1]-ny1)/bbox_size)
                        landmark[index] = rv
                    landmarks.append(landmark.reshape(10))
                    
                    landmark=np.zeros((5,2))
                    _landmark=landmarks[-1].reshape(-1,2)
                    bbox=BBox([nx1,ny1,nx2,ny2])

                    #mirror
                    if random.choice([0,1])>0:
                        face_flipped,landmark_flipped=flip(resized_im,_landmark)
                        face_flipped=cv2.resize(face_flipped,(size,size))
                        images.append(face_flipped)
                        landmarks.append(landmark_flipped.reshape(10))
                    
                    #rotate逆时针旋转
                    if random.choice([0,1])>0:
                        #reprojectLandmark将归一化的landmark恢复至原始坐标
                        face_rotated,landmark_rotated=rotate(img,bbox,bbox.reprojectLandmark(_landmark),5)
                        #重新归一化旋转后的landmark
                        landmark_rotated = bbox.projectLandmark(landmark_rotated)
                        face_rotated=cv2.resize(face_rotated,(size,size))
                        images.append(face_rotated)
                        landmarks.append(landmark_rotated.reshape(10))

                        #flip
                        face_flipped, landmark_flipped = flip(face_rotated, landmark_rotated)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        images.append(face_flipped)
                        landmarks.append(landmark_flipped.reshape(10)) 
                    
                    #顺时针rotate
                    if random.choice([0,1])>0:
                        #reprojectLandmark将归一化的landmark恢复至原始坐标
                        face_rotated,landmark_rotated=rotate(img,bbox,bbox.reprojectLandmark(_landmark),-5)
                        #重新归一化旋转后的landmark
                        landmark_rotated = bbox.projectLandmark(landmark_rotated)
                        face_rotated=cv2.resize(face_rotated,(size,size))
                        images.append(face_rotated)
                        landmarks.append(landmark_rotated.reshape(10))

                        #flip
                        face_flipped, landmark_flipped = flip(face_rotated, landmark_rotated)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        images.append(face_flipped)
                        landmarks.append(landmark_flipped.reshape(10)) 
            
            images,landmarks=np.asarray(images),np.asarray(landmarks)
            print(images)
            print(np.shape(landmarks))

            for i in range(len(images)):
                if np.sum(np.where(landmarks[i] <= 0, 1, 0)) > 0:
                    continue

                if np.sum(np.where(landmarks[i] >= 1, 1, 0)) > 0:
                    continue
                
                #保存图片
                cv2.imwrite(os.path.join(dstdir,'%d.jpg'%(image_id)),images[i])
                landmark=map(str,list(landmarks[i]))
                f.write(os.path.join(dstdir,'%d.jpg'%(image_id))+" -2 "+" ".join(landmark)+"\n")
                image_id+=1
    
    f.close()
    return images,landmarks


if __name__=="__main__":
    net='PNet'
    train_txt=r"D:\DeepLearning\FACE_DATASET\Facial_points\trainImageList.txt"
    data_path=r'D:\DeepLearning\FACE_DATASET\Facial_points'

    if not os.path.exists(output):
        os.mkdir(output)
    if not os.path.exists(dstdir):
        os.mkdir(dstdir)
    assert (os.path.exists(dstdir) and os.path.exists(output))

    images,landmarks=generateData(train_txt,data_path,net)











