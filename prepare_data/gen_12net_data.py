import os
import cv2
import numpy as np

from utils import IoU

annotation_file=r"D:\DeepLearning\FACE_DATASET\WIDER_DATASET\wider_face_split\wider_face_val.txt"
image_dir=r'D:\DeepLearning\FACE_DATASET\WIDER_DATASET\WIDER_val\images'
pos_save_dir= r"D:\DeepLearning\FACE_DATASET\WIDER_DATASET\DATA\12\positive"
part_save_dir= r"D:\DeepLearning\FACE_DATASET\WIDER_DATASET\DATA\12\part"
neg_save_dir=r'D:\DeepLearning\FACE_DATASET\WIDER_DATASET\DATA\12\negative'
save_dir=r"D:\DeepLearning\FACE_DATASET\WIDER_DATASET\DATA\12"

if not os.path.exists(pos_save_dir):
    os.makedirs(pos_save_dir)
if not os.path.exists(part_save_dir):
    os.makedirs(part_save_dir)
if not os.path.exists(neg_save_dir):
    os.makedirs(neg_save_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#新建数据文件
f1=open(os.path.join(save_dir,'pos_12.txt'),'w')
f2=open(os.path.join(save_dir,'neg_12.txt'),'w')
f3=open(os.path.join(save_dir,'part_12.txt'),'w')

with open(annotation_file,'r') as f:
    annotations=f.readlines()
num=len(annotations)
print("total %d images"%num)

p_idx = 0 # positive
n_idx = 0 # negative
d_idx = 0 # don't care
idx = 0
box_idx = 0

for annotation in annotations:
    annotation=annotation.strip().split(' ')
    image_path=annotation[0]
    #print(annotation)
    bbox=list(map(float,annotation[1:]))  #map是将annotation[1:]转为float
    #groundtruth
    boxes=np.array(bbox,dtype=np.float32).reshape(-1,4)
    img=cv2.imread(os.path.join(image_dir,image_path+'.jpg'))
    #print(img)
    idx+=1
    
    height=img.shape[0]
    width=img.shape[1]
    channel=img.shape[2]
    
    #随机采集50个负样本
    neg_num=0
    while neg_num<50:
        #negative_samples' size min(12,min(width,height)/2)
        size=np.random.randint(12,min(width,height)/2,size=1)[0]
        
        #top_left point
        nx=np.random.randint(0,width-size,size=1)[0]
        
        ny=np.random.randint(0,height-size,size=1)[0]
        
        crop_box=np.array([nx,ny,nx+size,ny+size])
        #计算iou
        iou=IoU(crop_box,boxes)
        if np.max(iou)<0.3:
            crop_img=img[ny:ny+size,nx:nx+size,:]
            resized_img=cv2.resize(crop_img,(12,12),interpolation=cv2.INTER_LINEAR)
            save_file=os.path.join(neg_save_dir,"%s.jpg"%n_idx)
            f2.write(save_file+' 0\n')
            cv2.imwrite(save_file,resized_img)
            n_idx+=1
            neg_num+=1
    
    #for each bounding boxes
    for box in boxes:
        #box(x_left,y_left,x_bottom_right,y)
        x1,y1,x2,y2=box
        #groundtruth (w,h)
        w=x2-x1+1
        h=y2-y1+1
        
        #防止人脸样本太小而不准确，剔除
        if max(w,h)<20 or x1<0 or y1<0:
            continue
        #另外截取人脸周边5张iou<0.3的图像作为负样本
        for i in range(5):
            size=np.random.randint(12,min(width,height)/2,size=1)[0]
            # delta_x and delta_y are offsets of (x1, y1)
            # max can make sure if the delta is a negative number , x1+delta_x >0
            delta_x = np.random.randint(max(-size, -x1), w,size=1)[0]
            delta_y = np.random.randint(max(-size, -y1), h,size=1)[0]
            # max here not really necessary
            nx1 = int(max(0, x1 + delta_x))
            ny1 = int(max(0, y1 + delta_y))
            if nx1 + size > width or ny1 + size > height:
                continue
            crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
            Iou = IoU(crop_box, boxes)
    
            if np.max(Iou) < 0.3:
                # Iou with all gts must below 0.3
                cropped_im = img[ny1: ny1 + size, nx1: nx1 + size, :]
                #rexize cropped image to be 12 * 12
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                f2.write(save_file+' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
        
        #生成正样本
        for i in range(20):
            #pos and part face size range(min(w,h)*0.8,max(w,h)*1.25)
            size=np.random.randint(int(min(w,h)*0.8),np.ceil(1.25*max(w,h)),size=1)[0]
            if w<5:
                print(w)
                continue
            delta_x=np.random.randint(-w*0.2,w*0.2,size=1)[0]
            delta_y=np.random.randint(-h*0.2,h*0.2,size=1)[0]

            nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
            #show this way: ny1 = max(y1+h/2-size/2+delta_y)
            ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
            nx2 = nx1 + size
            ny2 = ny1 + size

            if nx2 > width or ny2 > height:
                continue 
            crop_box = np.array([nx1, ny1, nx2, ny2])
            #yu gt de offset
            offset_x1 = (x1 - nx1) / float(size)
            offset_y1 = (y1 - ny1) / float(size)
            offset_x2 = (x2 - nx2) / float(size)
            offset_y2 = (y2 - ny2) / float(size)
            #crop
            cropped_im = img[ny1 : ny2, nx1 : nx2, :]
            #resize
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)


            box_ = box.reshape(1, -1)
            iou = IoU(crop_box, box_)
            if iou  >= 0.65:
                save_file = os.path.join(pos_save_dir, "%s.jpg"%p_idx)
                f1.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                p_idx += 1
            elif iou >= 0.4:
                save_file = os.path.join(part_save_dir, "%s.jpg"%d_idx)
                f3.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                d_idx += 1
        box_idx += 1
        if idx % 100 == 0:
            print("%s images done, pos: %s part: %s neg: %s" % (idx, p_idx, d_idx, n_idx))
f1.close()
f2.close()
f3.close()
