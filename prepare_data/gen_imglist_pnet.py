import numpy as np
import os 
import random

#合并两部分的图像
def gen_image_list(data_dir,size=12):
    if size==12:
        net="PNet"
    elif size==24:
        net="RNet"
    elif size==48:
        net="ONet"
    
    with open(os.path.join(data_dir,"%s/pos_%s.txt"%(size,size)),'r') as f:
        pos=f.readlines()
    
    with open(os.path.join(data_dir,'%s/neg_%s.txt'%(size,size)),'r') as f:
        neg=f.readlines()
    
    with open(os.path.join(data_dir, '%s/part_%s.txt' % (size, size)), 'r') as f:
        part = f.readlines()

    with open(os.path.join(data_dir,'%s/landmark_%s_aug.txt' %(size,size)), 'r') as f:
        landmark = f.readlines()
    
    dir_path=os.path.join(data_dir,"imglists")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if not os.path.exists(os.path.join(dir_path, "%s" %(net))):
        os.makedirs(os.path.join(dir_path, "%s" %(net)))
    
    with open(os.path.join(dir_path,"%s"%(net),"train_%s_landmark.txt"%(net)),"w") as f:
        nums=[len(neg),len(pos),len(part)]
        #ratio=[3,1,1]
        base_num=2000
        print(nums)

        #shuffle the order of the initial data
        #if negative examples are more than 750k then only choose 750k
        if len(neg)>base_num*3:
            neg_keep=np.random.choice(len(neg),size=base_num * 3, replace=True)
        else:
            neg_keep=np.random.choice(len(neg), size=len(neg), replace=True)
        
        pos_keep=np.random.choice(len(pos), size=base_num, replace=True)
        part_keep = np.random.choice(len(part), size=base_num, replace=True)
        print(len(neg_keep),len(pos_keep))

        for i in pos_keep:
            f.write(pos[i])
        for i in neg_keep:
            f.write(neg[i])
        for i in part_keep:
            f.write(part[i])
        for item in landmark:
            f.write(item)


if __name__=="__main__":
    data_dir=r'D:\DeepLearning\FACE_DATASET\WIDER_DATASET\DATA'
    gen_image_list(data_dir,size=12)




