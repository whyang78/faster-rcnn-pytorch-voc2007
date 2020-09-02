import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
from scipy.misc import imread
import cv2
from model.utils.config import cfg

# 每个批次的样本是由索引连续的几个样本组成
class sampler(Sampler):
    def __init__(self,trainsize,batchsize):
        self.train_size=trainsize
        self.batch_size=batchsize
        self.num_batch=int(self.train_size/self.batch_size)
        self.range=torch.arange(0,batchsize).view(1,-1).long()

        self.leftover_flag=False
        if self.train_size%self.batch_size:
            self.leftover_flag=True
            self.leftover=torch.arange(self.num_batch*self.batch_size,self.train_size).long()

    def __iter__(self):
        self.randnum=torch.randperm(self.num_batch).view(-1,1)*self.batch_size
        self.randnum=self.randnum.expand(self.num_batch,self.batch_size)+self.range
        self.randnum=self.randnum.view(-1).long()

        if self.leftover_flag:
            self.randnum=torch.cat((self.randnum,self.leftover),dim=0)
        return iter(self.randnum)

    def __len__(self):
        return self.train_size


class roibatchloader(Dataset):
    def __init__(self,roidbs, ratio_list, ratio_index,batch_size, num_classes, training=True, normalize=None):
        self.roidbs=roidbs
        self.ratio_list=ratio_list
        self.ratio_index=ratio_index
        self.batchsize=batch_size
        self.num_classes=num_classes
        self.training=training
        self.normalize=normalize
        self.gt_max_num=cfg.MAX_NUM_GT_BOXES #每个样本拥有的最多gt框个数，gt框指的是属于前景类别的框

        self.data_size=len(self.roidbs)
        self.num_batch=int(np.ceil(self.data_size/self.batchsize))
        self.ratio_batch_list=torch.Tensor(self.data_size).zero_()
        for i in range(self.num_batch):
            left_index=i*self.batchsize
            right_index=min((i+1)*batch_size-1,self.data_size-1)

            if self.ratio_list[right_index]<1.0:
                ratio=self.ratio_list[left_index]
            elif self.ratio_list[left_index]>1.0:
                ratio=self.ratio_list[right_index]
            else:
                ratio=1.0
            self.ratio_batch_list[left_index:(right_index+1)]=ratio

    def __getitem__(self, index):
        '''
        :return:
        训练集：
                训练时会读取一个batchsize的图像样本进入网络，但是不同图像的size不同，故先进行放缩操作，
            使得每个图像的最短边为cfg.TRAIN.SCALES，此处为600；再设置每个batch的目标ratio一致，这样最长边
            就可以根据最短边与ratio得到，也是一固定值。最终同一batch中每个图像的宽高一致。
                除此之外，训练时还会读取一个batchsize的框坐标样本进行网络的训练，但是不同图像样本对应的
            框数目不同，故需要设置框数目固定值cfg.MAX_NUM_GT_BOXES，少于该值则填充0，多于该值则取该值数目
            样本。并且每个框的类别都是对应前景类别的。
                训练部分需要对图像样本进行均值化、翻转、放缩、裁剪、填充等操作。其中翻转、放缩、裁剪都对
            图像样本中框坐标有所影响，此处填充是在(0,0)点开始填充，故不影响原坐标。
            padding_data：维度[3,h,w]
                可以举个例子，某个批次的ratio都小于1.0(设ratio=0.6)，即 w<h。此时会将所有样本的w放缩到600，即w=600；
                h则会根据w和ratio得到一固定值，即h=w/ratio=1000，将原始图像填充进去。故所有图像为[3,1000,600]
            im_info：维度[3]
            gt_boxes_padding：维度[cfg.MAX_NUM_GT_BOXES,5]
                前四列为坐标(x1,y1,x2,y2)，最后一列为对应的前景的类别。
            num_boxes：维度[]，非矩阵，为一常值
        测试集：
                测试时好像一张一张图像进入。测试时batchsize=1。

        '''
        if self.training:
            index_ratio=int(self.ratio_index[index]) # ratio排序后对应的样本序号
        else:
            index_ratio=index

        minibatch_db=[self.roidbs[index_ratio]]
        blobs=self.get_minibatch(minibatch_db,self.num_classes)
        data=torch.from_numpy(blobs['data'])
        im_info=torch.from_numpy(blobs['im_info'])

        height,width=data.shape[1],data.shape[2]
        if self.training:
            np.random.shuffle(blobs['gt_boxes'])
            gt_boxes=torch.from_numpy(blobs['gt_boxes'])
            ratio=self.ratio_batch_list[index] # 目标ratio
            if self.roidbs[index_ratio]['need_crop']:
                if ratio<1.0: # w < h，故裁剪 h
                    trim_height=int(np.floor(width/ratio))
                    min_y=int(torch.min(gt_boxes[:,1]))
                    max_y=int(torch.max(gt_boxes[:,3]))

                    if trim_height>height:
                        trim_height=height

                    if min_y==0:
                        y_s=0
                    else:
                        boxes_region=max_y-min_y+1
                        if boxes_region<trim_height:
                            y_s_min=max(0,max_y-trim_height)
                            y_s_max=min(min_y,height-trim_height)
                            if y_s_min==y_s_max:
                                y_s=y_s_min
                            else:
                                y_s=np.random.choice(range(y_s_min,y_s_max))
                        else:
                            y_s_add=int((boxes_region-trim_height)/2)
                            if y_s_add==0:
                                y_s=min_y
                            else:
                                y_s=np.random.choice(range(min_y,min_y+y_s_add))
                    data=data[:,y_s:(y_s+trim_height),:,:]
                    gt_boxes[:,1]=gt_boxes[:,1]-float(y_s)
                    gt_boxes[:,3]=gt_boxes[:,3]-float(y_s)

                    gt_boxes[:,1].clamp_(0,trim_height-1)
                    gt_boxes[:,3].clamp_(0,trim_height-1)
                else: # w > h，故裁剪 w
                    trim_width=int(np.ceil(height*ratio))
                    min_x=int(torch.min(gt_boxes[:,0]))
                    max_x=int(torch.max(gt_boxes[:,2]))

                    if trim_width>width:
                        trim_width=width

                    if min_x==0:
                        x_s=0
                    else:
                        boxes_region=max_x-min_x+1
                        if boxes_region<trim_width:
                            x_s_min=max(0,max_x-trim_width)
                            x_s_max=min(min_x,width-trim_width)
                            if x_s_min==x_s_max:
                                x_s=x_s_min
                            else:
                                x_s=np.random.choice(range(x_s_min,x_s_max))
                        else:
                            x_s_add=int((boxes_region-trim_width)/2)
                            if x_s_add==0:
                                x_s=min_x
                            else:
                                x_s=np.random.choice(range(min_x,min_x+x_s_add))
                    data=data[:,:,x_s:(x_s+trim_width),:]
                    gt_boxes[:,0]=gt_boxes[:,0]-float(x_s)
                    gt_boxes[:,2]=gt_boxes[:,2]-float(x_s)

                    gt_boxes[:,0].clamp_(0,trim_width-1)
                    gt_boxes[:,2].clamp_(0,trim_width-1)

            # ratio != 1.0时进行填充操作，反之进行裁剪操作
            if ratio<1.0: # w<h
                padding_data=torch.FloatTensor(int(np.ceil(width/ratio)),width,3).zero_()
                padding_data[:height,:,:]=data[0]
                im_info[0][0]=padding_data.shape[0]
            elif ratio>1.0: # w > h
                padding_data=torch.FloatTensor(height,int(np.ceil(height*ratio)),3).zero_()
                padding_data[:,:width,:]=data[0]
                im_info[0][1]=padding_data.shape[1]
            else:
                trim_size=min(width,height)
                padding_data=data[0][:trim_size,:trim_size,:]
                im_info[0][0]=trim_size
                im_info[0][1]=trim_size
                gt_boxes[:,:4].clamp_(0,trim_size)

            not_keep=(gt_boxes[:,0]==gt_boxes[:,2])|(gt_boxes[:,1]==gt_boxes[:,3])
            keep=torch.nonzero(not_keep==0).view(-1)
            # 每个样本的框的个数不能超过gt_max_num
            gt_boxes_padding=torch.FloatTensor(self.gt_max_num,gt_boxes.shape[1]).zero_()
            if keep.numel()!=0:
                gt_boxes=gt_boxes[keep]
                num_boxes=min(self.gt_max_num,gt_boxes.shape[0])
                gt_boxes_padding[:num_boxes,:]=gt_boxes[:num_boxes,:]
            else:
                num_boxes=0

            padding_data=padding_data.permute(2,0,1).contiguous()
            im_info=im_info.view(3)
            return padding_data,im_info,gt_boxes_padding,num_boxes
        else:
            data=data.permute(0,3,1,2).contiguous().view(3,height,width)
            im_info=im_info.view(3)
            num_boxes=0
            gt_boxes=torch.FloatTensor([1, 1, 1, 1, 1])
            return data,im_info,gt_boxes,num_boxes

    def __len__(self):
        return len(self.ratio_list)

    def get_minibatch(self,roidb,num_classes):
        num_images=len(roidb)
        # 其实传进来的样本数目为 1
        rand_scale_list=np.random.randint(0,len(cfg.TRAIN.SCALES),size=num_images)

        assert (cfg.TRAIN.BATCH_SIZE % num_images ==0),\
               'num images ({}) must divide Batch Size ({})'.\
                format(num_images, cfg.TRAIN.BATCH_SIZE)

        im_blob,im_scales=self.get_image_blob(roidb,rand_scale_list)
        blobs={'data':im_blob}
        assert len(im_blob)==1
        assert len(im_scales)==1

        gt_inds=np.nonzero(roidb[0]['gt_classes']!=0)[0] # 对应前景类别的框
        gt_boxes=np.zeros((len(gt_inds),5),dtype=np.float32)
        boxes=roidb[0]['boxes']
        gt_boxes[:,:4]=boxes[gt_inds,:] * im_scales[0] # 样本放缩后，坐标也要有相应的变化
        gt_boxes[:,4]=roidb[0]['gt_classes'][gt_inds]
        blobs['gt_boxes']=gt_boxes
        blobs['im_info']=np.array([[im_blob.shape[1],im_blob.shape[2],im_scales[0]]],dtype=np.float32)
        blobs['img_id']=roidb[0]['img_id']
        return blobs


    def get_image_blob(self,roidb,rand_scale_list):
        # 对样本进行中心化和翻转、放缩等操作
        num_images=len(roidb)

        im_list=[]
        im_scale_list=[]
        for i in range(num_images):
            image=imread(roidb[i]['image'])
            if len(image.shape)==2:
                image=image[:,:,np.newaxis]
                image=np.concatenate((image,image,image),axis=2)
            image=image[:,:,::-1]
            if roidb[i]['flipped']:
                image=image[:,::-1,:]
            target_size = cfg.TRAIN.SCALES[rand_scale_list[i]]
            im, im_scale = self.prep_im_for_blob(image, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE)
            im_list.append(im)
            im_scale_list.append(im_scale)

        blob=self.im_list_to_blob(im_list)
        return blob,im_scale_list

    def prep_im_for_blob(self,image,pixel_mean,target_size,max_size):
        img=image.astype(np.float32,copy=False)
        img-=pixel_mean
        min_shape=min(img.shape[:2])
        img_scale=float(target_size)/float(min_shape)
        img=cv2.resize(img,None,None,fx=img_scale,fy=img_scale,interpolation=cv2.INTER_LINEAR)
        return img,img_scale

    def im_list_to_blob(self,ims):
        num_images=len(ims)
        max_shape=np.array([im.shape for im in ims]).max(axis=0)
        blob=np.zeros((num_images,max_shape[0],max_shape[1],3),dtype=np.float32)
        for i in range(num_images):
            im=ims[i]
            blob[i,:im.shape[0],:im.shape[1],:]=im
        return blob


