import _init_path
import os
import numpy as np
import argparse
import cv2
import torch
from scipy.misc import imread
from model.utils.config import cfg, cfg_from_file, cfg_from_list
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv,clip_boxes
from model.utils.net_utils import vis_detections
from model.faster_rcnn.vgg16 import vgg16

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/vgg16.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models',
                      default="/srv/share/jyang375/models")
  parser.add_argument('--image_dir', dest='image_dir',
                      help='directory to load images for demo',
                      default="images")
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=10021, type=int)
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  parser.add_argument('--webcam_num', dest='webcam_num',
                      help='webcam ID number',
                      default=-1, type=int)

  args = parser.parse_args()
  return args

def im_list_to_blob(im_list):
    num_images=len(im_list)
    blob_shape=np.array([im.shape for im in im_list]).max(axis=0)
    blob=np.zeros((num_images,blob_shape[0],blob_shape[1],3),dtype=np.float32)
    for i in range(num_images):
        im=im_list[i]
        blob[i,:im.shape[0],:im.shape[1],:]=im
    return blob

def get_image_blob(im):
    im_origin=im.astype(np.float32,copy=True)
    im_origin-=cfg.PIXEL_MEANS

    im_shape=im_origin.shape
    im_shape_max=np.max(im_shape[:2])
    im_shape_min=np.min(im_shape[:2])

    processed_ims=[]
    im_scales=[]
    for target_size in cfg.TEST.SCALES:
        scale=target_size/im_shape_min
        # 若缩放后较长边超过设置最大值，则按照设置最大值缩放尺寸进行缩放
        if np.round(scale*im_shape_max)>cfg.TEST.MAX_SIZE:
            scale=cfg.TEST.MAX_SIZE/im_shape_max
        im=cv2.resize(im_origin,None,None,fx=scale,fy=scale,interpolation=cv2.INTER_LINEAR)
        processed_ims.append(im)
        im_scales.append(scale)

    blob=im_list_to_blob(processed_ims)
    return blob,np.array(im_scales)

if __name__ == '__main__':
    # cfg载入
    args=parse_args()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    # 超参数设置
    cfg.USE_GPU_NMS = args.cuda
    webcam_num=args.webcam_num # -1表示不使用摄像头，而是读取文件夹图像；非负时表示使用摄像头，读取摄像头图像
    det_thresh=0.05 # 设置rois类别置信度阈值（初步筛选阈值）
    vis=True # 是否可视化检测结果，可以传参args.vis进行设置

    # 初始化模型
    # 注意此处类别要与训练测试的类别一致，当自定义数据集时，可以自行修改
    pascal_classes = np.asarray(['__background__',
                                 'aeroplane', 'bicycle', 'bird', 'boat',
                                 'bottle', 'bus', 'car', 'cat', 'chair',
                                 'cow', 'diningtable', 'dog', 'horse',
                                 'motorbike', 'person', 'pottedplant',
                                 'sheep', 'sofa', 'train', 'tvmonitor'])
    if args.net == 'vgg16':
        fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic)
    fasterRCNN.create_architecture()

    # 载入模型参数
    input_dir=args.load_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_model_path=os.path.join(input_dir,'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

    if args.cuda:
        checkpoints=torch.load(load_model_path)
    else:
        checkpoints=torch.load(load_model_path,map_location=(lambda storage, loc: storage))
    fasterRCNN.load_state_dict(checkpoints['model'])

    if 'pooling_mode' in checkpoints.keys():
        cfg.POOLING_MODE = checkpoints['pooling_mode']

    # skip to cuda
    if args.cuda:
        cfg.CUDA = True

    if args.cuda:
        fasterRCNN.cuda()
    fasterRCNN.eval() # 设置测试模式，必须有

    # 初始化输入数据
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # 数据集
    if webcam_num>=0:
        # 摄像头读取
        cap=cv2.VideoCapture(webcam_num)
        num_images=0
    else:
        # 文件读取
        images_list=os.listdir(args.image_dir)
        num_images=len(images_list)

    while num_images>=0:
        '''
        该循环条件时num_images>=0，特殊点在于num_images==0，当文件读取时，其会对某个文件重复处理（结果覆盖），但不会影响最终结果。
        当摄像头读取时，可以发现其一直满足循环条件，故会不断读取摄像头图像进行检测，跳出条件cv2.waitKey(1) & 0xFF == ord('q')。
        '''
        if webcam_num==-1:
            num_images-=1 # 图像索引

        # 数据读取
        if webcam_num>=0:
            if not cap.isOpened():
                raise RuntimeError("Webcam could not open. Please check connection.")
            ret,frame=cap.read()
            image=np.array(ret)
        else:
            image_path=os.path.join(args.image_dir,images_list[num_images])
            image=np.array(imread(image_path))

        # 数据处理
        if len(image.shape)==2:
            image=image[:,:,np.newaxis]
            image=np.concatenate((image,image,image),axis=2)
        image=image[:,:,::-1] # rgb->bgr

        blob,im_scale=get_image_blob(image)
        assert len(im_scale) == 1, "Only single-image batch implemented"
        im_info=torch.from_numpy(np.array([[blob.shape[1],blob.shape[2],im_scale[0]]],dtype=np.float32))
        blob=torch.from_numpy(blob).permute(0,3,1,2).contiguous()

        # 模型测试
        with torch.no_grad():
            im_data.resize_(blob.size()).copy_(blob)
            im_info.resize_(im_info.size()).copy_(im_info)
            gt_boxes.resize_(1,1,5).zero_()
            num_boxes.resize_(1).zero_()

        rois, rois_label, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

        # 模型预测最为重要的是：框rois，对应的类别置信度cls_prob，对应的回归偏移bbox_pred
        scores=cls_prob.data # [1,num_rois,21]
        boxes=rois.data[:,:,1:] # [1,num_rois,4]
        if cfg.TEST.BBOX_REG:
            #应用回归偏移修正框
            box_deltas=bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # 若对回归偏移值进行过标准化，则这里需要反标准化delta后再修正框
                if args.class_agnostic:
                    if args.cuda:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    if args.cuda:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                    box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

            pred_boxes=bbox_transform_inv(boxes,box_deltas,1) # [1,num_rois,4(或21*4)]
            pred_boxes=clip_boxes(pred_boxes,im_info.data,1)
        else:
            # 不做回归偏移，则将维度4->21*4，每一类都有坐标，感觉若是使用这种，args.class_agnostic必设置为False
            pred_boxes=np.tile(boxes, (1, scores.shape[1])) # [1,num_rois,21*4]

        # 缩放至原始图像坐标
        pred_boxes/=im_scale[0]

        scores=scores.squeeze()
        pred_boxes=pred_boxes.squeeze()
        # 找出除bg的所有类的框
        image_show=np.copy(image)
        for i in range(1,len(pascal_classes)):
            '''
            该过程中有三个阈值：
            （1）det_thresh：初步筛选阈值，该值一般较小，能够得到大量框
            （2）cfg.TEST.NMS: NMS阈值，用于删除重复框
            （3）thresh：vis_detections参数，该值较det_thresh大（越大则筛选的框越准确），仅可视化时使用
            '''
            # 初步筛选
            inds=torch.nonzero(scores[:,i]>det_thresh).view(-1)
            if inds.numel()>0:
                cls_scores=scores[:,i][inds]
                # 从大到小排序
                _,order=torch.sort(cls_scores,dim=0,descending=True)
                if args.class_agnostic:
                    cls_boxes=pred_boxes[inds,:]
                else:
                    # 得到对应类别的框坐标
                    cls_boxes=pred_boxes[inds][:,i*4:(i+1)*4]

                # [x1, y1, x2, y2, cls_prob]
                cls_dets=torch.cat([cls_boxes,cls_scores.unsqueeze(1)],dim=1)
                cls_dets=cls_dets[order]
                # 合并重复框
                keep_inds=nms(cls_boxes[order,:],cls_scores[order],cfg.TEST.NMS)
                cls_dets=cls_dets[keep_inds.view(-1).long()]

                if vis:
                    # max_vis_boxes设置单类别框可视化最多数目，thresh设置可视化置信度阈值
                    image_show=vis_detections(image_show,pascal_classes[i],cls_dets.cpu().numpy(),max_vis_boxes=10, thresh=0.5)

        if vis and webcam_num==-1:
            result_path=os.path.join(args.image_dir,images_list[num_images][:-4]+'_{}'.format(args.net)+'_det.jpg')
            cv2.imwrite(result_path,image_show)
        else:
            im2showRGB = cv2.cvtColor(image_show, cv2.COLOR_BGR2RGB)
            cv2.imshow("frame", im2showRGB)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if webcam_num >= 0:
        cap.release()
        cv2.destroyAllWindows()




