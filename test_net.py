import _init_path
import os
import numpy as np
import argparse
import cv2
import torch
import pickle
from model.roi_layers import nms
from model.utils.config import cfg,cfg_from_file,cfg_from_list,get_output_dir
from dataset.pascal import get_imdb_and_roidbs
from model.dataloader.batchloader import roibatchloader
from model.faster_rcnn.vgg16 import vgg16
from model.utils.net_utils import vis_detections
from model.rpn.bbox_transform import bbox_transform_inv,clip_boxes

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
                        help='directory to load models', default="models",
                        type=str)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
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
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # cfg载入
    args = parse_args()

    # 提醒开启cuda
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # 超参数设置
    np.random.seed(cfg.RNG_SEED)
    cfg.USE_GPU_NMS = args.cuda
    det_thresh = 0.05 if args.vis else 0  # 设置rois类别置信度阈值（初步筛选阈值）
    vis = args.vis  # 是否可视化检测结果
    max_per_image = 100 # 每张图片最多可拥有的检测结果

    # 数据集设置
    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    # 数据集制作
    cfg.TRAIN.USE_FLIPPED = False
    imdb, roidbs, ratio_list, ratio_index = get_imdb_and_roidbs(args.imdbval_name,False)
    imdb.competition_mode(on=True)
    print('{:d} roidb entries'.format(len(roidbs)))

    # 数据集读取
    # 注意batchsize=1，即测试图像是一张一张的测试；并且shuffle=False，不打乱顺序
    dataset = roibatchloader(roidbs, ratio_list, ratio_index, 1, \
                             imdb.num_classes, training=False, normalize=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=False, num_workers=0,
                                             pin_memory=True)

    # 初始化模型
    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
    fasterRCNN.create_architecture()

    # 载入模型参数
    input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_model_path = os.path.join(input_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch,
                                                                                args.checkpoint))
    if args.cuda:
        checkpoints = torch.load(load_model_path)
    else:
        checkpoints = torch.load(load_model_path, map_location=(lambda storage, loc: storage))
    fasterRCNN.load_state_dict(checkpoints['model'])

    if 'pooling_mode' in checkpoints.keys():
        cfg.POOLING_MODE = checkpoints['pooling_mode']

    # skip to cuda
    if args.cuda:
        cfg.CUDA = True

    if args.cuda:
        fasterRCNN.cuda()
    fasterRCNN.eval()  # 设置测试模式，必须有

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

    # 模型测试
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]  # [num_classes, num_images] list类型，用于存储检测结果
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))  # [0,5] 当作空pred_box坐标

    data_iter = iter(dataloader)
    # 分别读取每张图像
    for i in range(num_images):
        data = next(data_iter)
        with torch.no_grad():
            im_data.resize_(data[0].size()).copy_(data[0])
            im_info.resize_(data[1].size()).copy_(data[1])
            gt_boxes.resize_(data[2].size()).copy_(data[2])
            num_boxes.resize_(data[3].size()).copy_(data[3])

        rois, rois_label, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
            = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

        # 模型预测最为重要的是：框rois，对应的类别置信度cls_prob，对应的回归偏移bbox_pred
        scores = cls_prob.data  # [1,num_rois,21]
        boxes = rois.data[:, :, 1:]  # [1,num_rois,4]
        if cfg.TEST.BBOX_REG:
            # 应用回归偏移修正框
            box_deltas = bbox_pred.data
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
                    box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)  # [1,num_rois,4(或21*4)]
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # 不做回归偏移，则将维度4->21*4，每一类都有坐标，感觉若是使用这种，args.class_agnostic必设置为False
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))  # [1,num_rois,21*4]

        # 缩放至原始图像坐标
        pred_boxes /= data[1][0][2].item()

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        # 找出除bg的所有类的框
        image=cv2.imread(imdb.image_path_from_index(i))
        image_show = np.copy(image)
        for j in range(1, len(imdb.classes)):
            '''
            该过程中有三个阈值：
            （1）det_thresh：初步筛选阈值，该值一般较小，能够得到大量框
            （2）cfg.TEST.NMS: NMS阈值，用于删除重复框
            （3）thresh：vis_detections参数，该值较det_thresh大（越大则筛选的框越准确），仅可视化时使用
            '''
            # 初步筛选
            inds = torch.nonzero(scores[:, j] > det_thresh).view(-1)
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                # 从大到小排序
                _, order = torch.sort(cls_scores, dim=0, descending=True)
                if args.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    # 得到对应类别的框坐标
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                # [x1, y1, x2, y2, cls_prob]
                cls_dets = torch.cat([cls_boxes, cls_scores.unsqueeze(1)], dim=1)
                cls_dets = cls_dets[order]
                # 合并重复框
                keep_inds = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                cls_dets = cls_dets[keep_inds.view(-1).long()]

                if vis:
                    # max_vis_boxes设置单类别框可视化最多数目，thresh设置可视化置信度阈值
                    image_show = vis_detections(image_show, imdb.classes[j], cls_dets.cpu().numpy(), max_vis_boxes=10,
                                                thresh=0.3)
                all_boxes[j][i]=cls_dets.cpu().numpy()
            else:
                all_boxes[j][i]=empty_array

        # 每张图像有最大可检测数目
        if max_per_image>0:
            image_scores=np.hstack([all_boxes[j][i][:,-1] for j in range(1, len(imdb.classes))])
            if len(image_scores)>max_per_image:
                keep_thresh=np.sort(image_scores)[-max_per_image]
                for j in range(1, len(imdb.classes)):
                    keep=np.nonzero(all_boxes[j][i][:,-1]>=keep_thresh)[0]
                    all_boxes[j][i]=all_boxes[j][i][keep,:]
        if vis:
            cv2.imwrite('result.jpg', image_show)

    # 检测结果存储
    save_name = 'faster_rcnn_10'
    output_dir = get_output_dir(imdb, save_name)
    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    # 评估检测结果
    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir)



