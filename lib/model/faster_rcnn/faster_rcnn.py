import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils.config import cfg
from model.rpn.rpn import RPN
from model.roi_layers import ROIAlign, ROIPool
from model.rpn.proposal_target_layer import ProposalTargetLayer
from model.utils.net_utils import smooth_l1_loss

class Faster_RCNN(nn.Module):
    def __init__(self,classes,class_agnostic=False):
        super(Faster_RCNN, self).__init__()
        self.n_classes=len(classes)
        self.classes=classes
        self.class_agnostic = class_agnostic
        # RCNN loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        self.RCNN_rpn = RPN(self.feature_out_dim)
        self.RCNN_proposal_target = ProposalTargetLayer(self.n_classes)

        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)

    def forward(self, im_data, im_info, gt_boxes,num_boxes):
        batch_size=im_data.shape[0]

        gt_boxes=gt_boxes.data
        im_info=im_info.data
        num_boxes=num_boxes.data

        # 1.获取骨干特征
        base_feat=self.RCNN_base(im_data)
        # 2.由rpn网络提取proposals，并得到rpn loss
        rois, rpn_loss_cls, rpn_loss_box=self.RCNN_rpn(base_feat,im_info,gt_boxes,num_boxes)
        # 测试时proposals担任rois，训练时rois还要由proposals进一步筛选得到
        # 训练时未筛选proposals由原始proposals(即由RPN网络得到的rois)和gt_boxes组成
        if self.training:
            # proposals筛选得到rois，并得到rois的真值
            rois,rois_labels,rois_targets_reg,rois_inside_weights,rois_outside_weights=self.RCNN_proposal_target(rois,gt_boxes,num_boxes)

            rois_labels=rois_labels.view(-1).long() # [ b*rois_per_image]
            rois_targets_reg=rois_targets_reg.view(-1,rois_targets_reg.shape[2]) # [ b*rois_per_image, 4]
            rois_inside_weights=rois_inside_weights.view(-1,rois_inside_weights.shape[2]) # [b*rois_per_image, 4]
            rois_outside_weights=rois_outside_weights.view(-1,rois_outside_weights.shape[2]) # [ b*rois_per_image, 4]
        else:
            rois_labels=None
            rois_targets_reg=None
            rois_inside_weights=None
            rois_outside_weights=None
            rpn_loss_cls=0
            rpn_loss_box=0

        # 3.在base_feat中提取rois区域的池化特征
        # 将rois展成二维矩阵传入，因为其数据中包含batch_index，故可以先根据batch_index选择对应样本的base_feat，然后再根据坐标选取区域特征。
        if cfg.POOLING_MODE == 'align':
            # vgg16特征骨架输出特征通道数为 512
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5)) # [b*rois_per_image, 512, cfg.POOLING_SIZE, cfg.POOLING_SIZE]
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5)) # [b*rois_per_image, 512, cfg.POOLING_SIZE, cfg.POOLING_SIZE]

        # 4.池化特征延展并得到类别分数与回归预测偏移
        pooled_feat=self.head_to_tail(pooled_feat)
        rcnn_cls_score=self.RCNN_cls_score(pooled_feat)
        rcnn_cls_prob=F.softmax(rcnn_cls_score,dim=1) # [b*rois_per_image, 21]

        rcnn_bbox_pred=self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # 若self.class_agnostic=False，则每个roi拥有对应21种类别的坐标，每一组坐标对应一类（21*4=84）。
            # 根据roi的label真值，选择出对应类别下的那组坐标，例如label=2，则[21,4]中选择选择行序号为2的那组坐标。
            # [b*rois_per_image, 84] --view-> [b*rois_per_image, 21 ,4] --gather-> [b*rois_per_image, 1 ,4]
            rcnn_bbox_pred_view=rcnn_bbox_pred.view(rcnn_bbox_pred.shape[0],int(rcnn_bbox_pred.shape[1]/4),4)
            rcnn_bbox_pred_select=torch.gather(rcnn_bbox_pred_view,dim=1,
                                             index=rois_labels.view(rois_labels.shape[0],1,1).expand(rois_labels.shape[0],1,4))
            rcnn_bbox_pred=rcnn_bbox_pred_select.squeeze(1) # [b*rois_per_image, 4]

        # 5.计算RCNN loss
        # 若是训练则要计算训练损失
        if self.training:
            self.RCNN_loss_cls=F.cross_entropy(rcnn_cls_score,rois_labels)
            self.RCNN_loss_bbox=smooth_l1_loss(rcnn_bbox_pred,rois_targets_reg,rois_inside_weights,rois_outside_weights)

        # 6.整理预测结果
        # 对应每个batch中各个样本所有rois的类别置信度以及回归偏移
        rcnn_cls_prob=rcnn_cls_prob.view(batch_size,rois.shape[1],-1) # [b, rois_per_image, -1]
        rcnn_bbox_pred=rcnn_bbox_pred.view(batch_size,rois.shape[1],-1) # [b, rois_per_image, -1]

        return rois,rois_labels,rcnn_cls_prob,rcnn_bbox_pred,rpn_loss_cls,rpn_loss_box,self.RCNN_loss_cls,self.RCNN_loss_bbox

    # 权重初始化
    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()