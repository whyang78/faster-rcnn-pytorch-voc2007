import torch
import torch.nn as nn
import numpy as np
from model.utils.config import cfg
from .generate_anchors import generate_anchors
from .bbox_transform import bbox_transform_inv, clip_boxes
from model.roi_layers import nms


class ProposalLayer(nn.Module):
    def __init__(self,feat_stride, scales, ratios):
        super(ProposalLayer, self).__init__()

        self._feat_stride=feat_stride
        self._anchors=torch.from_numpy(generate_anchors(ratios=np.array(ratios),scales=np.array(scales))).float()
        self._num_anchors=self._anchors.shape[0]

    def forward(self,rpn_cls_prob,rpn_bbox_pred,im_info,cfg_key):
        '''
        :param rpn_cls_prob: [b, num_anchors*2 , feat_height, feat_wigth]
        :param rpn_bbox_pred: [b, num_anchors*4 , feat_height, feat_wigth]
        :param im_info: [b, 3]
        :param cfg_key: str
        :return: output: [b, post_nms_topN, 5] 其中5指代[batch_index,x1,y1,x2,y2]，该种形式可以用于roi_align或者roi_pooling
        '''

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs
        score=rpn_cls_prob[:,self._num_anchors:,:,:] # 取出前景置信度分数 [b, num_anchors, feat_height, feat_wigth]

        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH
        min_size      = cfg[cfg_key].RPN_MIN_SIZE

        batch_size,feat_height,feat_width=score.shape[0],score.shape[2],score.shape[3]
        shift_x=np.arange(0,feat_width) * self._feat_stride
        shift_y=np.arange(0,feat_height) * self._feat_stride
        shift_x,shift_y=np.meshgrid(shift_x,shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                             shift_x.ravel(), shift_y.ravel())).transpose())
        shifts=shifts.contiguous().type_as(score).float() # 每张特征图中所有特征点对应原图坐标 [feat_height*feat_wigth, 4]
        # 初始化中生成的anchors是基于原点[0,0,0,0]的

        num_anchors=self._num_anchors
        num_featpoints=shifts.shape[0] # feat_height*feat_width

        # 1.生成整个batch中每个样本特征图上的全部anchors
        self._anchors=self._anchors.type_as(score)
        anchors=self._anchors.view(1,num_anchors,4)+shifts.view(num_featpoints,1,4) # [num_featpoints, num_anchors, 4]
        anchors=anchors.view(1,num_featpoints*num_anchors,4).expand(batch_size,num_featpoints*num_anchors,4)
           #[b, num_featpoints*num_anchors, 4]

        # 2.回归偏移调整anchors得到初始proposals
        rpn_bbox_pred=rpn_bbox_pred.permute(0,2,3,1).contiguous().view(batch_size,-1,4)
        proposals=bbox_transform_inv(anchors,rpn_bbox_pred,batch_size) # [b, num_featpoints*num_anchors, 4]

        # 3.修剪超过图像尺寸的proposals
        proposals=clip_boxes(proposals,im_info,batch_size)

        # 4.按照分数排序
        score=score.permute(0,2,3,1).contiguous().view(batch_size,-1) # [b, num_featpoints*num_anchors]
        _,order=torch.sort(score, 1, True)

        output=score.new(batch_size,post_nms_topN,5).zero_()
        for i in range(batch_size):
            score_single=score[i]
            order_single=order[i]
            proposals_single=proposals[i]

            # 5.筛选得分前pre_nms_topN个
            if pre_nms_topN>0 and pre_nms_topN<score.numel():
                order_single=order_single[:pre_nms_topN]

            proposals_single=proposals_single[order_single,:]
            score_single=score_single[order_single].view(-1,1)

            # 6.NMS
            # 传入的proposals和scores都是已排序的（从大到小）
            keep_inds=nms(proposals_single,score_single.squeeze(1),nms_thresh)
            keep_inds=keep_inds.long().view(-1)

            # 7.筛选得分前post_nms_topN个
            if post_nms_topN>0:
                keep_inds=keep_inds[:post_nms_topN]

            proposals_single=proposals_single[keep_inds,:]
            score_single=score_single[keep_inds,:]

            # 不足补零
            num_proposals=proposals_single.shape[0]
            output[i,:,0]=i
            output[i,:num_proposals,1:]=proposals_single
        return output

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


    