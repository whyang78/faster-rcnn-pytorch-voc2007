import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils.config import cfg
from .proposal_layer import ProposalLayer
from .anchor_target_layer import AnchorTargetLayer
from model.utils.net_utils import smooth_l1_loss

class RPN(nn.Module):
    def __init__(self,feature_out_dim):
        super(RPN, self).__init__()
        self.out_dim=feature_out_dim
        self.anchor_scales = cfg.ANCHOR_SCALES
        self.anchor_ratios = cfg.ANCHOR_RATIOS
        self.feat_stride = cfg.FEAT_STRIDE[0]

        self.RPN_Conv=nn.Conv2d(self.out_dim,512,3,1,1,bias=True)

        # fg/bg
        self.score_outdim=len(self.anchor_scales)*len(self.anchor_ratios)*2
        self.RPN_cls_score=nn.Conv2d(512,self.score_outdim,1,1,0)
        # dx,dy,dw,dh
        self.pred_outdim=len(self.anchor_scales)*len(self.anchor_ratios)*4
        self.RPN_bbox_pred=nn.Conv2d(512,self.pred_outdim,1,1,0)

        self.RPN_proposal=ProposalLayer(self.feat_stride,self.anchor_scales,self.anchor_ratios)
        self.RPN_anchor_target=AnchorTargetLayer(self.feat_stride,self.anchor_scales,self.anchor_ratios)

        self.rpn_loss_cls=0
        self.rpn_loss_box=0

    @staticmethod
    def reshape(x,d):
        input_shape=x.shape
        x=x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1])*float(input_shape[2])/float(d)),
            input_shape[3]
        )
        return x

    def forward(self, base_feat, im_info, gt_boxes, num_boxes):
        '''
        :param base_feat: 特征骨架提取的特征
        :param im_info: 参考数据集
        :param gt_boxes:  参照数据集
        :param num_boxes:  参照数据集
        :return:

        步骤：
        1.搭建rpn网络，得到base_feat经过网络的输出，包括fg/bg分数rpn_cls_score及置信度rpn_cls_prob，还有回归预测偏移rpn_bbox_pred。
            他们对应特征图中每个点的所有anchors。
            rpn_cls_score、rpn_cls_prob维度：[b, self.score_outdim, feat_height, feat_width]
            rpn_bbox_pred维度：[b, self.pred_outdim, feat_height, feat_width]
        2.根据所有anchors以及回归偏移获取proposals并进行筛选以及nms处理等，此处需要区分测试和训练的不同情况。在获取proposals时，
            测试与训练的某些超参数是不一样的，默认参数如下：
                训练：pre_nms_topN：12000     post_nms_topN：2000        nms_thresh：0.7
                测试：pre_nms_topN：6000     post_nms_topN：300        nms_thresh：0.7
            输出proposals维度：[b, post_nms_topN, 5] 其中5指代[batch_index,x1,y1,x2,y2]，该种形式可以用于roi_align或者roi_pooling
        3.测试得到的proposals在后续处理中直接担任rois，而训练得到的proposals还要进一步筛选得到rois，详见faster_rcnn.py。
        4.faster-rcnn网络训练时还要训练rpn网络，故还要得到rpn_loss_box，rpn_loss_cls。
            (1)计算rpn真值。包括所有anchors对应的标签labels，回归偏移真值bbox_target，权重矩阵bbox_inside_weights、bbox_outside_weights。
                    labels 维度：[b, 1, num_anchors*feat_height, feat_width]
                    bbox_targets 维度：[b, num_anchors*4, feat_height, feat_width]
                    bbox_inside_weights 维度：[b, num_anchors*4, feat_height, feat_width]
                    bbox_outside_weights 维度：[b, num_anchors*4, feat_height, feat_width]
            (2)计算分类损失rpn_loss_cls。
                注意labels包含三种值[-1,0,1]，其中0代表负样本，1代表正样本，仅计算正负样本的损失，忽略-1标签样本。
            (3)计算回归损失rpn_loss_box。
                bbox_inside_weights仅对正样本计算回归损失。
                bbox_outside_weights平衡分类损失与回归损失。
        '''
        rpn_conv=F.relu(self.RPN_Conv(base_feat),inplace=True)

        # fg/bg分数与置信度
        rpn_cls_score=self.RPN_cls_score(rpn_conv)
        rpn_cls_score_reshape=self.reshape(rpn_cls_score,2)
        rpn_cls_prob_reshape=F.softmax(rpn_cls_score_reshape,dim=1)
        rpn_cls_prob=self.reshape(rpn_cls_prob_reshape,self.score_outdim) # 维度 [b, self.score_outdim, base_feat_height, base_feat_width]

        # 回归偏移
        rpn_bbox_pred=self.RPN_bbox_pred(rpn_conv) # 维度 [b, self.pred_outdim, base_feat_height, base_feat_width]

        # 获取proposals
        # 测试与训练的超参数设置不一样 （例如：RPN_PRE_NMS_TOP_N默认训练12000，测试6000）
        # 测试生成的proposals直接担任rois，训练生成的proposals还需进一步处理
        cfg_key = 'TRAIN' if self.training else 'TEST'
        rois=self.RPN_proposal(rpn_cls_prob.data,rpn_bbox_pred.data,im_info,cfg_key) #[b, post_nms_topN, 5]

        self.rpn_loss_box=0
        self.rpn_loss_cls=0
        # 若是测试则直接返回上述结果，若是训练则继续执行rpn损失的计算
        if self.training:
            assert gt_boxes is not None

            # 1.计算rpn真值
            rpn_data=self.RPN_anchor_target(rpn_cls_score.data,gt_boxes,im_info,num_boxes)
            # labels [b, 1, num_anchors*feat_height, feat_width]
            # bbox_targets [b, num_anchors*4, feat_height, feat_width]
            # bbox_inside_weights [b, num_anchors*4, feat_height, feat_width]
            # bbox_outside_weights [b, num_anchors*4, feat_height, feat_width]

            # 2.计算rpn_loss_cls
            batchsize=base_feat.shape[0]
            cls_score=rpn_cls_score_reshape.permute(0,2,3,1).contiguous().view(batchsize,-1,2)
            labels=rpn_data[0].view(batchsize,-1)

            # 计算正负样本的损失，-1表示忽略该样本
            keep=labels.view(-1).ne(-1).nonzero().view(-1)
            keep_cls_score=torch.index_select(cls_score.view(-1,2),dim=0,index=keep)
            keep_labels=torch.index_select(labels.view(-1),dim=0,index=keep).long()
            self.rpn_loss_cls=F.cross_entropy(keep_cls_score,keep_labels)

            # 3.计算rpn_loss_box
            bbox_targets,bbox_inside_weights,bbox_outside_weights=rpn_data[1:]
            self.rpn_loss_box=smooth_l1_loss(rpn_bbox_pred,bbox_targets,bbox_inside_weights,bbox_outside_weights,
                                             sigma=3, dim=[1,2,3])

        return rois,self.rpn_loss_cls,self.rpn_loss_box




