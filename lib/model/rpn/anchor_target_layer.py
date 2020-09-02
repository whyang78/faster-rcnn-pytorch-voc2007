import torch
import torch.nn as nn
import numpy as np
from model.utils.config import cfg
from .generate_anchors import generate_anchors
from .bbox_transform import bbox_overlaps_batch, bbox_transform_batch

class AnchorTargetLayer(nn.Module):
    def __init__(self,feat_stride,scales,ratios):
        super(AnchorTargetLayer, self).__init__()
        self._feat_stride=feat_stride
        self._ratios=ratios
        self._scales=scales
        self._anchors=torch.from_numpy(generate_anchors(ratios=np.array(ratios),scales=np.array(scales))).float()
        self._num_anchors=self._anchors.shape[0]

        # 允许anchor超过边界的一定距离
        self._allow_boarder=0

    def forward(self, rpn_cls_score, gt_boxes, im_info, num_boxes):
        # 1.生成全部anchors
        batch_size,feat_height,feat_width=rpn_cls_score.shape[0],rpn_cls_score.shape[2],rpn_cls_score.shape[3]

        shift_x=np.arange(0,feat_width)*self._feat_stride
        shift_y=np.arange(0,feat_height)*self._feat_stride
        shift_x,shift_y=np.meshgrid(shift_x,shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                             shift_x.ravel(), shift_y.ravel())).transpose())
        shifts=shifts.contiguous().type_as(rpn_cls_score).float()

        num_anchors=self._num_anchors
        num_featpoints=shifts.shape[0]

        all_anchors=self._anchors.type_as(rpn_cls_score)
        all_anchors=all_anchors.view(1,num_anchors,4)+shifts.view(num_featpoints,1,4)
        all_anchors=all_anchors.view(num_featpoints*num_anchors,4)

        # 2. 过滤超界anchors
        keep=((all_anchors[:,0]>=-self._allow_boarder)&
              (all_anchors[:,1]>=-self._allow_boarder)&
              (all_anchors[:,2]<int(im_info[0][1])+self._allow_boarder)&
              (all_anchors[:,3]<int(im_info[0][0])+self._allow_boarder))
        inside_inds=torch.nonzero(keep).view(-1) # 维度[num_anchors]
        anchors=all_anchors[inside_inds,:]

        # 3. anchors与 gt_boxes匹配打标
        labels=gt_boxes.new(batch_size,inside_inds.shape[0]).fill_(-1)
        bbox_inside_weights=gt_boxes.new(batch_size,inside_inds.shape[0]).zero_()
        bbox_outside_weights=gt_boxes.new(batch_size,inside_inds.shape[0]).zero_()

        overlaps=bbox_overlaps_batch(anchors,gt_boxes) # 维度[b,num_anchors,num_gt_boxes]
        max_overlaps,argmax_overlaps=torch.max(overlaps,2) # 维度[b,num_anchors]
        gt_max_overlaps,_=torch.max(overlaps,1) # 维度[b,num_gt_boxes]

        # 若cfg.TRAIN.RPN_CLOBBER_POSITIVES=True,则当anchors同时符合正样本和负样本条件时，判为负样本
        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            labels[max_overlaps<cfg.TRAIN.RPN_NEGATIVE_OVERLAP]=0

        # gt_max_overlaps=0表示该图中所有anchors与该gt_box无交集，此gt_box可能是全零值或者无效值（生成数据集gt_boxes时不足补零）
        # 若不修正零值，则所有anchors都是与该gt_box具有最大IOU，其都可判定为正样本，与实际不符。修正零值其实就是抛弃该gt_box。
        gt_max_overlaps[gt_max_overlaps==0]=1e-5
        keep=torch.sum(torch.eq(overlaps,gt_max_overlaps.view(batch_size,1,-1).expand_as(overlaps)),2)
        if torch.sum(keep)>0:
            labels[keep>0]=1

        labels[max_overlaps>= cfg.TRAIN.RPN_POSITIVE_OVERLAP]=1

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # 4.anchors筛选
        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)

        sum_fg = torch.sum((labels == 1).int(), 1)
        sum_bg = torch.sum((labels == 0).int(), 1)

        for i in range(batch_size):
            # 超出指定数目则进行随机筛选
            if sum_fg[i]>num_fg:
                fg_inds=torch.nonzero(labels[i]==1).view(-1)
                randnum=torch.from_numpy(np.random.permutation(fg_inds.shape[0])).type_as(gt_boxes).long()
                disable_inds=fg_inds[randnum[:(fg_inds.shape[0]-num_fg)]]
                labels[i][disable_inds]=-1

            num_bg=cfg.TRAIN.RPN_BATCHSIZE-torch.sum((labels == 1).int(), 1)[i]
            if sum_bg[i]>num_bg:
                bg_inds=torch.nonzero(labels[i]==0).view(-1)
                randnum=torch.from_numpy(np.random.permutation(bg_inds.shape[0])).type_as(gt_boxes).long()
                disable_inds=bg_inds[randnum[:(bg_inds.shape[0]-num_bg)]]
                labels[i][disable_inds]=-1

        # 5.求解回归偏移真值
        # 注意：每个anchor取与自己有最大overlap的gt_box作为目标框，即argmax_overlap
        offset=torch.arange(0,batch_size)*gt_boxes.shape[1]
        argmax_overlaps=argmax_overlaps+offset.view(batch_size,1).type_as(argmax_overlaps) #维度[b,num_anchors]
        bbox_targets=self._compute_targets_batch(anchors,gt_boxes.view(-1,5)[argmax_overlaps.view(-1),:].view(batch_size,-1,5))
        # 传入gt_boxes.view(-1,5)[argmax_overlaps.view(-1),:].view(batch_size,-1,5)，即对应每个anchor的gt_box，维度[b,num_anchors,5]
        # 返回bbox_targets维度[b,num_anchors,4]

        # 6.权重矩阵
        # 都是针对回归损失的权重矩阵
        bbox_inside_weights[labels == 1] = cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[0]

        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            num_examples = torch.sum(labels[i] >= 0)
            positive_weights = 1.0 / num_examples.item()
            negative_weights = 1.0 / num_examples.item()
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))

        bbox_outside_weights[labels == 1] = positive_weights
        bbox_outside_weights[labels == 0] = negative_weights

        # 7.all_anchors类别真值（fg/bg）以及回归偏移真值整理，权重矩阵等
        total_anchors=int(num_anchors*num_featpoints)
        # 因为rpn网络的输出结果是针对于total_anchors，即每个特征点都会有9个anchors
        # 所以要将上面满足inside_inds的anchors对应得到的一些结果进行整理，嵌入到total_anchors对应的结果中
        labels=self._umap(labels,total_anchors,inside_inds,batch_size,fill=-1) # [b,total_anchors]
        bbox_targets=self._umap(bbox_targets,total_anchors,inside_inds,batch_size,fill=0) # [b,total_anchors,4]
        bbox_inside_weights=self._umap(bbox_inside_weights,total_anchors,inside_inds,batch_size,fill=0) # [b,total_anchors]
        bbox_outside_weights=self._umap(bbox_outside_weights,total_anchors,inside_inds,batch_size,fill=0) # [b,total_anchors]

        output=[]
        labels=labels.view(batch_size,feat_height,feat_width,num_anchors).permute(0,3,1,2).contiguous()
        labels=labels.view(batch_size,1,num_anchors*feat_height,feat_width)
        output.append(labels)  # [b, 1, num_anchors*feat_height, feat_width]

        # 维度调整主要为了适应网络预测结果维度
        bbox_targets=bbox_targets.view(batch_size,feat_height,feat_width,num_anchors*4).permute(0,3,1,2).contiguous()
        output.append(bbox_targets) # [b, num_anchors*4, feat_height, feat_width]

        bbox_inside_weights=bbox_inside_weights.view(batch_size,total_anchors,1).expand(batch_size,total_anchors,4).contiguous()
        bbox_inside_weights=bbox_inside_weights.view(batch_size,feat_height,feat_width,num_anchors*4).permute(0,3,1,2).contiguous()
        output.append(bbox_inside_weights) # [b, num_anchors*4, feat_height, feat_width]

        bbox_outside_weights=bbox_outside_weights.view(batch_size,total_anchors,1).expand(batch_size,total_anchors,4).contiguous()
        bbox_outside_weights=bbox_outside_weights.view(batch_size,feat_height,feat_width,num_anchors*4).permute(0,3,1,2).contiguous()
        output.append(bbox_outside_weights) # [b, num_anchors*4, feat_height, feat_width]

        return output

    def _compute_targets_batch(self,anchors,gt_boxes):
        return bbox_transform_batch(anchors,gt_boxes[:,:,:4])

    def _umap(self,data,count,inds,batchsize,fill=0):
        if data.dim()==2:
            ret=torch.Tensor(batchsize,count).fill_(fill).type_as(data)
            ret[:,inds]=data
        else:
            ret=torch.Tensor(batchsize,count,data.shape[2]).fill_(fill).type_as(data)
            ret[:,inds,:]=data
        return ret

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


