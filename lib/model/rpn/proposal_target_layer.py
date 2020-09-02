import torch
import torch.nn as nn
import numpy as np
from model.utils.config import cfg
from .bbox_transform import bbox_overlaps_batch, bbox_transform_batch

class ProposalTargetLayer(nn.Module):
    def __init__(self,nclasses):
        super(ProposalTargetLayer, self).__init__()
        self._num_classes = nclasses
        # 用于对回归预测偏移真值进行标准化
        self.BBOX_NORMALIZE_MEANS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
        self.BBOX_NORMALIZE_STDS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)
        self.BBOX_INSIDE_WEIGHTS = torch.FloatTensor(cfg.TRAIN.BBOX_INSIDE_WEIGHTS)

    def forward(self, all_rois, gt_boxes, num_boxes):
        self.BBOX_NORMALIZE_MEANS = self.BBOX_NORMALIZE_MEANS.type_as(gt_boxes)
        self.BBOX_NORMALIZE_STDS = self.BBOX_NORMALIZE_STDS.type_as(gt_boxes)
        self.BBOX_INSIDE_WEIGHTS = self.BBOX_INSIDE_WEIGHTS.type_as(gt_boxes)

        gt_boxes_append=gt_boxes.new(gt_boxes.size()).zero_()
        gt_boxes_append[:,:,1:5]=gt_boxes[:,:,:4]

        # 将gt_boxes加入到候选proposals中
        all_rois=torch.cat([all_rois,gt_boxes_append],dim=1)

        # 设置每个样本的rois数（由候选proposals筛选得到）
        num_images = 1
        rois_per_image = int(cfg.TRAIN.BATCH_SIZE / num_images)
        fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))
        fg_rois_per_image = 1 if fg_rois_per_image == 0 else fg_rois_per_image

        # 得到该batch的rois，及其对应的标签labels_batch，回归偏移真值rois_targets_reg，还有两个权重矩阵
        rois_batch, labels_batch, rois_targets_reg, rois_inside_weights=self.get_rois_data(all_rois,gt_boxes,
                                                             fg_rois_per_image,rois_per_image,self._num_classes)
        rois_outside_weights=(rois_inside_weights>0).float() # [b, rois_per_image, 4]
        # 这个outside_weights好像就是1，表示分类损失权重与回归损失权重相同

        return rois_batch,labels_batch,rois_targets_reg,rois_inside_weights,rois_outside_weights

    def _compute_rois_targets(self,rois_batch,gt_rois_batch):
        assert rois_batch.shape[1]==gt_rois_batch.shape[1]
        assert rois_batch.shape[2]==4
        assert gt_rois_batch.shape[2]==4

        rois_targets=bbox_transform_batch(rois_batch,gt_rois_batch)
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # normalize targets by a precomputed mean and stdev
            rois_targets = ((rois_targets - self.BBOX_NORMALIZE_MEANS.expand_as(rois_targets))
                        / self.BBOX_NORMALIZE_STDS.expand_as(rois_targets))
        return rois_targets

    def _compute_rois_targets_regression(self,rois_targets,labels_batch):
        batch_size,num_rois=labels_batch.shape[0],labels_batch.shape[1]
        rois_targets_reg=rois_targets.new(batch_size,num_rois,4).zero_()
        rois_inside_weights=rois_targets.new(rois_targets_reg.size()).zero_()

        for i in range(batch_size):
            if labels_batch[i].sum()==0:
                continue
            keep_inds=torch.nonzero(labels_batch[i]>0).view(-1)
            for j in range(keep_inds.numel()):
                ind=keep_inds[j]
                rois_targets_reg[i,ind,:]=rois_targets[i,ind,:]
                rois_inside_weights[i,ind,:]=self.BBOX_INSIDE_WEIGHTS
        return rois_targets_reg,rois_inside_weights

    def get_rois_data(self,all_rois,gt_boxes,fg_rois_per_image,rois_per_image,num_classes):
        '''
        :param all_rois: [b, post_nms_num + num_gt_boxes, 5]
        :param gt_boxes: [b, num_gt_boxes, 5]
        :param fg_rois_per_image: int 设置的每个样本的fg_rois数目
        :param rois_per_image: int 设置的每个样本的rois数目
        :param num_classes: int
        :return:
            rois_batch维度：[b, rois_per_image, 5] 其中5指代[batch_index, x1, y1, x2, y2]
            labels_batch维度：[b, rois_per_image]
            rois_targets_reg维度：[b, rois_per_image, 4]
            rois_inside_weights维度：[b, rois_per_image, 4]
        '''
        # 1.proposals与gt_boxes匹配打标
        overlaps=bbox_overlaps_batch(all_rois,gt_boxes) # [b, num_proposals, num_gt_boxes]
        max_overlaps,argmax_overlaps=torch.max(overlaps,dim=2) # [b, num_proposals]

        batch_size=gt_boxes.shape[0]
        offset=torch.arange(0,batch_size)*gt_boxes.shape[1]
        offset=argmax_overlaps+offset.view(-1,1).type_as(argmax_overlaps)

        # proposals打标不是仅区分正负样本，要找到对应的类别标签（voc数据集有21类）
        labels=gt_boxes[:,:,4].contiguous().view(-1)[(offset.view(-1),)].view(batch_size,-1) # [b, num_proposals]

        # 2.筛选proposals得到rois及其对应的标签和gt_boxes
        labels_batch=gt_boxes.new(batch_size,rois_per_image).zero_() # 每个roi对应的标签
        rois_batch=all_rois.new(batch_size,rois_per_image,5).zero_() # 每个roi的坐标以及batch_index
        gt_rois_batch=all_rois.new(batch_size,rois_per_image,5).zero_() # 每个roi对应的gt_box
        for i in range(batch_size):
            # 正负样本筛选
            fg_inds = torch.nonzero(max_overlaps[i] >= cfg.TRAIN.FG_THRESH).view(-1)
            fg_num_rois = fg_inds.numel()

            bg_inds = torch.nonzero((max_overlaps[i] < cfg.TRAIN.BG_THRESH_HI) &
                                    (max_overlaps[i] >= cfg.TRAIN.BG_THRESH_LO)).view(-1)
            bg_num_rois = bg_inds.numel()

            if fg_num_rois>0 and bg_num_rois>0:
                fg_rois_per_this_image=min(fg_num_rois,fg_rois_per_image)
                randnum=torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(gt_boxes).long()
                fg_inds=fg_inds[randnum[:fg_rois_per_this_image]]

                # 此处采用如下代码得到对rois进行采样，可能会重复采样同一roi，但是能够避免正样本或者负样本不足
                # 设置的rois_per_image，重复采样毕竟要强于对roi填零
                bg_rois_per_this_image=rois_per_image-fg_rois_per_this_image
                randnum=np.floor(np.random.rand(bg_rois_per_this_image)*bg_num_rois)
                randnum=torch.from_numpy(randnum).type_as(gt_boxes).long()
                bg_inds=bg_inds[randnum]
            elif fg_num_rois>0 and bg_num_rois==0:
                randnum=np.floor(np.random.rand(rois_per_image)*fg_num_rois)
                randnum=torch.from_numpy(randnum).type_as(gt_boxes).long()
                fg_inds=fg_inds[randnum]
                fg_rois_per_this_image=rois_per_image
                bg_rois_per_this_image=0
            elif fg_num_rois==0 and bg_num_rois>0:
                randnum=np.floor(np.random.rand(rois_per_image)*bg_num_rois)
                randnum=torch.from_numpy(randnum).type_as(gt_boxes).long()
                bg_inds=bg_inds[randnum]
                fg_rois_per_this_image=0
                bg_rois_per_this_image=rois_per_image
            else:
                raise ValueError("bg_num_rois = 0 and fg_num_rois = 0, this should not happen!")

            keep_inds=torch.cat([fg_inds,bg_inds],dim=0)

            labels_batch[i].copy_(labels[i][keep_inds])
            # 所有负样本标签全部设为0，即background
            if fg_rois_per_this_image<rois_per_image:
                labels_batch[i][fg_rois_per_this_image:]=0

            rois_batch[i]=all_rois[i][keep_inds]
            rois_batch[i,:,0]=i # [batch_index,x1,y1,x2,y2]

            gt_rois_batch[i]=gt_boxes[i][argmax_overlaps[i][keep_inds]]

        # 3.计算回归偏移真值rois_targets
        rois_targets=self._compute_rois_targets(rois_batch[:,:,1:5],gt_rois_batch[:,:,:4]) # [b, num_rois (rois_per_image), 4]
        # 此处是筛选用于计算回归损失的rois_target，其仅计算属于fg类别的rois_targets
        rois_targets_reg, rois_inside_weights=self._compute_rois_targets_regression(rois_targets,labels_batch) # [b, num_rois (rois_per_image), 4]
        # 其实也可以按照之前计算anchors_targets的写法，得到inside_weights即可，后续计算损失再根据inside_weights筛选targets

        return rois_batch,labels_batch,rois_targets_reg,rois_inside_weights

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
