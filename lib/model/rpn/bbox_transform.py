import torch

def bbox_transform_inv(boxes, deltas, batch_size):
    '''
    该函数应用于训练时根据anchors和RPN预测偏移得到proposals，
    或者测试和demo时根据rois与RCNN预测偏移得到最终预测框坐标。

    (1)训练时求解proposals
    :param boxes: 所有anchors [b, num_featpoints*num_anchors, 4]
    :param deltas: RPN预测回归偏移 [b, num_featpoints*num_anchors, 4]
    :param batch_size:
    :return: 回归偏移调整anchors得到pred，[b, num_featpoints*num_anchors, 4]

    (2)demo求解最终预测坐标
    :param boxes: 所有rois [b, num_rois, 4]
    :param deltas: RCNN预测回归偏移 [b, num_rois, 4(或4*21)] (针对voc数据集的情况)
    :param batch_size:
    :return: 回归偏移调整rois得到pred，[b, num_rois, 4(或4*21)]
        注意deltas的size(2)不一定，根据class_agnostic而定。这也是为何dx=deltas[:,:,0::4]
    这样写，而不是dx=deltas[:,:,0](dy,dw,dh同理)。

    '''
    boxes_width=boxes[:,:,2]-boxes[:,:,0]+1
    boxes_height=boxes[:,:,3]-boxes[:,:,1]+1
    boxes_ctr_x=boxes[:,:,0]+0.5*boxes_width
    boxes_ctr_y=boxes[:,:,1]+0.5*boxes_height

    dx=deltas[:,:,0::4]
    dy=deltas[:,:,1::4]
    dw=deltas[:,:,2::4]
    dh=deltas[:,:,3::4]

    pred_ctr_x=dx*boxes_width.unsqueeze(2)+boxes_ctr_x.unsqueeze(2)
    pred_ctr_y=dy*boxes_height.unsqueeze(2)+boxes_ctr_y.unsqueeze(2)
    pred_width=boxes_width.unsqueeze(2)*torch.exp(dw)
    pred_height=boxes_height.unsqueeze(2)*torch.exp(dh)

    pred=deltas.clone()
    pred[:,:,0::4]=pred_ctr_x-0.5*pred_width
    pred[:,:,1::4]=pred_ctr_y-0.5*pred_height
    pred[:, :, 2::4] = pred_ctr_x+0.5 * pred_width
    pred[:, :, 3::4] = pred_ctr_y+0.5 * pred_height
    return pred


def clip_boxes(boxes, im_shape, batch_size):
    for i in range(batch_size):
        boxes[i,:,0::4].clamp_(0, im_shape[i, 1]-1)
        boxes[i,:,1::4].clamp_(0, im_shape[i, 0]-1)
        boxes[i,:,2::4].clamp_(0, im_shape[i, 1]-1)
        boxes[i,:,3::4].clamp_(0, im_shape[i, 0]-1)

    return boxes

def bbox_overlaps_batch(anchors,gt_boxes):
    batch_size=gt_boxes.shape[0]

    if anchors.dim()==2:
        '''
        anchor_target_layer.py中调用，主要计算anchors与整个batch中的gt_boxes计算overlaps，传入的
        anchors只是单个样本上的，是因为整个批次所有样本的anchors都是一样的，对于同等大小的特征图，先验框
        都是一样的，故可以对其expand到整个batch上。
        
        传入参数：
            anchors:[num_anchors,4]
            gt_boxes:[b,num_gt_boxes,5]
        返回结果：
            overlaps:[b,num_anchors,num_gt_boxes]
        '''
        num_anchors=anchors.shape[0]
        num_gt_boxes=gt_boxes.shape[1]

        anchors=anchors.view(1,num_anchors,4).expand(batch_size,num_anchors,4).contiguous()
        gt_boxes=gt_boxes[:,:,:4].contiguous()

        anchors_w=anchors[:,:,2]-anchors[:,:,0]+1
        anchors_h=anchors[:,:,3]-anchors[:,:,1]+1
        anchors_areas=(anchors_w*anchors_h).view(batch_size,num_anchors,1)

        gt_boxes_w=gt_boxes[:,:,2]-gt_boxes[:,:,0]+1
        gt_boxes_h=gt_boxes[:,:,3]-gt_boxes[:,:,1]+1
        gt_boxes_areas=(gt_boxes_w*gt_boxes_h).view(batch_size,1,num_gt_boxes)

        anchors_=anchors.view(batch_size,num_anchors,1,4).expand(batch_size,num_anchors,num_gt_boxes,4)
        gt_boxes_=gt_boxes.view(batch_size,1,num_gt_boxes,4).expand(batch_size,num_anchors,num_gt_boxes,4)

        iws=torch.min(anchors_[:,:,:,2],gt_boxes_[:,:,:,2])-torch.max(anchors_[:,:,:,0],gt_boxes_[:,:,:,0])+1
        iws[iws<0]=0

        ihs=torch.min(anchors_[:,:,:,3],gt_boxes_[:,:,:,3])-torch.max(anchors_[:,:,:,1],gt_boxes_[:,:,:,1])+1
        ihs[ihs<0]=0

        unions=anchors_areas+gt_boxes_areas-iws*ihs
        overlaps=(iws*ihs)/unions  # 维度[b, num_anchors, num_gt_boxes]

        gt_boxes_zero=(gt_boxes_w==1)&(gt_boxes_h==1)
        anchors_zero=(anchors_w==1)&(anchors_h==1)

        overlaps.masked_fill_(gt_boxes_zero.view(batch_size,1,num_gt_boxes).expand(batch_size,num_anchors,num_gt_boxes),0)
        overlaps.masked_fill_(anchors_zero.view(batch_size,num_anchors,1).expand(batch_size,num_anchors,num_gt_boxes),-1)
    elif anchors.dim()==3:
        '''
        proposal_target_layer.py中调用，主要用于计算所有proposals（即未筛选rois）与gt_boxes的交并比。因为每个batch中不同样本
        对应的proposals不一样，故传入dim=3的矩阵。
        
        传入参数：
            anchors(proposals):[b,num_proposals,4]
            gt_boxes:[b,num_gt_boxes,5]
        返回结果：
            overlaps:[b,num_proposals,num_gt_boxes]
        '''
        num_proposals=anchors.shape[1]
        num_gt_boxes=gt_boxes.shape[1]
        if anchors.shape[2]==4:
            proposals=anchors[:,:,:4].contiguous()
        else:
            proposals=anchors[:,:,1:5].contiguous()
        gt_boxes=gt_boxes[:,:,:4].contiguous()

        gt_boxes_w=gt_boxes[:,:,2]-gt_boxes[:,:,0]+1
        gt_boxes_h=gt_boxes[:,:,3]-gt_boxes[:,:,1]+1
        gt_boxes_areas=(gt_boxes_w*gt_boxes_h).view(batch_size,1,num_gt_boxes)

        proposals_w=proposals[:,:,2]-proposals[:,:,0]+1
        proposals_h=proposals[:,:,3]-proposals[:,:,1]+1
        proposals_areas=(proposals_w*proposals_h).view(batch_size,num_proposals,1)

        gt_boxes_=gt_boxes.view(batch_size,1,num_gt_boxes,4).expand(batch_size,num_proposals,num_gt_boxes,4)
        proposals_=proposals.view(batch_size,num_proposals,1,4).expand(batch_size,num_proposals,num_gt_boxes,4)

        iws=torch.min(gt_boxes_[:,:,:,2],proposals_[:,:,:,2])-torch.max(gt_boxes_[:,:,:,0],proposals_[:,:,:,0])+1
        iws[iws<0]=0

        ihs=torch.min(gt_boxes_[:,:,:,3],proposals_[:,:,:,3])-torch.max(gt_boxes_[:,:,:,1],proposals_[:,:,:,1])+1
        ihs[ihs<0]=0

        unions=gt_boxes_areas+proposals_areas-(iws*ihs)
        overlaps=(iws*ihs)/unions

        gt_boxes_zero=(gt_boxes_w==1) & (gt_boxes_h==1)
        proposals_zero=(proposals_w==1) & (proposals_h==1)

        overlaps.masked_fill_(gt_boxes_zero.view(batch_size,1,num_gt_boxes).expand(batch_size,num_proposals,num_gt_boxes),0)
        overlaps.masked_fill_(proposals_zero.view(batch_size,num_proposals,1).expand(batch_size,num_proposals,num_gt_boxes),-1)
    else:
        raise ValueError('anchors input dimension is not correct.')

    return overlaps

def bbox_transform_batch(ex_rois,gt_rois):
    if ex_rois.dim()==2:
        '''
        anchor_target_layer.py中调用，主要计算anchors与其对应的gt_boxes的回归偏移真值。原因同bbox_overlaps_batch。
        
        传入参数：
            ex_rois:[num_anchors,4]
            gt_rois:[b,num_anchors,5]
        返回结果：
            targets:[b,num_anchors,4]
        '''
        ex_width=ex_rois[:,2]-ex_rois[:,0]+1
        ex_height=ex_rois[:,3]-ex_rois[:,1]+1
        ex_ctr_x=ex_rois[:,0]+0.5*ex_width
        ex_ctr_y=ex_rois[:,1]+0.5*ex_height

        gt_width=gt_rois[:,:,2]-gt_rois[:,:,0]+1
        gt_height=gt_rois[:,:,3]-gt_rois[:,:,1]+1
        gt_ctr_x=gt_rois[:,:,0]+0.5*gt_width
        gt_ctr_y=gt_rois[:,:,1]+0.5*gt_height

        target_dx=(gt_ctr_x-ex_ctr_x.view(1,-1).expand_as(gt_ctr_x))/ex_width
        target_dy=(gt_ctr_y-ex_ctr_y.view(1,-1).expand_as(gt_ctr_y))/ex_height
        target_dw=torch.log(gt_width/ex_width.view(1,-1).expand_as(gt_width))
        target_dh = torch.log(gt_height / ex_height.view(1, -1).expand_as(gt_height))
    elif ex_rois.dim()==3:
        '''
        proposal_target_layer.py中调用，主要用于通过已筛选rois与其对应gt_boxes计算回归预测偏差真值。
        原因同bbox_overlaps_batch。
        
        传入参数：
            ex_rois:[b,num_rois,4]
            gt_rois:[b,num_rois,4]
        返回结果：
            targets:[b,num_rois,4]
        '''
        ex_width = ex_rois[:, :,2] - ex_rois[:, :,0] + 1
        ex_height = ex_rois[:, :,3] - ex_rois[:, :,1] + 1
        ex_ctr_x = ex_rois[:, :,0] + 0.5 * ex_width
        ex_ctr_y = ex_rois[:, :,1] + 0.5 * ex_height

        gt_width = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1
        gt_height = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_width
        gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_height

        target_dx = (gt_ctr_x - ex_ctr_x) / ex_width
        target_dy = (gt_ctr_y - ex_ctr_y) / ex_height
        target_dw = torch.log(gt_width / ex_width)
        target_dh = torch.log(gt_height / ex_height)
    else:
        raise ValueError('anchors input dimension is not correct.')

    targets=torch.stack((target_dx,target_dy,target_dw,target_dh),dim=2)
    return targets


