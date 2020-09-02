import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from model.utils.config import cfg
import cv2


def smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    sigma_2=sigma**2
    bbox_diff=bbox_targets-bbox_pred
    bbox_diff=bbox_inside_weights*bbox_diff # 只对fg样本计算回归损失

    abs_bbox_diff=torch.abs(bbox_diff)
    sign_=(abs_bbox_diff<(1.0/sigma_2)).detach_().float()
    loss = sign_*torch.pow(abs_bbox_diff,2)*sigma_2*0.5 + (1-sign_)*(abs_bbox_diff-0.5/sigma_2)
    output_loss=bbox_outside_weights*loss # outside_weights平衡分类与回归损失权重
    for i in sorted(dim,reverse=True): # 对特征图中所有anchors的dx,dy,dw,dh计算总损失，故对后三维度进行求和操作
        output_loss=output_loss.sum(dim=i)
    output_loss=output_loss.mean() # 计算该batch的平均损失
    return output_loss

def adjust_learning_rate(optimizer,lr_decay):
    for param_group in optimizer.param_groups():
        param_group['lr']*=lr_decay

def clip_gradient(model,clip_norm):
    total_norm=0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            param_norm=p.grad.norm()
            total_norm+=param_norm**2
    total_norm=torch.sqrt(total_norm).item()
    norm_=clip_norm/max(clip_norm,total_norm)
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            p.grad.mul_(norm_)

def save_checkpoint(state, filename):
    torch.save(state, filename)

def vis_detections(im, class_name, dets, max_vis_boxes=10, thresh=0.5):
    for i in range(np.minimum(max_vis_boxes,dets.shape[0])):
        bbox=tuple(int(np.round(x)) for x in dets[i,:-1])
        score=dets[i,-1]
        if score>thresh:
            cv2.rectangle(im,bbox[0:2],bbox[2:4],(0, 204, 0), 2)
            cv2.putText(im,'%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 255), thickness=1)
    return im