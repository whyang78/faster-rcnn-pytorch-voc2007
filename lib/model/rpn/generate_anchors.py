import numpy as np

# 生成回归框 9*4
def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2**np.arange(3, 6)):

    base_anchor=np.array([1,1,base_size,base_size])-1
    ratios_anchors=ratio_enum(base_anchor,ratios) # 宽高比变换

    anchors=np.vstack([scale_enum(ratios_anchors[i],scales)
                       for i in range(ratios_anchors.shape[0])]) # 尺度变化
    return anchors

def get_whctrs(anchor):
    '''
    :param anchor:[x1,y1,x2,y2]
    :return: [x_ctr,y_ctr,w,h]
    '''
    w=anchor[2]-anchor[0]+1
    h=anchor[3]-anchor[1]+1
    x_ctr=anchor[0]+0.5*(w-1)
    y_ctr=anchor[1]+0.5*(h-1)

    return x_ctr,y_ctr,w,h

def transfer_anchors(x_ctr,y_ctr,ws,hs):
    '''
    :return: [x1,y1,x2,y2]
    '''
    ws=ws[:,np.newaxis]
    hs=hs[:,np.newaxis]

    x1_s=x_ctr-0.5*(ws-1)
    y1_s=y_ctr-0.5*(hs-1)
    x2_s=x_ctr+0.5*(ws-1)
    y2_s=y_ctr+0.5*(hs-1)

    anchors=np.hstack((x1_s,y1_s,x2_s,y2_s))
    return anchors

def ratio_enum(anchor,ratios):
    x_ctr,y_ctr,w,h=get_whctrs(anchor)

    size_=w*h #宽高比变化保持面积不变
    new_ws=np.round(np.sqrt(size_/ratios))
    new_hs=np.round(new_ws*ratios)

    anchors=transfer_anchors(x_ctr,y_ctr,new_ws,new_hs)
    return anchors


def scale_enum(anchor,scales):
    x_ctr, y_ctr, w, h = get_whctrs(anchor)

    new_ws=w*scales
    new_hs=h*scales

    anchors=transfer_anchors(x_ctr,y_ctr,new_ws,new_hs)
    return anchors

if __name__ == '__main__':
    a=generate_anchors()