import numpy as np
import pickle
import xml.etree.ElementTree as ET
import os


def get_gt_annotation(filename):
    tree=ET.parse(filename)
    objects_info=[]
    for obj in tree.findall('object'):
        obj_info={}
        obj_info['name']=obj.find('name').text
        obj_info['pose'] = obj.find('pose').text
        obj_info['truncated'] = int(obj.find('truncated').text)
        obj_info['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_info['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]

        objects_info.append(obj_info)

    return objects_info

def voc_ap(rec,prec,use_07_metric):
    # 两种计算ap的方法
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             overlap_thresh=0.5,
             use_07_metric=False):
    '''
    :param detpath:  该类别对应的各样本检测结果
    :param annopath:    annotation路径(xml) 可拼接image name，得到指定image的annotation
    :param imagesetfile: imageset文件路径（txt）
    :param classname: 类别名称
    :param cachedir: 各图像annotation信息缓存文件(pkl)
    :param overlap_thresh: 判为正例的阈值
    :param use_07_metric: 是否使用VOC2007版的评估方法，即11points
    :return:
    '''
    if not os.path.isdir(cachedir):
        os.makedirs(cachedir)
    cachefile=os.path.join(cachedir,'%s_annots.pkl' % imagesetfile)

    with open(imagesetfile,'r+') as f:
        lines=f.readlines()
    imagenames=[line.strip() for line in lines]

    # 得到各图像annotation信息
    if not os.path.isfile(cachefile):
        gt_annotations={}
        for i,img_name in enumerate(imagenames):
            gt_annotations[img_name]=get_gt_annotation(annopath.format(img_name))
            if (i+1)%100==0:
                print('Reading annotation for {:d}/{:d}'.format(
                        i + 1, len(imagenames)))

        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(gt_annotations, f)
    else:
        with open(cachefile,'rb') as f:
            try:
                gt_annotations=pickle.load(f)
            except:
                gt_annotations=pickle.load(f,encoding='bytes')

    # 得到该类别下各图像真实框坐标
    gt_cls_bbox={}
    npos=0 # 正样本数目
    for img_name in imagenames:
        cls_objs=[obj for obj in gt_annotations[img_name] if obj['name']==classname]
        obj_bbox=np.array([cls_obj['bbox'] for cls_obj in cls_objs])
        difficult=np.array([cls_obj['difficult'] for cls_obj in cls_objs]).astype(np.bool)
        det=[False]*len(cls_objs) # 标识是否已使用过该gt_bbox

        npos+=sum(~difficult)
        gt_cls_bbox[img_name]={
            'bbox': obj_bbox,
            'difficult': difficult,
            'det': det
        }

    # 载入该类别检测结果
    detfile=detpath
    with open(detpath,'r+') as f:
        detections=f.readlines()

    detections_split=[det.strip().split() for det in detections]
    image_ids=[split[0] for split in detections_split]
    detections_confidence=np.array([float(split[1]) for split in detections_split])
    detections_bboxes=np.array([[float(x) for x in split[2:]]for split in detections_split])

    # 计算tp fp 其对应每个预测框
    num_bboxes=len(detections_bboxes)
    tp=np.zeros(num_bboxes)
    fp=np.zeros(num_bboxes)

    if num_bboxes>0:
        # 从大到小排序
        scores_ind=np.argsort(-1*detections_confidence)
        sorted_bboxes=detections_bboxes[scores_ind,:]
        sorted_img_ids=[image_ids[ind] for ind in scores_ind]

        for i in range(num_bboxes):
            img_name=sorted_img_ids[i]
            gt_bboxes_info=gt_cls_bbox[img_name]

            gt_bboxes=gt_bboxes_info['bbox'].astype(float)
            bbox=sorted_bboxes[i,:].astype(float)
            overlap_max=-np.inf

            if gt_bboxes.shape[0]>0:
                x1_max=np.maximum(gt_bboxes[:,0],bbox[0])
                y1_max=np.maximum(gt_bboxes[:,1],bbox[1])
                x2_min=np.minimum(gt_bboxes[:,2],bbox[2])
                y2_min=np.minimum(gt_bboxes[:,3],bbox[3])

                iws=np.maximum(x2_min-x1_max+1,0)
                ihs=np.maximum(y2_min-y1_max+1,0)
                inters=iws*ihs

                gt_bboxes_area=(gt_bboxes[:,3]-gt_bboxes[:,1]+1)*(gt_bboxes[:,2]-gt_bboxes[:,0]+1)
                bbox_area=(bbox[3]-bbox[1]+1)*(bbox[2]-bbox[0]+1)
                unions=gt_bboxes_area+bbox_area-inters

                overlaps=inters/unions
                argmax_overlaps=np.argmax(overlaps)
                overlap_max=np.max(overlaps)

            if overlap_max>overlap_thresh:
                # difficult的样本不参与评估
                if not gt_bboxes_info['difficult'][argmax_overlaps]:
                    if not gt_bboxes_info['det'][argmax_overlaps]:
                        tp[i]=1
                        gt_bboxes_info['det'][argmax_overlaps]=True
                    else:
                        fp[i]=1
            else:
                # 两种情况：该预测框所在图像并没有对应类别的gt框；交并比阈值不太够
                fp[i]=1

    # 计算 precision recall 和 ap
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec,prec,ap







