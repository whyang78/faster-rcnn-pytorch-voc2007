import os
import pickle
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import uuid
from model.utils.config import cfg
from model.utils.voc_eval import voc_eval

def prepare_roidbs(imdb):
    num_images=imdb.num_images
    sizes=[Image.open(imdb.image_path_from_index(i)).size for i in range(num_images)]
    roidbs=imdb.roidb
    for i in range(num_images):
        roidbs[i]['img_id']=i
        roidbs[i]['image']=imdb.image_path_from_index(i)
        roidbs[i]['width']=sizes[i][0]
        roidbs[i]['height']=sizes[i][1]

        gt_overlaps=roidbs[i]['gt_overlaps']
        max_overlap=np.max(gt_overlaps,axis=1)
        max_class=np.argmax(gt_overlaps,axis=1)
        roidbs[i]['max_overlaps'] = max_overlap
        roidbs[i]['max_classes'] = max_class

        # max overlap of 0 => class should be zero (background)
        zero_ind = np.where(max_overlap == 0)[0]
        assert all(max_class[zero_ind] == 0)
        # max overlap > 0 => class should not be zero (must be a fg class)
        nonzero_ind = np.where(max_overlap > 0)[0]
        assert all(max_class[nonzero_ind] != 0)

def get_train_roidbs(imdb):
    if cfg.TRAIN.USE_FLIPPED:
        print('use flipped !')
        imdb.append_flipped_roidbs()
    prepare_roidbs(imdb)
    print('prepare done !')
    return imdb.roidb

def get_roidbs(imdb_name):
    imdb=pascal_voc(imdb_name)
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print('set proposal method:{}'.format(cfg.TRAIN.PROPOSAL_METHOD))
    roidbs=get_train_roidbs(imdb)
    return roidbs

def filter_roidbs(roidbs):
    num_roidb = len(roidbs)
    print('before filter,there are %d roidbs!' % num_roidb)
    i = 0
    while i < num_roidb:
        if len(roidbs[i]['boxes']) == 0:
            del roidbs[i]
            i -= 1
        i += 1
    print('after filter,there are %d/%d roidbs!' % (i,len(roidbs)))
    return roidbs

def rank_roidb_ratio(roidbs):
    num_roidbs=len(roidbs)
    ratio_small=0.5
    ratio_large=2.0

    ratio_list=[]
    for i in range(num_roidbs):
        width=roidbs[i]['width']
        height=roidbs[i]['height']
        ratio=float(width)/float(height)
        if ratio>ratio_large:
            ratio=ratio_large
            roidbs[i]['need_crop']=1
        elif ratio<ratio_small:
            ratio=ratio_small
            roidbs[i]['need_crop'] = 1
        else:
            roidbs[i]['need_crop'] = 0
        ratio_list.append(ratio)

    ratio_list=np.array(ratio_list)
    ratio_index=np.argsort(ratio_list)
    ratio_list=ratio_list[ratio_index]
    return ratio_list,ratio_index

def get_imdb_and_roidbs(imdb_name,training=True):
    '''
    :return:
        imdb：pascal_voc对象
        roidbs: 训练对象。每个训练样本包含元素如下：
            通用(包含origin和flipped)：
                boxes：坐标(x1,y1,x2,y2)，维度[num_objs,4]
                gt_overlaps：交并比，维度[num_objs,num_classes]
                gt_classes：框对应类别，维度[num_objs]
                flipped：True or False。
                img_id：样本序号。(0,1,2,....)
                image：样本路径。
                width：样本宽度。
                height：样本高度。
                max_classes：框最大交并比对应的类别，维度[num_objs]
                max_overlaps：框最大交并比，维度[num_objs]
                need_crop：0 or 1。
            origin特有：
                gt_ishard：框对应的是否有难度，维度[num_objs]
                seg_areas：框的面积大小，维度[num_objs]
            说明：
                num_objs为每个样本包含的框个数(不同样本对应的框个数不尽相同)
                num_classes为类别总数，此处对应voc2007数据集，故为21
        ratio_list：对所有样本的宽高比递增排序的结果。
        ratio_index：对所有样本的宽高比递增排序对应的索引。
    '''
    roidbs=get_roidbs(imdb_name)
    imdb=pascal_voc(imdb_name)

    if training:
        roidbs=filter_roidbs(roidbs)
    ratio_list, ratio_index=rank_roidb_ratio(roidbs)
    return imdb,roidbs,ratio_list,ratio_index


class pascal_voc(object):
    def __init__(self,imdb_name,devkit_path=None):
        self._year=str(imdb_name).split('_')[-2]
        self._image_set=str(imdb_name).split('_')[-1]
        self._name=imdb_name
        self._devkit_path=self._set_default_devkit_path() if devkit_path is None else devkit_path
        self._data_path=os.path.join(self._devkit_path,'VOC'+self._year)

        self._classes=('__background__',  # always index 0
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
        self._class_to_id=dict(zip(self._classes,range(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index=self._load_image_set_index()

        self._roidb=None
        # 该类中只是使用了gt_roidb，还有另外一种方法，此处没有细写
        self._roidb_handler=self.gt_roidb

        # 评估所用
        self._salt = str(uuid.uuid4()) # 当前网关和时间组成的随机字符串
        self._comp_id = 'comp4'
        self.config = {'cleanup': True, # 决定是否删除按类保存的检测结果
                       'use_salt': True, # 保存检测结果进行命名时，是否添加随机数
                       'use_diff': False,
                       # 'matlab_eval': False, # 是否使用MATLAB版的评估，这里直接去掉了，咱不用
                       'rpn_file': None,
                       'min_size': 2}

        assert os.path.exists(self._devkit_path), \
            'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @property
    def num_images(self):
        return len(self._image_index)

    @property
    def image_index(self):
        return self._image_index

    @property
    def name(self):
        return self._name

    @property
    def roidb_handler(self):
        return self._roidb_handler

    @roidb_handler.setter
    def roidb_handler(self,val):
        self._roidb_handler=val

    @property
    def roidb(self):
        if self._roidb is not None:
            return self._roidb
        self._roidb=self.roidb_handler()
        return self._roidb

    @property
    def cache_path(self):
        path=os.path.join(cfg.DATA_DIR,'cache')
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def _set_default_devkit_path(self):
        return os.path.join(cfg.DATA_DIR,'VOCdevkit'+self._year)

    def _load_image_set_index(self):
        image_set_path=os.path.join(self._data_path,'ImageSets','Main',self._image_set+'.txt')
        assert os.path.exists(image_set_path),'path not exits:{}'.format(image_set_path)
        with open(image_set_path,'r+') as f:
            image_index=f.readlines()
        return [img_index.strip() for img_index in image_index]

    def set_proposal_method(self,method):
        self.roidb_handler=eval('self.'+method+'_roidb')

    def _load_pascal_annotation(self,img_index):
        anno_path=os.path.join(self._data_path,'Annotations',img_index+'.xml')
        tree=ET.parse(anno_path)
        objs=tree.findall('object')
        num_objs=len(objs)

        bboxes=np.zeros((num_objs,4),dtype=np.uint16)
        gt_classes=np.zeros((num_objs),dtype=np.uint32)
        gt_overlaps=np.zeros((num_objs,self.num_classes),dtype=np.float32)
        gt_ishards=np.zeros((num_objs),dtype=np.int32)
        seg_areas=np.zeros((num_objs),dtype=np.float32)

        for i,obj in enumerate(objs):
            bbox=obj.find('bndbox')
            # voc数据集中坐标是one-base，这里转换成zero-base
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            bboxes[i,:]=[x1,y1,x2,y2]

            class_name=obj.find('name').text
            class_id=self._class_to_id[class_name.lower().strip()]
            gt_classes[i]=class_id
            gt_overlaps[i,class_id]=1.0

            difficult=obj.find('difficult')
            diffi=0 if difficult is None else difficult.text
            gt_ishards[i]=diffi

            seg_areas[i]=(x2-x1+1)*(y2-y1+1)

        return {'boxes':bboxes,
                'gt_classes':gt_classes,
                'gt_overlaps':gt_overlaps,
                'gt_ishard':gt_ishards,
                'seg_areas':seg_areas,
                'flipped':False}

    def gt_roidb(self):
        roidb_path=os.path.join(self.cache_path,self._name+'_gt_roidbs.pkl')
        if os.path.exists(roidb_path):
            with open(roidb_path,'rb') as f:
                roidbs=pickle.load(f)
                print('{} load cache from {}'.format(self._name, roidb_path))
                return roidbs

        roidbs=[self._load_pascal_annotation(index) for index in self.image_index]
        with open(roidb_path,'wb') as f:
            pickle.dump(roidbs,f,pickle.HIGHEST_PROTOCOL)
        print('write cache in {}'.format(roidb_path))
        return roidbs

    def image_path_from_index(self,ix):
        image_path=os.path.join(self._data_path,'JPEGImages',self.image_index[ix]+self._image_ext)
        assert os.path.exists(image_path), 'path not exits:{}'.format(image_path)
        return image_path

    def _get_widths(self):
        widths=[]
        for i in range(self.num_images):
            image_path=self.image_path_from_index(i)
            width=Image.open(image_path).size[0]
            widths.append(width)
        return widths

    def append_flipped_roidbs(self):
        widths=self._get_widths()
        for i in range(self.num_images):
            bbox=self.roidb[i]['boxes'].copy()
            old_x1=bbox[:,0].copy()
            old_x2=bbox[:,2].copy()
            bbox[:,0]=widths[i]-old_x2-1
            bbox[:,2]=widths[i]-old_x1-1
            assert (bbox[:,2]>=bbox[:,0]).all()

            entry={
                'boxes': bbox,
                'gt_classes': self.roidb[i]['gt_classes'],
                'gt_overlaps': self.roidb[i]['gt_overlaps'],
                'flipped': True
            }
            self.roidb.append(entry)
        self._image_index=self._image_index * 2

 #### 以下函数全部是用于测试评估

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

    def _get_comp_id(self):
        # 注意competition_mode的设置
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        filedir = os.path.join(self._devkit_path, 'results', 'VOC' + self._year, 'Main')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} VOC results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # VOC数据坐标是one-base，模型得到的结果是zero-base，故进行转换
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir):
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        annopath = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache') # 用于存储图像真实框的缓存文件路径
        #PASCAL VOC评估方法主要有两种，在2010年发生变化
        use_07_metric = True if int(self._year) < 2010 else False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))

        # 计算各类别的AP，并计算mAP，打印结果
        aps = []
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, overlap_thresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(all_boxes) # 将检测结果按类别分开保存下来(其实就是检测结果的缓存文件)
        # 存储形式：各个TXT文档以类别命名，每个文档每行包含image_index_name、置信度、one-base坐标

        self._do_python_eval(output_dir) # 评估
        if self.config['cleanup']: # 决定是否删除按类保存的检测结果，即第一步代码保存的结果
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)

