import torch
import torch.nn as nn
import torchvision.models as models
from model.faster_rcnn.faster_rcnn import Faster_RCNN

# 继承Faster_RCNN，此时子类继承父类中一些属性或者方法，并且在父类中也可以使用子类中的属性或者方法（前提是程序以子类为主体，
# 实例化子类，若直接实例化父类且父类使用子类的东西，则会报错）
class vgg16(Faster_RCNN):
    def __init__(self,classes,pretrained=False, class_agnostic=False):
        self.classes=classes
        self.pretrained=pretrained
        self.pretrained_model_path='data/pretrained_model/vgg16_caffe.pth'
        self.class_agnostic=class_agnostic
        self.feature_out_dim=512

        Faster_RCNN.__init__(self,classes,class_agnostic)

    def _init_modules(self):
        model=models.vgg16()
        if self.pretrained:
            print('load pretrained model from path:{}'.format(self.pretrained_model_path))
            params=torch.load(self.pretrained_model_path)
            model.load_state_dict({k:v for k,v in params.items() if k in model.state_dict()})

        # 不使用最后一层max_pool
        self.RCNN_base=nn.Sequential(*list(model.features._modules.values())[:-1])
        # 除去最后的全连接层
        self.RCNN_top=nn.Sequential(*list(model.classifier._modules.values())[:-1])

        # 获取类别分数
        self.RCNN_cls_score=nn.Linear(4096,self.n_classes)
        # 获取框回归预测偏移
        if self.class_agnostic:
            self.RCNN_bbox_pred=nn.Linear(4096,4) # 不考虑roi类别，只预测偏移
        else:
            self.RCNN_bbox_pred = nn.Linear(4096, 4*self.n_classes) # 预测roi在每个类别下的偏移

        # 冻结特征提取模块前10层，即conv3之前
        for i in range(10):
            for p in self.RCNN_base[i].parameters():
                p.requires_grad = False

    # 在pooling层之后
    def head_to_tail(self, pool):
        pool_flat = pool.view(pool.size(0), -1)
        fc = self.RCNN_top(pool_flat)
        return fc