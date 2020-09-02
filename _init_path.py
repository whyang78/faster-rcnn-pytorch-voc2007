import sys
import os

print('init path ......')

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0,path)

# 根目录路径
root_dir=os.path.dirname(__file__)

# 将lib添加到路径
lib_path=os.path.join(root_dir,'lib')
add_path(lib_path)

# 将coco api添加到路径
coco_path=os.path.join(root_dir,'data','coco','PythonAPI')
add_path(coco_path)