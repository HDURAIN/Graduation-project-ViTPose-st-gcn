import argparse
import os
import os.path as osp

import torch
from torch import Tensor

from pathlib import Path
import cv2
import numpy as np


from time import time
from PIL import Image
from torchvision.transforms import transforms

from net.vitmodel import ViTPose
from tools.utils.vp_visualization import draw_points_and_skeleton, joints_dict
from tools.utils.dist_util import get_dist_info, init_dist
from tools.utils.top_down_eval import keypoints_from_heatmaps
from tools.estimate_st import estimate

__all__ = ['inference']
    

if __name__ == "__main__":
    
    # 获取命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', nargs='+', type=str, default='/hy-tmp/demo_resource/demo_pic/1.jpg', help='image path(s)')
    parser.add_argument('--ckpt_path', type=str, default='/hy-tmp/models/pretrained_vit/vitpose-b.pth', help='ckpt path')
    parser.add_argument('--save_dir', type=str, default='/hy-tmp/result/image_result_vit', help='save path')
    args = parser.parse_args()
    
    # 原实现总是要自己手动改一堆地方太不方便，这里根据输入的ckpt文件名自动导入相关cfg文件
    ckpt_name = os.path.basename(args.ckpt_path)
    if ckpt_name == "vitpose-l.pth":
        from config.ViTPose_large_coco_256x192 import model as model_cfg
        from config.ViTPose_large_coco_256x192 import data_cfg
    elif ckpt_name == "vitpose-h.pth":
        from config.ViTPose_huge_coco_256x192 import model as model_cfg
        from config.ViTPose_huge_coco_256x192 import data_cfg
    elif ckpt_name == "vitpose-b.pth":
        from config.ViTPose_base_coco_256x192 import model as model_cfg
        from config.ViTPose_base_coco_256x192 import data_cfg
        
    # 设置预训练模型文件路径
    # CUR_DIR = osp.dirname(__file__)
    # CKPT_PATH = f"{CUR_DIR}/vitpose-b-multi-coco.pth"
    CKPT_PATH = args.ckpt_path
    
    # 可以同时输入多张图像的路径
    img_size = data_cfg['image_size']
    if type(args.image_path) != list:
         args.image_path = [args.image_path]
    for img_path in args.image_path:
        print('Read image from ' + img_path)
        img_name = img_path.split('/')[-1].split('.')[0]
        img_type = img_path.split('/')[-1].split('.')[1]
        img = Image.open(img_path)
        # 每张输入的图像调用一次estimate函数 返回为 point
        keypoints = estimate(img=img, img_size=img_size, model_cfg=model_cfg, ckpt_path=CKPT_PATH, 
                              device=torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu'),
                             )
        # Visualization 在图像上可视化关键点并输出
        for pid, point in enumerate(keypoints):
            img = np.array(img)[:, :, ::-1] # RGB to BGR for cv2 modules
            img = draw_points_and_skeleton(img.copy(), point, joints_dict()['openpose']['skeleton'], person_index=pid,
                                            points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                            points_palette_samples=10, confidence_threshold=0.4)
            
            save_name = args.save_dir + '/' + str(img_name) + '_result.' + str(img_type)
            cv2.imwrite(save_name, img)
        print('Image save at "' + save_name + '" successfully!')