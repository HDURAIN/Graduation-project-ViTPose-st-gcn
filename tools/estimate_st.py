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
from tools.utils.dist_util import get_dist_info, init_dist
from tools.utils.top_down_eval import keypoints_from_heatmaps

__all__ = ['inference']
            
            
@torch.no_grad()
def estimate(img, img_size: tuple,
              model_cfg: dict, ckpt_path: Path, device: torch.device) -> np.ndarray:
    
    # Prepare model 调用vitmodel中的ViTPose函数
    vit_pose = ViTPose(model_cfg)
    
    # 加载预训练好的模型权重文件
    ckpt = torch.load(ckpt_path)
    if 'state_dict' in ckpt:
        vit_pose.load_state_dict(ckpt['state_dict'])
    else:
        vit_pose.load_state_dict(ckpt)
    vit_pose.to(device)
    print(f">>> Model loaded: {ckpt_path}")
    
    # Prepare input data 'image --> tensor'
    org_w, org_h = img.size
    print(f">>> Original image size: {org_h} X {org_w} (height X width)")
    print(f">>> Resized image size: {img_size[1]} X {img_size[0]} (height X width)")
    print(f">>> Scale change: {org_h/img_size[1]}, {org_w/img_size[0]}")
    img_tensor = transforms.Compose (
        [transforms.Resize((img_size[1], img_size[0])),
         transforms.ToTensor()]
    )(img).unsqueeze(0).to(device)
    
    
    # Feed to model
    tic = time()
    # numpy不能读取CUDA tensor 需要将它转化为 CPU tensor
    heatmaps = vit_pose(img_tensor).detach().cpu().numpy() # N, 17, h/4, w/4
    # 计算所用的时间并打印
    elapsed_time = time()-tic
    print(f">>> Output size: {heatmaps.shape} ---> {elapsed_time:.4f} sec. elapsed [{elapsed_time**-1: .1f} fps]\n")    
    
    # points = heatmap2coords(heatmaps=heatmaps, original_resolution=(org_h, org_w))
    # 关键点预测
    points, prob = keypoints_from_heatmaps(heatmaps=heatmaps, center=np.array([[org_w//2, org_h//2]]), 
                                           scale=np.array([[org_w, org_h]]), 
                                           unbiased=True, use_udp=True)
    points = np.concatenate([points[:, :, ::-1], prob], axis=2)
    # 将vitpose输出的关键点矩阵转化为openpose输出的关键点矩阵
    points = vpToOp(points)
    
    return points
    
def vpToOp(points):
    # 将vitpose输出的关键点矩阵转化为openpose输出的关键点矩阵
    # 计算出vitpose输出中左右肩膀关键点的中间点并插入points矩阵
    point_center = np.array([[[0.0,0.0,0.0]]])
    point_center[0,0,:] = (points[0,5,:] + points[0,6,:])/2
    points = np.insert(points, 17, point_center[0,0,:],axis=1)
    
    # 将关节点顺序重新排序为openpose的输出顺序
    points = points[:, [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3], :]
    
    return points