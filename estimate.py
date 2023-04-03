import argparse
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
from tools.utils.visualization import draw_points_and_skeleton, joints_dict
from tools.utils.dist_util import get_dist_info, init_dist
from tools.utils.top_down_eval import keypoints_from_heatmaps

__all__ = ['inference']
            
            
@torch.no_grad()
def estimate(img_path: Path, img_size: tuple,
              model_cfg: dict, ckpt_path: Path, device: torch.device, save_result: bool=True) -> np.ndarray:
    
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
    img = Image.open(img_path)
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
    print(points)
    points = np.concatenate([points[:, :, ::-1], prob], axis=2)
    
    # Visualization 在图像上可视化关键点并输出
    if save_result:
        for pid, point in enumerate(points):
            img = np.array(img)[:, :, ::-1] # RGB to BGR for cv2 modules
            img = draw_points_and_skeleton(img.copy(), point, joints_dict()['coco']['skeleton'], person_index=pid,
                                           points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                           points_palette_samples=10, confidence_threshold=0.4)
            save_name = img_path.replace(".jpg", "_result.jpg")
            cv2.imwrite(save_name, img)
    
    return points
    

if __name__ == "__main__":
    from config.ViTPose_huge_coco_256x192 import model as model_cfg
    from config.ViTPose_huge_coco_256x192 import data_cfg
    
    # 获取命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', nargs='+', type=str, default='examples/sample.jpg', help='image path(s)')
    parser.add_argument('--ckpt_path', type=str, default='/hy-tmp/train_result/vitpose-h.pth', help='ckpt path(s)')
    args = parser.parse_args()
    
    # 设置预训练模型文件路径
    # CUR_DIR = osp.dirname(__file__)
    # CKPT_PATH = f"{CUR_DIR}/vitpose-b-multi-coco.pth"
    CKPT_PATH = args.ckpt_path
    
    # 可以同时输入多张图像的路径
    img_size = data_cfg['image_size']
    if type(args.image_path) != list:
         args.image_path = [args.image_path]
    for img_path in args.image_path:
        print(img_path)
        # 每张输入的图像调用一次estimate函数 返回为 point
        keypoints = estimate(img_path=img_path, img_size=img_size, model_cfg=model_cfg, ckpt_path=CKPT_PATH, 
                              device=torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu'),
                              save_result=True)