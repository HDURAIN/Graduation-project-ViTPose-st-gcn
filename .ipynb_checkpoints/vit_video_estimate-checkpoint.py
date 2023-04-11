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

if __name__ == "__main__":
    
    # 获取命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default='resource/media/ta_chi.avi', help='video path')
    parser.add_argument('--save_dir', type=str, default='/hy-tmp/video_result_vit', help='save path')
    parser.add_argument('--ckpt_path', type=str, default='/hy-tmp/pretrained_vit/vitpose-b.pth', help='ckpt path')
    args = parser.parse_args()
    
    # 源实现总是要自己手动改一堆地方太不方便，这里根据输入的ckpt文件名自动导入相关cfg文件
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
    CKPT_PATH = args.ckpt_path
    # 获取路径中最后的的视频文件名称
    video_name = args.video_path.split('/')[-1].split('.')[0]
    video_capture = cv2.VideoCapture(args.video_path)
    video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # pose estimation
    frame_index = 0
    flag = 1
    while(True):
        # get image and resize it 
        ret, orig_image = video_capture.read()
        if orig_image is None:
            videoWrite.release()
            print('Video save at "' + save_name + '" successfully!')
            break
        
        H, W, _ = orig_image.shape
        if flag == 1 :
            save_name = args.save_dir + '/' + str(video_name) + '_result.avi'
            videoWrite = cv2.VideoWriter(save_name, 
                                         cv2.VideoWriter_fourcc(*'MJPG'), 
                                         30, (W, H), True)
            flag = 0
        # pose estimation
        img_size = data_cfg['image_size']
        orig_image = Image.fromarray(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
        keypoints = estimate(img=orig_image, img_size=img_size, model_cfg=model_cfg, ckpt_path=CKPT_PATH, 
                              device=torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu'),
                             )  # Shape '(num_person, num_joint, 3)'
        
        # Visualization 在图像上可视化关键点
        for pid, point in enumerate(keypoints):
            orig_image = np.array(orig_image)[:, :, ::-1] # RGB to BGR for cv2 modules
            orig_image = draw_points_and_skeleton(orig_image.copy(), point, joints_dict()['openpose']['skeleton'], person_index=pid,
                                            points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                            points_palette_samples=10, confidence_threshold=0.4)
        
        videoWrite.write(orig_image)
        frame_index += 1
        print('Pose estimation ({}/{}).'.format(frame_index, video_length))