#!/usr/bin/env python
import os
import sys
import argparse
import json
import shutil
import time

import numpy as np
import torch
import skvideo.io
from PIL import Image
from .io import IO
import tools
import tools.utils as utils

import cv2

class DemoOfflineViT(IO):

    def start(self):
        
        # initiate
        # label_name中加载了kinetics_skeleton数据集中所有的动作分类
        label_name_path = './resource/kinetics_skeleton/label_name.txt'
        with open(label_name_path) as f:
            label_name = f.readlines()
            label_name = [line.rstrip() for line in label_name]
            self.label_name = label_name

        video_cap = cv2.VideoCapture(self.arg.video)
        
        # pose estimation
        video, data_numpy = self.pose_estimation()

        # action recognition
        data = torch.from_numpy(data_numpy)
        data = data.unsqueeze(0)
        data = data.float().to(self.dev).detach()  # (1, channel, frame, joint, person)

        # model predict
        voting_label_name, video_label_name, output, intensity = self.predict(data)

        # render the video
        images = self.render_video(data_numpy, voting_label_name, 
                            video_label_name, intensity, video)
        
        shape_flag = 1
        save_name = self.arg.save_dir + '/' + str(self.video_name) + '_result.avi'
        # visualize
        for image in images:
            image = image.astype(np.uint8)
            if shape_flag == 1:
                videoWrite = cv2.VideoWriter(save_name, 
                                             cv2.VideoWriter_fourcc(*'MJPG'), 
                                             30, (image.shape[1], image.shape[0]), True)# 写入对象
                shape_flag = 0
            videoWrite.write(image)
        videoWrite.release()
        print('Video save at "' + save_name + '" successfully!')

    def predict(self, data):
        # forward
        output, feature = self.model.extract_feature(data)
        output = output[0]
        feature = feature[0]
        intensity = (feature*feature).sum(dim=0)**0.5
        intensity = intensity.cpu().detach().numpy()

        # get result
        # classification result of the full sequence
        voting_label = output.sum(dim=3).sum(
            dim=2).sum(dim=1).argmax(dim=0)
        voting_label_name = self.label_name[voting_label]
        # classification result for each person of the latest frame
        num_person = data.size(4)
        latest_frame_label = [output[:, :, :, m].sum(
            dim=2)[:, -1].argmax(dim=0) for m in range(num_person)]
        latest_frame_label_name = [self.label_name[l]
                                   for l in latest_frame_label]

        num_person = output.size(3)
        num_frame = output.size(1)
        video_label_name = list()
        for t in range(num_frame):
            frame_label_name = list()
            for m in range(num_person):
                person_label = output[:, t, :, m].sum(dim=1).argmax(dim=0)
                person_label_name = self.label_name[person_label]
                frame_label_name.append(person_label_name)
            video_label_name.append(frame_label_name)
        return voting_label_name, video_label_name, output, intensity

    def render_video(self, data_numpy, voting_label_name, video_label_name, intensity, video):
        images = utils.visualization.stgcn_visualize(
            data_numpy,
            self.model.graph.edge,
            intensity, video,
            voting_label_name,
            video_label_name,
            self.arg.height)
        return images

    def pose_estimation(self):
        # load ViTPose module
        from tools.estimate_st import estimate
        ckpt_name = os.path.basename(self.arg.ckpt_path)
        if ckpt_name == "vitpose-l.pth":
            from config.ViTPose_large_coco_256x192 import model as model_cfg
            from config.ViTPose_large_coco_256x192 import data_cfg
        elif ckpt_name == "vitpose-h.pth":
            from config.ViTPose_huge_coco_256x192 import model as model_cfg
            from config.ViTPose_huge_coco_256x192 import data_cfg
        elif ckpt_name == "vitpose-b.pth":
            from config.ViTPose_base_coco_256x192 import model as model_cfg
            from config.ViTPose_base_coco_256x192 import data_cfg

        # 获取路径中最后的的视频文件名称
        self.video_name = self.arg.video.split('/')[-1].split('.')[0]

        # initiate
        video_capture = cv2.VideoCapture(self.arg.video)
        video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        pose_tracker = naive_pose_tracker(data_frame=video_length)

        # pose estimation
        start_time = time.time()
        frame_index = 0
        video = list()
        while(True):

            # get image and resize it 
            ret, orig_image = video_capture.read()
            if orig_image is None:
                break
            source_H, source_W, _ = orig_image.shape
            orig_image = cv2.resize(
                orig_image, (256 * source_W // source_H, 256))
            H, W, _ = orig_image.shape
            video.append(orig_image)

            # pose estimation
            img_size = data_cfg['image_size']
            CKPT_PATH = self.arg.ckpt_path
            orig_image = Image.fromarray(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
            multi_pose = estimate(img=orig_image, img_size=img_size, model_cfg=model_cfg, ckpt_path=CKPT_PATH, 
                                  device=torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu'),
                                  ) # Shape '(num_person, num_joint, 3)'
            # estimate的输出用于st-gcn时x-y坐标位置会互换，这里要先换一次，否则输出的关键点的坐标是错误的
            multi_pose[:,:,[0,1]] = multi_pose[:,:,[1,0]]
            if len(multi_pose.shape) != 3:
                continue

            # normalization
            multi_pose[:, :, 0] = multi_pose[:, :, 0]/W
            multi_pose[:, :, 1] = multi_pose[:, :, 1]/H
            multi_pose[:, :, 0:2] = multi_pose[:, :, 0:2] - 0.5
            multi_pose[:, :, 0][multi_pose[:, :, 2] == 0] = 0
            multi_pose[:, :, 1][multi_pose[:, :, 2] == 0] = 0

            # pose tracking
            pose_tracker.update(multi_pose, frame_index)
            frame_index += 1

            print('Pose estimation ({}/{}).'.format(frame_index, video_length))

        data_numpy = pose_tracker.get_skeleton_sequence()
        return video, data_numpy

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = IO.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Demo for Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        parser.add_argument('--video',
                            default='./resource/media/ta_chi.mp4',
                            help='Path to video')
        parser.add_argument('--openpose',
                            default=None,
                            help='Path to openpose')
        parser.add_argument('--model_input_frame',
                            default=128,
                            type=int)
        parser.add_argument('--model_fps',
                            default=30,
                            type=int)
        parser.add_argument('--height',
                            default=1080,
                            type=int,
                            help='height of frame in the output video.')
        parser.add_argument('--ckpt_path', 
                            type=str, 
                            default='/hy-tmp/models/pretrained_vit/vitpose-b.pth', 
                            help='ckpt path')
        parser.add_argument('--save_dir', 
                            type=str, 
                            default='/hy-tmp/result/video_result_stgcn', 
                            help='save path')
        parser.set_defaults(
            config='./config/st_gcn/kinetics-skeleton/demo_offline.yaml')
        parser.set_defaults(print_log=False)
        # endregion yapf: enable

        return parser

class naive_pose_tracker():
    """ A simple tracker for recording person poses and generating skeleton sequences.
    For actual occasion, I recommend you to implement a robuster tracker.
    Pull-requests are welcomed.
    """

    def __init__(self, data_frame=128, num_joint=18, max_frame_dis=np.inf):
        self.data_frame = data_frame
        self.num_joint = num_joint
        self.max_frame_dis = max_frame_dis
        self.latest_frame = 0
        self.trace_info = list()

    def update(self, multi_pose, current_frame):
        # multi_pose.shape: (num_person, num_joint, 3)

        if current_frame <= self.latest_frame:
            return

        if len(multi_pose.shape) != 3:
            return

        score_order = (-multi_pose[:, :, 2].sum(axis=1)).argsort(axis=0)
        for p in multi_pose[score_order]:

            # match existing traces
            matching_trace = None
            matching_dis = None
            for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
                # trace.shape: (num_frame, num_joint, 3)
                if current_frame <= latest_frame:
                    continue
                mean_dis, is_close = self.get_dis(trace, p)
                if is_close:
                    if matching_trace is None:
                        matching_trace = trace_index
                        matching_dis = mean_dis
                    elif matching_dis > mean_dis:
                        matching_trace = trace_index
                        matching_dis = mean_dis

            # update trace information
            if matching_trace is not None:
                trace, latest_frame = self.trace_info[matching_trace]

                # padding zero if the trace is fractured
                pad_mode = 'interp' if latest_frame == self.latest_frame else 'zero'
                pad = current_frame-latest_frame-1
                new_trace = self.cat_pose(trace, p, pad, pad_mode)
                self.trace_info[matching_trace] = (new_trace, current_frame)

            else:
                new_trace = np.array([p])
                self.trace_info.append((new_trace, current_frame))

        self.latest_frame = current_frame

    def get_skeleton_sequence(self):

        # remove old traces
        valid_trace_index = []
        for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
            if self.latest_frame - latest_frame < self.data_frame:
                valid_trace_index.append(trace_index)
        self.trace_info = [self.trace_info[v] for v in valid_trace_index]

        num_trace = len(self.trace_info)
        if num_trace == 0:
            return None

        data = np.zeros((3, self.data_frame, self.num_joint, num_trace))
        for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
            end = self.data_frame - (self.latest_frame - latest_frame)
            d = trace[-end:]
            beg = end - len(d)
            data[:, beg:end, :, trace_index] = d.transpose((2, 0, 1))

        return data

    # concatenate pose to a trace
    def cat_pose(self, trace, pose, pad, pad_mode):
        # trace.shape: (num_frame, num_joint, 3)
        num_joint = pose.shape[0]
        num_channel = pose.shape[1]
        if pad != 0:
            if pad_mode == 'zero':
                trace = np.concatenate(
                    (trace, np.zeros((pad, num_joint, 3))), 0)
            elif pad_mode == 'interp':
                last_pose = trace[-1]
                coeff = [(p+1)/(pad+1) for p in range(pad)]
                interp_pose = [(1-c)*last_pose + c*pose for c in coeff]
                trace = np.concatenate((trace, interp_pose), 0)
        new_trace = np.concatenate((trace, [pose]), 0)
        return new_trace

    # calculate the distance between a existing trace and the input pose

    def get_dis(self, trace, pose):
        last_pose_xy = trace[-1, :, 0:2]
        curr_pose_xy = pose[:, 0:2]

        mean_dis = ((((last_pose_xy - curr_pose_xy)**2).sum(1))**0.5).mean()
        wh = last_pose_xy.max(0) - last_pose_xy.min(0)
        scale = (wh[0] * wh[1]) ** 0.5 + 0.0001
        is_close = mean_dis < scale * self.max_frame_dis
        return mean_dis, is_close