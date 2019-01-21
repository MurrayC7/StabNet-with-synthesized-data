import argparse
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
import math
from config import *

parser = argparse.ArgumentParser()
parser.add_argument('--input_vdir', default='/home/lazycal/workspace/qudou/frames')
parser.add_argument('--output_vdir', default='')
parser.add_argument('--prefix', default=[6, 12, 18, 24, 30], type=int, nargs='+')
parser.add_argument('--suffix', default=[0, 6, 12, 18, 24], type=int, nargs='+')
parser.add_argument('--num', default=-1, type=int)


def make_dirs(path):
    if not os.path.exists(path): os.makedirs(path)


cvt_train2img = lambda x: ((np.reshape(x, (height, width)) + 0.5) * 255).astype(np.uint8)


def stn(img, theta):
    theta = theta.view(-1, 2, 3)

    grid = F.affine_grid(theta, img.size())
    img = F.grid_sample(img, grid)

    return img


def cvt_theta_mat_bundle(Hs):
    # theta_mat * x = x'
    # ret * scale_mat * x = scale_mat * x'
    # ret = scale_mat * theta_mat * scale_mat^-1
    scale_mat = np.eye(3)
    scale_mat[0, 0] = width / 2.
    scale_mat[0, 2] = width / 2.
    scale_mat[1, 1] = height / 2.
    scale_mat[1, 2] = height / 2.

    Hs = Hs.reshape((grid_h, grid_w, 3, 3))
    from numpy.linalg import inv

    return np.matmul(np.matmul(scale_mat, Hs), inv(scale_mat))


def warpRevBundle2(img, x_map, y_map):
    assert (img.ndim == 3)
    assert (img.shape[-1] == 3)
    rate = 4
    x_map = cv2.resize(cv2.resize(x_map, (int(width / rate), int(height / rate))), (width, height))
    y_map = cv2.resize(cv2.resize(y_map, (int(width / rate), int(height / rate))), (width, height))
    x_map = (x_map + 1) / 2 * width
    y_map = (y_map + 1) / 2 * height
    dst = cv2.remap(img, x_map, y_map, cv2.INTER_LINEAR)
    assert (dst.shape == (height, width, 3))
    return dst


def warpRevBundle(img, Hs):
    assert (img.ndim == 3)
    assert (img.shape[-1] == 3)
    Hs_cvt = cvt_theta_mat_bundle(Hs)

    gh = int(math.floor(height / grid_h))
    gw = int(math.floor(width / grid_w))
    img_ = []
    for i in range(grid_h):
        row_img_ = []
        for j in range(grid_w):
            H = Hs_cvt[i, j, :, :]
            sh = i * gh
            eh = (i + 1) * gh - 1
            sw = j * gw
            ew = (j + 1) * gw - 1
            if (i == grid_h - 1):
                eh = height - 1
            if (j == grid_w - 1):
                ew = width - 1
            temp = cv2.warpPerspective(img, H, dsize=(width, height), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
            row_img_.append(temp[sh:eh + 1, sw:ew + 1, :])
        img_.append(np.concatenate(row_img_, axis=1))
    img = np.concatenate(img_, axis=0)
    assert (img.shape == (height, width, 3))
    return img


def vid2frame(input_video_path, temp_video_path):
    videos = os.listdir(input_video_path)
    videos = filter(lambda x: x.endswith('avi'), videos)
    for each_video in videos:
        print(each_video)

        # get the name of each video, and make the directory to save frames
        each_video_name, _ = each_video.split('.')
        os.mkdir(temp_video_path + '/' + each_video_name)

        each_video_save_full_path = os.path.join(temp_video_path, each_video_name) + '/'

        # get the full path of each video, which will open the video tp extract frames
        each_video_full_path = os.path.join(input_video_path, each_video)

        cap = cv2.VideoCapture(each_video_full_path)
        frame_count = 1
        success = True
        while success:
            success, frame = cap.read()
            print('Read a new frame: ', success)

            params = [cv2.IMWRITE_PXM_BINARY, 1]
            cv2.imwrite(each_video_save_full_path + each_video_name + "_%d.ppm" % frame_count, frame, params)

            frame_count = frame_count + 1

    cap.release()
    return frame_list

# def all_vid2frame(input_video_path):
#     all_frame_list = []
#     for :
#         all_frame_list.append(vid2frame(input_video_path))
#     return all_frame_list


def frame_transform():
    pass


def main():
    pass


if __name__ == '__main__':
    main()
