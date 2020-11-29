# Copyright (C) 2019 Jin Han Lee
#
# This file is a part of BTS.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

import time
import argparse
import datetime
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from tensorboardX import SummaryWriter

import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt
import threading
from tqdm import tqdm

from bts import BtsModel
from bts_dataloader_depth_eval import *

from ndmodel3.nd_mod import NormDiff
from ndmodel3.cwsm import ChannelWiseSoftmax
from paint_utils.paint import paint_multiple, paint_true_depth
from paint_utils.img2pcd import img2pcd


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='BTS PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
parser.add_argument('--model_name',                type=str,   help='model name', default='bts_eigen_v2')
parser.add_argument('--encoder',                   type=str,   help='type of encoder, desenet121_bts, densenet161_bts, '
                                                                    'resnet101_bts, resnet50_bts, resnext50_bts or resnext101_bts',
                                                               default='densenet161_bts')
# Dataset
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='nyu')
parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
parser.add_argument('--gt_path',                   type=str,   help='path to the groundtruth data', required=True)
parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)
parser.add_argument('--input_height',              type=int,   help='input height', default=480)
parser.add_argument('--input_width',               type=int,   help='input width',  default=640)
parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=10)

# Log and save
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a checkpoint to load', default='')
parser.add_argument('--log_freq',                  type=int,   help='Logging frequency in global steps', default=100)
parser.add_argument('--save_freq',                 type=int,   help='Checkpoint saving frequency in global steps', default=500)

# Training
parser.add_argument('--fix_first_conv_blocks',                 help='if set, will fix the first two conv blocks', action='store_true')
parser.add_argument('--fix_first_conv_block',                  help='if set, will fix the first conv block', action='store_true')
parser.add_argument('--bn_no_track_stats',                     help='if set, will not track running stats in batch norm layers', action='store_true')
parser.add_argument('--weight_decay',              type=float, help='weight decay factor for optimization', default=1e-2)
parser.add_argument('--bts_size',                  type=int,   help='initial num_filters in bts', default=512)
parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
parser.add_argument('--adam_eps',                  type=float, help='epsilon in Adam optimizer', default=1e-6)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=4)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--end_learning_rate',         type=float, help='end learning rate', default=-1)
parser.add_argument('--variance_focus',            type=float, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error', default=0.85)

# Preprocessing
parser.add_argument('--do_random_rotate',                      help='if set, will perform random rotation for augmentation', action='store_true')
parser.add_argument('--degree',                    type=float, help='random rotation maximum degree', default=2.5)
parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--use_right',                             help='if set, will randomly use right images when train on KITTI', action='store_true')

# Multi-gpu training
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=1)
parser.add_argument('--world_size',                type=int,   help='number of nodes for distributed training', default=1)
parser.add_argument('--rank',                      type=int,   help='node rank for distributed training', default=0)
parser.add_argument('--dist_url',                  type=str,   help='url used to set up distributed training', default='tcp://127.0.0.1:1234')
parser.add_argument('--dist_backend',              type=str,   help='distributed backend', default='nccl')
parser.add_argument('--gpu',                       type=int,   help='GPU id to use.', default=None)
parser.add_argument('--multiprocessing_distributed',           help='Use multi-processing distributed training to launch '
                                                                    'N processes per node, which has N GPUs. This is the '
                                                                    'fastest way to use PyTorch for either single node or '
                                                                    'multi node data parallel training', action='store_true',)
# Online eval
parser.add_argument('--do_online_eval',                        help='if set, perform online eval in every eval_freq steps', action='store_true')
parser.add_argument('--data_path_eval',            type=str,   help='path to the data for online evaluation', required=False)
parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for online evaluation', required=False)
parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for online evaluation', required=False)
parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=80)
parser.add_argument('--eigen_crop',                            help='if set, crops according to Eigen NIPS14', action='store_true')
parser.add_argument('--garg_crop',                             help='if set, crops according to Garg  ECCV16', action='store_true')
parser.add_argument('--eval_freq',                 type=int,   help='Online evaluation frequency in global steps', default=500)
parser.add_argument('--eval_summary_directory',    type=str,   help='output directory for eval summary,'
                                                                    'if empty outputs to checkpoint folder', default='')

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

if args.mode == 'train' and not args.checkpoint_path:
    from bts import *

elif args.mode == 'train' and args.checkpoint_path:
    model_dir = os.path.dirname(args.checkpoint_path)
    model_name = os.path.basename(model_dir)
    import sys
    sys.path.append(model_dir)
    for key, val in vars(__import__(model_name)).items():
        if key.startswith('__') and key.endswith('__'):
            continue
        vars()[key] = val


inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

eval_metrics = ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3']


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]


def block_print():
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    sys.stdout = sys.__stdout__


def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def colorize(value, vmin=None, vmax=None, cmap='Greys'):
    value = value.cpu().numpy()[:, :, :]
    value = np.log10(value)

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value*0.

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)

    img = value[:, :, :3]

    return img.transpose((2, 0, 1))


def normalize_result(value, vmin=None, vmax=None):
    value = value.cpu().numpy()[0, :, :]

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0.

    return np.expand_dims(value, 0)


def set_misc(model):
    if args.bn_no_track_stats:
        print("Disabling tracking running stats in batch norm layers")
        model.apply(bn_init_as_tf)

    if args.fix_first_conv_blocks:
        if 'resne' in args.encoder:
            fixing_layers = ['base_model.conv1', 'base_model.layer1.0', 'base_model.layer1.1', '.bn']
        else:
            fixing_layers = ['conv0', 'denseblock1.denselayer1', 'denseblock1.denselayer2', 'norm']
        print("Fixing first two conv blocks")
    elif args.fix_first_conv_block:
        if 'resne' in args.encoder:
            fixing_layers = ['base_model.conv1', 'base_model.layer1.0', '.bn']
        else:
            fixing_layers = ['conv0', 'denseblock1.denselayer1', 'norm']
        print("Fixing first conv block")
    else:
        if 'resne' in args.encoder:
            fixing_layers = ['base_model.conv1', '.bn']
        else:
            fixing_layers = ['conv0', 'norm']
        print("Fixing first conv layer")

    for name, child in model.named_children():
        if not 'encoder' in name:
            continue
        for name2, parameters in child.named_parameters():
            # print(name, name2)
            if any(x in name2 for x in fixing_layers):
                parameters.requires_grad = False




def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # Create model
    nd_model = NormDiff(2e-4).cuda()
    cs_model = ChannelWiseSoftmax().cuda()
    
    global_step = 0
    
    
    
    cudnn.benchmark = True

    dataloader = BtsDataLoader(args, 'online_eval')
    


    steps_per_epoch = len(dataloader.data)
    epoch = global_step // steps_per_epoch

    while epoch < args.num_epochs:

        for step, sample_batched in enumerate(dataloader.data):

            image = torch.autograd.Variable(sample_batched['image'].cuda(args.gpu, non_blocking=True))
            focal = torch.autograd.Variable(sample_batched['focal'].cuda(args.gpu, non_blocking=True))
            depth_gt = torch.autograd.Variable(sample_batched['depth'].cuda(args.gpu, non_blocking=True))

            depth_gt = depth_gt.transpose(3,2).transpose(2,1)

            print('depth_gt shape =', depth_gt.shape)

            #lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = model(image, focal)

            if args.dataset == 'nyu':
                mask = depth_gt > 0.1
            else:
                mask = depth_gt > 1.0

            real_bmask = mask & (depth_gt != 0)
            disp_gt = 1 / depth_gt
            disp_gt[~real_bmask] = 0.

            #loss_silog = silog_criterion.forward(depth_est, depth_gt, mask.to(torch.bool))
            
            nd_gt, diff_gt, invd_bmask = nd_model(disp_gt)
            #nd_gt = F.avg_pool2d(nd_gt, kernel_size=5, stride=1, padding=2)
            #nd_gt = F.avg_pool2d(nd_gt, kernel_size=5, stride=1, padding=2)
            #nd_est, diff_est, _ = nd_model(depth_est)
            
            
            paint_multiple(image[0].cpu().detach(), depth_gt[0].cpu().detach(), nd_gt[0].cpu().detach(),
                           nd_gt[0,0:2].cpu().detach(), nd_gt[0,1:3].cpu().detach(), torch.cat((nd_gt[0,0:1].cpu().detach(), nd_gt[0,2:3].cpu().detach()),dim=0),
                           images_per_row=2, to_screen=True)
            
            diff_xy_len = (diff_gt[:,0:1,:,:].cpu().detach() ** 2 + diff_gt[:,1:2,:,:].cpu().detach() ** 2).sqrt()
            diff_xy_len[invd_bmask[:,0:1]] = 0.
            mean_val = diff_xy_len[~invd_bmask[:,0:1]].mean()

            diff_xy_len_list = diff_xy_len.reshape(-1)

            N, _, H, W = diff_xy_len.shape

            print('max = %.8f' % diff_xy_len_list.max())
            print('mean = %.8f' % mean_val)

            #mean_plane = mean_val.reshape(1, 1, 1, 1).expand(N, 1, H, W)

            nd_cls = cs_model(nd_gt, dim=1, scaling=10)
            
            N, _, H, W = nd_cls.shape
            cls_map = torch.zeros(N, 3, H, W, device=nd_cls.device)

            up_thresh = .98


            # all classes tegether
            cls_map[:,0:1][nd_cls[:,0:1] > up_thresh] = 1.
            cls_map[:,1:2][nd_cls[:,0:1] > up_thresh] = 1.
            
            cls_map[:,0:1][nd_cls[:,1:2] > up_thresh] = 1.
            cls_map[:,2:3][nd_cls[:,1:2] > up_thresh] = 1.

            cls_map[:,1:2][nd_cls[:,2:3] > up_thresh] = 1.
            cls_map[:,2:3][nd_cls[:,2:3] > up_thresh] = 1.

            cls_map[:,0:1][nd_cls[:,3:4] > up_thresh] = 1.
            
            cls_map[:,1:2][nd_cls[:,4:5] > up_thresh] = 1.

            # different planes
            cls_l = torch.zeros(N, 3, H, W, device=nd_cls.device)
            cls_r = cls_l.clone()
            cls_d = cls_l.clone()
            cls_u = cls_l.clone()
            cls_b = cls_l.clone()

            cls_r[:,0:1][nd_cls[:,0:1] > up_thresh] = 1. # channel 0 is for 'sure'
            cls_r[:,2:3][nd_cls[:,0:1] < 1-up_thresh] = 1. # channel 2 is for 'surely not'
            # channel 1 is for 'not sure'

            cls_l[:,0:1][nd_cls[:,1:2] > up_thresh] = 1.
            cls_l[:,2:3][nd_cls[:,1:2] < 1-up_thresh] = 1.
            
            cls_u[:,0:1][nd_cls[:,2:3] > up_thresh] = 1. # channel 1
            cls_u[:,2:3][nd_cls[:,2:3] < 1-up_thresh] = 1. # channel 1
            
            cls_d[:,0:1][nd_cls[:,3:4] > up_thresh] = 1. # channel 1
            cls_d[:,2:3][nd_cls[:,3:4] < 1-up_thresh] = 1. # channel 1

            cls_b[:,0:1][nd_cls[:,4:5] > up_thresh] = 1. # channel 1
            cls_b[:,2:3][nd_cls[:,4:5] < 1-up_thresh] = 1. # channel 1

            paint_multiple(nd_gt[0], cls_map[0], cls_l[0], cls_r[0],
                           depth_gt[0], cls_u[0], cls_d[0], cls_b[0],
                           images_per_row=4)
            
            #img2pcd(nd_gt[0,0:1])
            #img2pcd(nd_gt[0,1:2])
            #img2pcd(nd_gt[0,2:3])
            #img2pcd(nd_gt[0,3:4])
            #img2pcd(nd_gt[0,4:5])
            #img2pcd(nd_gt[0,5:6])

            #img2pcd(diff_xy_len[0], mean_plane[0])

            #plt.scatter(range(len(diff_xy_len_list)), diff_xy_len_list)
            #plt.scatter(range(len(diff_xy_len_list)), mean_plane.reshape(-1))
            #plt.show()
            #plt.clf()

            

            global_step += 1

        epoch += 1


def main():
    if args.mode != 'train':
        print('bts_main.py is only for training. Use bts_test.py instead.')
        return -1

    model_filename = args.model_name + '.py'
    command = 'mkdir ' + args.log_directory + '/' + args.model_name
    os.system(command)

    command = 'mkdir ' + args.log_directory + '/' + args.model_name + '/images_save'
    os.system(command)

    args_out_path = args.log_directory + '/' + args.model_name + '/' + sys.argv[1]
    command = 'cp ' + sys.argv[1] + ' ' + args_out_path
    os.system(command)

    if args.checkpoint_path == '':
        model_out_path = args.log_directory + '/' + args.model_name + '/' + model_filename
        command = 'cp bts.py ' + model_out_path
        os.system(command)
        aux_out_path = args.log_directory + '/' + args.model_name + '/.'
        command = 'cp bts_main.py ' + aux_out_path
        os.system(command)
        command = 'cp bts_dataloader.py ' + aux_out_path
        os.system(command)
    else:
        loaded_model_dir = os.path.dirname(args.checkpoint_path)
        loaded_model_name = os.path.basename(loaded_model_dir)
        loaded_model_filename = loaded_model_name + '.py'

        model_out_path = args.log_directory + '/' + args.model_name + '/' + model_filename
        command = 'cp ' + loaded_model_dir + '/' + loaded_model_filename + ' ' + model_out_path
        os.system(command)

    torch.cuda.empty_cache()
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1 and not args.multiprocessing_distributed:
        print("This machine has more than 1 gpu. Please specify --multiprocessing_distributed, or set \'CUDA_VISIBLE_DEVICES=0\'")
        return -1

    if args.do_online_eval:
        print("You have specified --do_online_eval.")
        print("This will evaluate the model every eval_freq {} steps and save best models for individual eval metrics."
              .format(args.eval_freq))

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


if __name__ == '__main__':
    main()
