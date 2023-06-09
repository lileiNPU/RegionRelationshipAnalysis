import numpy as np
import os
gpus = [0]
# 设置CUDA_VISIBLE_DEVICES
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(f) for f in gpus)
import torch
import torch.nn as nn
import torch.optim
from torchvision import transforms
from PIL import Image
import random
from torch.utils.data import Dataset, DataLoader
import math
import scipy.io as scio
import cv2
import sys
sys.path.append('core')
from raft import Basic_timesformer_mse_multiscale_cnn
from utils import flow_viz
from utils.utils import InputPadder
import argparse
import skimage.feature
import skimage.segmentation
import scipy.io as io

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_txtlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.txt')]


class Data_myself(Dataset):

    def __init__(self, listroot=None, labelroot=None, shuffle=True):
        self.listroot = listroot
        self.labelroot = labelroot
        self.transform = transforms.ToTensor()
        listfile_root = self.listroot#os.path.join(self.listroot, 'train_img_label.txt')

        with open(listfile_root, 'r') as file:
            self.lines = file.readlines()
        if shuffle:
            random.shuffle(self.lines)
        # self.nSamples = len(self.lines[:30]) if debug else len(self.lines)
        self.nSamples = len(self.lines)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        assert index <= len(self), 'index range error'
        imgpath_labelpath = self.lines[index].rstrip()
        img_rgb_all, label_texture = self.load_data_label(imgpath_labelpath)
        return (img_rgb_all, label_texture)

    def load_data_label(self, imgpath):
        img_path = imgpath
        f = open(img_path)
        sequence_path = f.read()
        sequence_path = sequence_path.split()
        # img read
        img_rgb_all = []
        list_len = len(sequence_path)
        for i in range(list_len - 1):
            img_name = sequence_path[i]
            img_rgb = cv2.imread(img_name, cv2.IMREAD_COLOR)
            # norm scale
            img_rgb = cv2.resize(img_rgb, (224, 224))
            #img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
            img_rgb = self.transform(img_rgb).float()
            img_rgb = img_rgb.unsqueeze(0)
            if i == 0:
                img_rgb_all = img_rgb
            else:
                img_rgb_all = np.concatenate((img_rgb_all, img_rgb), axis=0)

        label_texture = int(sequence_path[-1])

        return img_rgb_all, label_texture#, label_color

img_transforms = transforms.ToTensor()

# printed attack
############################RGB HSV YCBCR#########################
# set network
parser = argparse.ArgumentParser()
parser.add_argument('--name', default='raft', help="name your experiment")
parser.add_argument('--stage', help="determines which dataset to use for training")
parser.add_argument('--restore_ckpt', help="restore checkpoint")
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--validation', type=str, nargs='+')

parser.add_argument('--lr', type=float, default=0.00002)
parser.add_argument('--frame', type=float, default=5)
parser.add_argument('--num_steps', type=int, default=100000)
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

parser.add_argument('--iters', type=int, default=12)
parser.add_argument('--wdecay', type=float, default=.00005)
parser.add_argument('--epsilon', type=float, default=1e-8)
parser.add_argument('--clip', type=float, default=1.0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
parser.add_argument('--add_noise', action='store_true')
args = parser.parse_args()
model = Basic_timesformer_mse_multiscale_cnn(args)
print("Parameter Count: %d" % count_parameters(model))

# load image and label
batch_size = 1
epoch_num = 10

for epoch in range(epoch_num):

    if (epoch > 0) & (epoch % 1 == 0):

        model_save_path = "./results/model_texture_pulse_overall_basic_timesformer_mse_cnn_rgb"
        model.load_state_dict(torch.load(model_save_path + "/" + "net_epoch_" + str(epoch_num-1) + ".pkl"))

        #net = AttentionNet(img_channel=37)
        # gpu
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        fea_save_path = "./results/feature"
        if not os.path.exists(fea_save_path):
            os.makedirs(fea_save_path)

        ## devel set
        list_path = "./data/TxtList/devel_deep_flow_overall.txt"
        devel_data = Data_myself(listroot=list_path)
        devel_loader = DataLoader(dataset=devel_data, batch_size=batch_size, shuffle=True)
        b_counter = 0
        for batch_L, batch_label_texture in devel_loader:
            b_counter = b_counter + 1
            ################################################################################
            # two classification
            # forward + backward + optimize
            # forward + backward + optimize
            batch_L = batch_L.to(device)

            batch_label_texture = batch_label_texture.type(torch.LongTensor)
            batch_label_texture = batch_label_texture.view(1)
            batch_label_texture = batch_label_texture.detach().numpy()
            batch_label_texture = batch_label_texture[0]

            outputs, img_sub_original, similarity = model(batch_L)
            outputs = outputs.cpu().detach().numpy()
            similarity = similarity.cpu().detach().numpy()

            if b_counter == 1:
                outputs_texture_pulse_mse = outputs.mean()
                outputs_texture_pulse_similarity = similarity
                label = batch_label_texture
            else:
                outputs_texture_pulse_mse = np.vstack((outputs_texture_pulse_mse, outputs.mean()))
                outputs_texture_pulse_similarity = np.vstack((outputs_texture_pulse_similarity, similarity))
                label = np.hstack((label, batch_label_texture))
            print('Devel set:', (epoch, b_counter))
        io.savemat((fea_save_path + "/" + "devel_fc_label_basic_timesformer_mse_multiscale_cnn_rgb_epoch_" + str(epoch) + ".mat"),
                   {"outputs_texture_pulse_mse": outputs_texture_pulse_mse,
                    "outputs_texture_pulse_similarity": outputs_texture_pulse_similarity,
                    "label": label})

        ## test setor
        list_path = "./data/TxtList/test_deep_flow_overall.txt"
        test_data = Data_myself(listroot=list_path)
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
        b_counter = 0
        for batch_L, batch_label_texture in test_loader:
            b_counter = b_counter + 1
            ################################################################################
            # two classification
            # forward + backward + optimize
            batch_L = batch_L.to(device)

            batch_label_texture = batch_label_texture.type(torch.LongTensor)
            batch_label_texture = batch_label_texture.view(1)
            batch_label_texture = batch_label_texture.detach().numpy()
            batch_label_texture = batch_label_texture[0]

            outputs, img_sub_original, similarity = model(batch_L)
            outputs = outputs.cpu().detach().numpy()
            similarity = similarity.cpu().detach().numpy()

            if b_counter == 1:
                outputs_texture_pulse_mse = outputs.mean()
                outputs_texture_pulse_similarity = similarity
                label = batch_label_texture
            else:
                outputs_texture_pulse_mse = np.vstack((outputs_texture_pulse_mse, outputs.mean()))
                outputs_texture_pulse_similarity = np.vstack((outputs_texture_pulse_similarity, similarity))
                label = np.hstack((label, batch_label_texture))
            print('Test set:', (epoch, b_counter))
        io.savemat((fea_save_path + "/" + "test_fc_label_basic_timesformer_mse_multiscale_cnn_rgb_epoch_" + str(epoch) + ".mat"),
                   {"outputs_texture_pulse_mse": outputs_texture_pulse_mse,
                    "outputs_texture_pulse_similarity": outputs_texture_pulse_similarity,
                    "label": label})