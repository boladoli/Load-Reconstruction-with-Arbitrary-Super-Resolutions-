from os import listdir
from os.path import isfile, join
from argparse import Namespace
from model import Proposed, batched_predict, make_coord
import torch
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def reverse(data, max=9.7691, min=0):
    data = data * (max - min) + min
    return data


def display():

    up_scale = 4
    file_path_x = 'E:/new data/PycharmProjects/revised_arbitrary/data/processed_data_week/test_{}/lr'.format(up_scale)
    file_path_y = 'E:/new data/PycharmProjects/revised_arbitrary/data/processed_data_week/test_{}/hr'.format(up_scale)
    onlyfiles_x = [f for f in listdir(file_path_x) if isfile(join(file_path_x, f))]


    args = Namespace()
    args.out_dim = 1
    args.hidden_list = [256, 256, 256, 256]
    h, w = 1, up_scale * 48
    coord = make_coord(w, 1).cuda()

    proposed = Proposed(args, local_regression=True).cuda()
    url = 'E:/new data/PycharmProjects/revised_arbitrary/models/proposed.pth'
    proposed.load_state_dict(torch.load(url))

    for entry in onlyfiles_x:
        fullPath_x = os.path.join(file_path_x, entry)
        fullPath_y = os.path.join(file_path_y, entry)
        x = pd.read_csv(fullPath_x).iloc[:, 1].to_numpy()
        y = pd.read_csv(fullPath_y).iloc[:, 1].to_numpy()

        y = torch.as_tensor(y).float().unsqueeze(0).unsqueeze(0).cuda()
        x = torch.as_tensor(x).float().unsqueeze(0).unsqueeze(0).cuda()

        proposed.eval()

        with torch.no_grad():
            pred = batched_predict(proposed, x, coord, bsize=48)
            pred = reverse(np.array(pred.cpu().squeeze(0).squeeze(0)))

            plt.figure(figsize=(8, 4))
            plt.ylabel('Power [kW]', fontsize=12)
            plt.xlabel('Minute index', fontsize=12)
            plt.plot(y, 'black', label='Actual load', linestyle='-', linewidth=1.5)
            plt.plot(pred, label='Proposed LSR method', linestyle='-', linewidth=1.5, color='crimson')
            plt.legend(loc='best', fontsize=12)
            plt.xticks(size=12)
            plt.yticks(size=12)
            plt.tight_layout()
            plt.grid(color='gray', linestyle='--')
            plt.show()

if __name__ == '__main__':
    display()
