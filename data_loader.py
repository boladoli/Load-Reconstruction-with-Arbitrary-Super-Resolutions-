from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import random
import os
import torch


def make_coord(shape, ranges=None):
    if ranges is None:
        v0, v1 = -1, 1
    else:
        v0, v1 = ranges
    r = (v1 - v0) / (2 * shape)
    coordinates = v0 + r + (2 * r) * np.arange(shape).astype(float)
    return coordinates


def downscale_load_profile(high_res_load, low_res_length):
    high_res_length = len(high_res_load)
    scaling_factor = high_res_length / low_res_length

    low_res_load = []
    for i in range(low_res_length):
        start_idx = int(i * scaling_factor)
        end_idx = int((i + 1) * scaling_factor)
        avg_value = np.mean(high_res_load[start_idx:end_idx])
        low_res_load.append(avg_value)
    return np.array(low_res_load)


def generate_consecutive_points(N, X):
    start = random.randint(0, X - N)
    points = [start + i for i in range(N)]
    return points


def smooth_load_profile(load_profile, window_size):
    smoothed_profile = np.convolve(load_profile, np.ones(window_size)/window_size, mode='same')
    return smoothed_profile


class set_data(Dataset):

    def __init__(self, p, scale_min=1, scale_max=None, sample_q=None, test=False, repeat=1):
        self.paths = []
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.sample_q = sample_q
        self.test = test
        self.repeat = repeat
        self.p = p

        dir = 'E:/new data/PycharmProjects/revised_arbitrary/data/processed_data_week'
        dir = os.path.join(dir, p)
        paths = os.listdir(dir)
        for elem in paths:
            self.paths.append(os.path.join(dir, elem))
        self.y_size = pd.read_csv(self.paths[0])["grid"].to_numpy().shape[0]


    def __len__(self):
        return len(self.paths) * self.repeat


    def __getitem__(self, idx):
        idx = idx % len(self.paths)
        file_path = self.paths[idx]
        s = random.uniform(self.scale_min, self.scale_max)
        load_data = pd.read_csv(file_path, parse_dates=["localminute"])
        load_data = load_data.set_index('localminute')
        load_data.index = pd.to_datetime(load_data.index, utc=True)
        load_data = load_data.tz_convert('US/Central')

        d = load_data['grid'].to_numpy()
        d = smooth_load_profile(d, 6)
        crop_lr = downscale_load_profile(d, 48)
        w_hr = round(48 * s)
        crop_hr = downscale_load_profile(d, w_hr)

        noise = np.random.normal(loc=0, scale=np.sqrt(0.01), size=crop_lr.shape[0])
        crop_lr = crop_lr + noise

        crop_hr = (crop_hr - 0) / (9.7691 - 0)
        crop_lr = (crop_lr - 0) / (9.7691 - 0)

        crop_lr = torch.as_tensor(crop_lr).float().unsqueeze(0)
        crop_hr = torch.as_tensor(crop_hr).float().unsqueeze(0)

        hr_coord = torch.from_numpy(make_coord(crop_hr.shape[-1])).unsqueeze(0)
        hr_rgb = crop_hr


        if self.test == True:
            return {
                'crop_lr': crop_lr,
                'hr_coord': hr_coord.squeeze(0),
                'crop_hr': crop_hr,
                'ordi_load': d
            }

        if self.sample_q is not None:
            sample_lst = generate_consecutive_points(self.sample_q, hr_coord.shape[1])
            #sample_lst = np.random.choice(hr_coord.shape[1], self.sample_q, replace=False)

        hr_coord = hr_coord.squeeze(0)
        hr_rgb = hr_rgb.squeeze(0)
        hr_coord = hr_coord[sample_lst]
        hr_rgb = hr_rgb[sample_lst].unsqueeze(0)

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'gt': hr_rgb
        }


if __name__ == '__main__':
    b = make_coord(4, ranges=None)
    print(b)
    print(np.arange(4))