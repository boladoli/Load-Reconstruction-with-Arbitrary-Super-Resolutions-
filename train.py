from model import Proposed, batched_predict
import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader import set_data
import copy
from argparse import Namespace


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


def run(args):

    model = Proposed(args, local_regression=True).cuda()

    batch_size = 64
    learning_rate = 0.0001

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    num_epoch = 100
    loss_fn = nn.L1Loss()

    train_data = set_data(p='train_', scale_min=1, scale_max=4, sample_q=48, repeat=1)
    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True, num_workers=8, pin_memory=True, drop_last=False)

    val_data = set_data(p='validation_', scale_min=1, scale_max=4, sample_q=48, repeat=1)
    val_loader = DataLoader(val_data, batch_size=batch_size,
                            shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    best_mae = 100.0
    best_epoch = 0
    outputs_dir = 'E:/new data/PycharmProjects/revised_arbitrary/models3'
    V_lst = []
    T_lst = []

    print("Training start")
    for epoch in range(0, num_epoch):
        train_loss = Averager()
        model.train()

        for batch in train_loader:
            for k, v in batch.items():
                batch[k] = v.to('cuda', non_blocking=True, dtype=torch.float)
            optimizer.zero_grad()
            pred = model(batch['inp'], batch['coord'])
            loss = loss_fn(pred, batch['gt'])
            train_loss.add(loss.item())
            loss.backward()
            optimizer.step()

        T_lst.append(train_loss.item())
        val_loss = Averager()
        model.eval()
        scheduler.step()

        for vali_batch in val_loader:
            for a, b in vali_batch.items():
                vali_batch[a] = b.to('cuda', non_blocking=True, dtype=torch.float)
            with torch.no_grad():
                v_pred = model(vali_batch['inp'], vali_batch['coord'])
                v_loss = loss_fn(v_pred, vali_batch['gt'])
                val_loss.add(v_loss.item())

        if val_loss.item() < best_mae:
            best_epoch = epoch
            best_mae = val_loss.item()
            weights = copy.deepcopy(model.state_dict())

        V_lst.append(val_loss.item())
        print("Epoch {} training loss {} validation {}".format(epoch, train_loss.item(), val_loss.item()))

    torch.save(weights, os.path.join(outputs_dir, 'proposed.pth'))
    print("best epoch", best_epoch)
    print(V_lst)
    print("training loss")
    print(T_lst)

if __name__ == '__main__':
    ######################
    args = Namespace()
    ###########################
    args.out_dim = 1
    args.hidden_list = [256, 256, 256, 256]
    ###########################
    run(args)


