import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace


def make_coord(shape, batch_size=1, ranges=None):
    # If ranges is not provided, set default values for v0 and v1
    if ranges is None:
        v0, v1 = -1, 1
    else:
        # If ranges is provided, assign v0 and v1 accordingly
        v0, v1 = ranges
    # Calculate the range value for each coordinate
    r = (v1 - v0) / (2 * shape)
    # Generate the coordinates based on the shape
    coordinates = v0 + r + (2 * r) * torch.arange(shape).float()
    # Reshape the coordinates tensor to match the batch size
    coordinates = coordinates.unsqueeze(0).repeat(batch_size, 1)
    return coordinates


def gridsample1d_by2d(input, grid, padding_mode, align_corners):
    # Get the shape of the grid
    shape = grid.shape
    # Add a dimension to the input tensor
    input = input.unsqueeze(-1)  # batch_size * C * L_in * 1
    # Add a dimension to the grid tensor
    grid = grid.unsqueeze(1)  # batch_size * 1 * L_out
    # Create a new grid tensor with two channels: [-1, grid]
    grid = torch.stack([-torch.ones_like(grid), grid], dim=-1)
    # Perform grid sampling using the input and grid tensors
    z = F.grid_sample(input, grid, mode=padding_mode, align_corners=align_corners)
    # Get the number of channels in the input tensor
    C = input.shape[1]
    # Define the shape of the output tensor
    out_shape = [shape[0], C, shape[1]]
    # Reshape the output tensor to match the defined shape
    z = z.view(*out_shape)  # batch_size * C * L_out
    return z


def batched_predict(model, inp, coord, bsize):
    # Disable gradient calculation
    with torch.no_grad():
        # Generate features using the model for the input
        model.gen_feat(inp)
        # Get the number of coordinates
        n = coord.shape[-1]
        # Initialize variables for query limits and predictions
        ql = 0
        preds = []
        # Perform batched predictions
        while ql < n:
            qr = min(ql + bsize, n)
            # Query RGB values from the model for the coordinates within the current batch limits
            pred = model.query_rgb(coord[:, ql:qr])
            preds.append(pred)
            ql = qr
        # Concatenate the predictions along the last dimension
        pred = torch.cat(preds, dim=-1)
    return pred


###########################################
# this is codes for decoder

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)


    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)


############################################
# this is codes for EDSR model

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv1d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=True, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm1d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class EDSR(nn.Module):
    def __init__(self, conv=default_conv):
        super(EDSR, self).__init__()

        n_resblocks = 32 #32
        n_feats = 256 #256
        kernel_size = 3
        act = nn.ReLU(True)

        # define head module
        m_head = [conv(1, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=1
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))


        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)


    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x

        return res
##############################################################


class Proposed(nn.Module):

    def __init__(self, args, local_regression=True):
        super().__init__()
        self.args = args
        self.local_regression = local_regression
        self.encoder = EDSR().cuda()
        imnet_in_dim = 256 + 1

        self.imnet = MLP(in_dim=imnet_in_dim, out_dim=args.out_dim, hidden_list=args.hidden_list)
        self.fc1 = nn.Linear(1, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)


    def position_encoder(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        return x

    def gen_feat(self, inp):

        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord):
        feat = self.feat

        if self.local_regression:
            vx_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst = [0]
            eps_shift = 0

        rx = 2 / feat.shape[-1] / 2
        feat_coord = make_coord(feat.shape[-1], feat.shape[0]).cuda().unsqueeze(1)
        preds = []
        dists = []

        for vx in vx_lst:
            coord_ = coord.clone().unsqueeze(1)
            coord_[:, 0, :] += vx * rx + eps_shift
            coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

            coord_ = coord_.squeeze(1)

            q_feat = gridsample1d_by2d(feat, coord_, padding_mode='nearest', align_corners=True)
            q_coord = gridsample1d_by2d(feat_coord, coord_, padding_mode='nearest', align_corners=True)

            rel_coord = coord.unsqueeze(1) - q_coord
            rel_coord[:, 0, :] *= feat.shape[-1]

            tmp = rel_coord.permute(0, 2, 1).contiguous()
            num_maps, width, num_dim = tmp.shape

            rel_coord = self.position_encoder(tmp.view(num_maps * width, -1)).\
                view(num_maps, width, -1).permute(0, 2, 1)

            inp = torch.cat([q_feat, rel_coord], dim=1).permute(0, 2, 1).contiguous()

            bs, q = coord.shape[:2]
            pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1).permute(0, 2, 1)
            preds.append(pred)

            dist = torch.abs(rel_coord[:, 0, :])
            dists.append(dist + 1e-9)

        tot_dist = torch.stack(dists).sum(dim=0)

        if self.local_regression:
            t = dists[0]
            dists[0] = dists[1]
            dists[1] = t
        ret = 0

        for pred, dist in zip (preds, dists):
            ret = ret + pred * (dist/tot_dist).unsqueeze(1)
        return ret

    def forward(self, inp, coord):
        self.gen_feat(inp)
        return self.query_rgb(coord)


if __name__ == '__main__':

    # input with shape [batch_size, 1, len_load_profile]
    # coord with shape [batch_size, len_load_profile]

    x = torch.randn(64, 1, 48).cuda()
    coord = torch.randn(64, 48).cuda()
    args = Namespace()
    ###########################
    args.out_dim = 1
    args.hidden_list = [128, 128, 128]
    net = Proposed(args).cuda()
    out = net(x, coord)
    print(out.shape)

