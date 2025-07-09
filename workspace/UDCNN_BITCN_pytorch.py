import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet_mSEM(nn.Module):
    def __init__(self, F1=64, kernLength=64, D=2, Chans=32, dropout=0.3):
        super().__init__()
        self.F1 = F1
        self.D = D
        self.F2 = 32
        self.Chans = Chans
        self.kernLength = kernLength

        # Block 1
        self.conv1 = nn.Conv2d(1, F1, (kernLength, 1), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        self.depthwise2 = nn.Conv2d(F1, F1, (9, 1), groups=F1, bias=False)
        self.depthwise3 = nn.Conv2d(F1, F1, (10, 1), groups=F1, bias=False)
        self.depthwise4 = nn.Conv2d(F1, F1, (2, 1), groups=F1, bias=False)
        self.depthwise5 = nn.Conv2d(F1, F1, (3, 1), groups=F1, bias=False)
        self.depthwise6 = nn.Conv2d(F1, F1, (2, 1), groups=F1, bias=False)
        self.depthwise7 = nn.Conv2d(F1, F1, (3, 1), groups=F1, bias=False)
        self.depthwise8 = nn.Conv2d(F1, F1, (3, 1), groups=F1, bias=False)

        # Block 2 (Depthwise)
        self.depthwise = nn.Conv2d(F1, F1 * D, (Chans + 7, 1),  groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 8))
        self.drop1 = nn.Dropout(dropout)

        # Block 3 (Separable)
        self.separable = nn.Conv2d(128, 32, (16, 1), padding="same",
                                   bias=False)
        # Pointwise
        self.pointwise = nn.Conv2d(
            32, 32,
            kernel_size=(1, 1),
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(32)
        self.act2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        self.channel1 = [0, 15, 1, 14, 2, 11, 12, 3, 13]  # 9
        self.channel2 = [27, 26, 18, 29, 17, 30, 31, 16, 28, 19]  # 10
        self.channel3 = [4, 9]  # 2
        self.channel4 = [7, 8, 22]  # 3
        self.channel5 = [21, 25]  # 2
        self.channel6 = [5, 6, 24]  # 3
        self.channel7 = [10, 20, 23]  # 3

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)

        device = x.device

        channel1_tensor = torch.tensor(self.channel1, device=device)
        channel2_tensor = torch.tensor(self.channel2, device=device)
        channel3_tensor = torch.tensor(self.channel3, device=device)
        channel4_tensor = torch.tensor(self.channel4, device=device)
        channel5_tensor = torch.tensor(self.channel5, device=device)
        channel6_tensor = torch.tensor(self.channel6, device=device)
        channel7_tensor = torch.tensor(self.channel7, device=device)

        x1 = x.index_select(2, channel1_tensor)
        x1 = self.depthwise2(x1)

        x2 = x.index_select(2, channel2_tensor)
        x2 = self.depthwise3(x2)

        x3 = x.index_select(2, channel3_tensor)
        x3 = self.depthwise4(x3)

        x4 = x.index_select(2, channel4_tensor)
        x4 = self.depthwise5(x4)

        x5 = x.index_select(2, channel5_tensor)
        x5 = self.depthwise6(x5)

        x6 = x.index_select(2, channel6_tensor)
        x6 = self.depthwise7(x6)

        x7 = x.index_select(2, channel7_tensor)
        x7 = self.depthwise8(x7)

        x = torch.cat([x, x1, x2, x3, x4, x5, x6, x7], dim=2)

        # Block 2
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 3
        x = self.separable(x)
        x = self.bn3(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        return x

class TCN_block(nn.Module):
    def __init__(self, input_dimension, depth, kernel_size, filters, dropout, activation='elu'):
        super().__init__()
        self.blocks = nn.ModuleList()
        current_dim = input_dimension

        for i in range(depth):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation

            block = nn.ModuleDict({
                'conv1': nn.Conv1d(current_dim, filters, kernel_size,
                                   padding=0,
                                   dilation=dilation),
                'bn1': nn.BatchNorm1d(filters),
                'act1': nn.ELU() if activation == 'elu' else nn.ReLU(),
                'drop1': nn.Dropout(dropout),

                'conv2': nn.Conv1d(filters, filters, kernel_size,
                                   padding=0,
                                   dilation=dilation),
                'bn2': nn.BatchNorm1d(filters),
                'act2': nn.ELU() if activation == 'elu' else nn.ReLU(),
                'drop2': nn.Dropout(dropout),

                'res_conv': nn.Conv1d(current_dim, filters, 1) if current_dim != filters else None,
            })

            block.padding1 = padding
            block.padding2 = padding
            self.blocks.append(block)
            current_dim = filters

    def forward(self, x):
        for block in self.blocks:
            residual = x

            pad1 = block.padding1
            out = F.pad(x, (pad1, 0))
            out = block['conv1'](out)
            out = block['bn1'](out)
            out = block['act1'](out)
            out = block['drop1'](out)

            pad2 = block.padding2
            out = F.pad(out, (pad2, 0))
            out = block['conv2'](out)
            out = block['bn2'](out)
            out = block['act2'](out)
            out = block['drop2'](out)

            if block['res_conv'] is not None:
                residual = block['res_conv'](residual)

            out += residual
            out = block['act1'](out)
            x = out

        return x

class UDCNN_BiTCN(nn.Module):
    def __init__(self, n_classes, Chans=32, Samples=1500, F1=64, D=2,
                 kernLength=64, dropout_eeg=0.3, layers=2, kernel_s=4,
                 filt=32, dropout=0.3, activation='elu', ):
        super().__init__()

        self.eegnet_sem = EEGNet_mSEM(F1, kernLength, D, Chans, dropout_eeg)

        # TCN部分
        self.tcn1 = TCN_block(input_dimension=32, depth=layers,
                              kernel_size=kernel_s, filters=filt,
                              dropout=dropout, activation=activation)

        self.tcn2 = TCN_block(input_dimension=32, depth=layers,
                              kernel_size=kernel_s, filters=filt,
                              dropout=dropout, activation=activation)

        self.flatten = nn.Flatten()

        self.dense = nn.Linear(64, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.eegnet_sem(x)
        x = x.squeeze(2)
        x_reversed = x.flip(dims=[-1])
        tcn_out = self.tcn1(x)
        out = tcn_out[:, :, -1]
        x_reversed = self.tcn2(x_reversed)
        out2 = x_reversed[:, :, -1]
        x = torch.cat([out, out2], dim=1)
        x = self.flatten(x)
        x = self.dense(x)
        return self.softmax(x)

if __name__ == '__main__':
    from torchinfo import summary
    net = UDCNN_BiTCN(2, 32, 1500)
    summary(net, (1, 1, 32, 1500), device='cpu')








