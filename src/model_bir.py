'''Model script where the network architecture is defined for the
polarized light field images. Training script is train_bir.py'''
import numpy as np
import torch
from torch import nn
from torchsummary import summary
from Data import BirefringenceDataset

class BirNetworkDense(nn.Module):
    '''Network that mainly uses a fully connected layer.
    9/7/23: network layers are to be changed. They were copied from BirNetwork1.
    '''
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        conv1_out_chs = 64
        # after the convolutions, the HxW will shrink
        conv1_out_dim = 10 # calc from kernel size
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(128, conv1_out_chs, kernel_size=3),
            nn.ReLU(),
        )
        linear_input_size = conv1_out_dim * conv1_out_dim * conv1_out_chs
        target_size = 4 * 8 * 32 * 32
        combat_conv_size = 4 ** 3
        target_size_expand = 4 * (8+6) * (32+6) * (32+6)
        # target_size_expand = target_size * combat_conv_size
        self.fully_connected = nn.Sequential(
            nn.Linear(linear_input_size, target_size_expand),
            nn.ReLU(),
            # nn.Linear(target_size_expand, target_size_expand),
            # nn.ReLU()
        )
        # convolutions layers within target domain
        self.conv2a = nn.Sequential(
            nn.Conv3d(4, 4, kernel_size=3, padding='valid'),
            nn.ReLU(),
        )
        self.conv2b = nn.Sequential(
            nn.Conv3d(4, 4, kernel_size=3, padding='valid'),
            nn.LeakyReLU(),
        )
        self.conv_final = nn.Conv3d(4, 4, kernel_size=3)

    def forward(self, x):
        batch_size = x.shape[0]
        step1 = self.conv1(x)
        step1 = self.flatten(step1)
        step2 = self.fully_connected(step1)
        step3 = step2.view(batch_size, 4, 8+6, 32+6, 32+6)
        # step3 = step2.view(batch_size, 4, 8, 32, 32)
        step3 = self.conv2a(step3)
        step4 = self.conv2b(step3)
        # add a skip connection
        step3_crop = step3[:, :, 1:-1, 1:-1, 1:-1]
        step5 = self.conv_final(step3_crop + step4)
        output = step5
        # output = step3
        return output

class BirNet(nn.Module):
    '''Network that outputs a volume of size 4x8x11x11.
    target_conv (bool): performs 3D convolutions after the dense layer

    Note that without target_conv, the output is has only one channel.
    '''
    def __init__(self, target_conv=False):
        super().__init__()
        self.flatten = nn.Flatten()
        self.target_conv = target_conv
        hw = 16
        num_chs = 512
        linear_input_size = hw * hw * num_chs
        if target_conv:
            self.expand3D = 6 # calc from conv3D kernel sizes
        else:
            self.expand3D = 0
        tgt_sh_expanded = (1, 8+self.expand3D, 11+self.expand3D, 11+self.expand3D)
        tgt_expand_size = np.prod(tgt_sh_expanded)
        linear_target_size = tgt_expand_size
        self.fully_connected = nn.Sequential(
            nn.Linear(linear_input_size, linear_target_size),
            # nn.ReLU(),
        )
        self.activation = nn.ReLU()
        # convolutions layers within target domain
        self.conv2a = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=3, padding='valid'),
            nn.ReLU(),
        )
        self.conv2b = nn.Sequential(
            nn.Conv3d(4, 4, kernel_size=3, padding='valid'),
            nn.LeakyReLU(),
        )
        self.conv_final = nn.Conv3d(4, 4, kernel_size=3)

    def forward(self, x):
        batch_size = x.shape[0]
        step1 = self.flatten(x)
        step2 = self.fully_connected(step1)
        ex3D = self.expand3D
        step3 = step2.view(batch_size, 1, 8+ex3D, 11+ex3D, 11+ex3D)
        if self.target_conv:
            step3 = self.conv2a(self.activation(step3))
            step4 = self.conv2b(step3)
            # add a skip connection
            step3_crop = step3[:, :, 1:-1, 1:-1, 1:-1]
            step5 = self.conv_final(step3_crop + step4)
            output = step5
        else:
            output = step3
        return output

class BirNetSmallVol(nn.Module):
    '''Network that outputs a volume of size 4x8x11x11.
    9/8/23: network layers similar to BirNetwork1.
    '''
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        conv1_out_chs = 64
        # after the convolutions, the HxW will shrink
        conv1_out_dim = 10 # calc from conv2D kernel sizes
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(128, conv1_out_chs, kernel_size=3),
            nn.ReLU(),
        )
        linear_input_size = conv1_out_dim * conv1_out_dim * conv1_out_chs
        self.expand3D = 6 # calc from conv3D kernel sizes
        tgt_sh_expanded = (4, 8+self.expand3D, 11+self.expand3D, 11+self.expand3D)
        tgt_expand_size = np.prod(tgt_sh_expanded)
        linear_target_size = tgt_expand_size
        self.fully_connected = nn.Sequential(
            nn.Linear(linear_input_size, linear_target_size),
            nn.ReLU(),
            # nn.Linear(target_size_expand, target_size_expand),
            # nn.ReLU()
        )
        # convolutions layers within target domain
        self.conv2a = nn.Sequential(
            nn.Conv3d(4, 4, kernel_size=3, padding='valid'),
            nn.ReLU(),
        )
        self.conv2b = nn.Sequential(
            nn.Conv3d(4, 4, kernel_size=3, padding='valid'),
            nn.LeakyReLU(),
        )
        self.conv_final = nn.Conv3d(4, 4, kernel_size=3)

    def forward(self, x):
        batch_size = x.shape[0]
        step1 = self.conv1(x)
        step1 = self.flatten(step1)
        step2 = self.fully_connected(step1)
        ex3D = self.expand3D
        step3 = step2.view(batch_size, 4, 8+ex3D, 11+ex3D, 11+ex3D)
        step3 = self.conv2a(step3)
        step4 = self.conv2b(step3)
        # add a skip connection
        step3_crop = step3[:, :, 1:-1, 1:-1, 1:-1]
        step5 = self.conv_final(step3_crop + step4)
        output = step5
        # output = step3
        return output

class BirNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        conv1_out_chs = 64
        # after the convolutions, the HxW will shrink
        conv1_out_dim = 4 # calc from kernel size
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(128, conv1_out_chs, kernel_size=5),
            nn.ReLU(),
        )
        linear_input_size = conv1_out_dim * conv1_out_dim * conv1_out_chs
        target_size = 4 * 8 * 32 * 32
        combat_conv_size = 4 ** 3
        target_size_expand = 4 * (8+6) * (32+6) * (32+6)
        # target_size_expand = target_size * combat_conv_size
        self.fully_connected = nn.Sequential(
            nn.Linear(linear_input_size, target_size_expand),
            nn.ReLU(),
            # nn.Linear(target_size_expand, target_size_expand),
            # nn.ReLU()
        )
        # convolutions layers within target domain
        self.conv2a = nn.Sequential(
            nn.Conv3d(4, 4, kernel_size=3, padding='valid'),
            nn.ReLU(),
        )
        self.conv2b = nn.Sequential(
            nn.Conv3d(4, 4, kernel_size=3, padding='valid'),
            nn.ReLU(),
        )
        self.conv_final = nn.Conv3d(4, 4, kernel_size=3)

    def forward(self, x):
        batch_size = x.shape[0]
        step1 = self.conv1(x)
        step1 = self.flatten(step1)
        step2 = self.fully_connected(step1)
        step3 = step2.view(batch_size, 4, 8+6, 32+6, 32+6)
        # step3 = step2.view(batch_size, 4, 8, 32, 32)
        step3 = self.conv2a(step3)
        step4 = self.conv2b(step3)
        # add a skip connection
        step3_crop = step3[:, :, 1:-1, 1:-1, 1:-1]
        step5 = self.conv_final(step3_crop + step4)
        output = step5
        # output = step3
        return output

class BirNetwork1(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        conv1_out_chs = 64
        # after the convolutions, the HxW will shrink
        conv1_out_dim = 10 # calc from kernel size
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(128, conv1_out_chs, kernel_size=3),
            nn.ReLU(),
        )
        linear_input_size = conv1_out_dim * conv1_out_dim * conv1_out_chs
        target_size = 4 * 8 * 32 * 32
        combat_conv_size = 4 ** 3
        target_size_expand = 4 * (8+6) * (32+6) * (32+6)
        # target_size_expand = target_size * combat_conv_size
        self.fully_connected = nn.Sequential(
            nn.Linear(linear_input_size, target_size_expand),
            nn.ReLU(),
            # nn.Linear(target_size_expand, target_size_expand),
            # nn.ReLU()
        )
        # convolutions layers within target domain
        self.conv2a = nn.Sequential(
            nn.Conv3d(4, 4, kernel_size=3, padding='valid'),
            nn.ReLU(),
        )
        self.conv2b = nn.Sequential(
            nn.Conv3d(4, 4, kernel_size=3, padding='valid'),
            nn.LeakyReLU(),
        )
        self.conv_final = nn.Conv3d(4, 4, kernel_size=3)

    def forward(self, x):
        batch_size = x.shape[0]
        step1 = self.conv1(x)
        step1 = self.flatten(step1)
        step2 = self.fully_connected(step1)
        step3 = step2.view(batch_size, 4, 8+6, 32+6, 32+6)
        # step3 = step2.view(batch_size, 4, 8, 32, 32)
        step3 = self.conv2a(step3)
        step4 = self.conv2b(step3)
        # add a skip connection
        step3_crop = step3[:, :, 1:-1, 1:-1, 1:-1]
        step5 = self.conv_final(step3_crop + step4)
        output = step5
        # output = step3
        return output

if __name__ == "__main__":
    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {DEVICE} device")

    # model = BirNetwork().to(device)
    # print(model)
    # print(summary(model, (512, 16, 16)))

    model = BirNet(target_conv=True).to(DEVICE)
    print(model)
    print(summary(model, (512, 16, 16)))

    APPLY_MODEL = False
    if APPLY_MODEL:
        TRAIN_DATA_PATH = "../../../NN_data/small_sphere_random_bir1000/spheres_11by11"
        train_data = BirefringenceDataset(TRAIN_DATA_PATH, split='test',
                                          source_norm=True, target_norm=True)
        X = train_data[0][0].to(DEVICE).to(torch.float32).unsqueeze(dim=0)
        y_pred = model(X)
        print(f"Predicted values: {y_pred}")
