import torch  # PyTorch machine learning library
import torch.nn as nn
from torchsummary import summary


class B13Net(nn.Module):
    """BirImg2C1
    A torch model class for the birefringence problem
    Assume the dimension of the input is:
    (D0 = channel, D1 = X_1, D2 =Y_1, D3 = X2, D4 = Y2)
    XY_1 are index of the microlens
    XY_2 are index of the pixel of each microlens
    """

    # General 3d convolutional block
    def _conv3d_block(self, in_channels, out_channels, kernel_size=3, padding=0):
        assert in_channels is not None, "Input is None"
        assert out_channels is not None, "Output is None"
        return nn.Sequential(
            nn.Conv3d(
                in_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            nn.ReLU(),
            nn.Conv3d(
                out_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            nn.ReLU(),
        )

    # General 2d convolutional block
    def _conv2d_block(self, in_channels, out_channels, kernel_size=3, padding=None):
        assert in_channels is not None, "Input is None"
        assert out_channels is not None, "Output is None"
        if padding is None:
            padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            nn.ReLU(),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            nn.ReLU(),
        )

    def complex_input(self, input, mode=None):
        assert input is not None, "Input is None"
        assert mode is not None, "Mode is None"
        
        real_part = input[:, 0] * torch.cos(input[:, 1])
        imag_part = input[:, 0] * torch.sin(input[:, 1])
        
        if mode == "OrignalComplexStack":
            return torch.cat((input, torch.stack((real_part, imag_part), dim=1)), dim=-5)
        elif mode == "ComplexComponents":
            return torch.stack((real_part, imag_part), dim=1)
        elif mode == "ComplexNumbers":
            return torch.view_as_complex(torch.stack((real_part, imag_part), dim=1))
        else:
            raise ValueError("Mode is not supported")


    # Swap the dimension of the input
    def swap_input(self, input, mode=None):
        assert input is not None, "Input is None"
        output = []
        if mode is None:  # return the exact input, (C, 0, 1, 2, 3) is the orignal order
            output.append(input)  # (C, 0, 1, 2, 3)
        elif mode == "swapD1D2":  # return 4 inputs with swaped XY_1 and XY_2
            output.append(input)  # (C, 0, 1, 2, 3)
            output.append(
                torch.permute(input, (0, -5, -3, -4, -2, -1))
            )  # (C, 1, 0, 2, 3)
            output.append(
                torch.permute(input, (0, -5, -2, -1, -3, -4))
            )  # (C, 2, 3, 0, 1)
            output.append(
                torch.permute(input, (0, -5, -1, -2, -3, -4))
            )  # (C, 3, 2, 0, 1)
        elif mode == "swapX1":  # return 4 inputs with each dimension swapped to X_1
            output.append(input)  # (C, 0, 1, 2, 3)
            output.append(input.swapaxes(-3, -4))  # (C, 1, 0, 2, 3)
            output.append(input.swapaxes(-2, -4))  # (C, 2, 1, 0, 3)
            output.append(input.swapaxes(-1, -4))  # (C, 3, 1, 2, 0)
        elif mode == "flatten2D":  # return 2 inputs with 2 dimensions flattened to 2D
            output.append(input.swapaxes(-3, -5))  # (1, 0, C, 2, 3)
            output.append(torch.permute(input, (0, 4, 5, 1, 2, 3)))  # (3, 2, C, 0, 1)
        else:
            raise ValueError("Mode is not supported")
        return output

    # Apply (1, k, k) convolutions,
    def _conv3d_square(self, in_channels, kernel_size=3, scale=2, padding=1):
        if in_channels is None:
            in_channels = self.input_Shape
        return self._conv3d_block(
            in_channels,
            in_channels * scale,
            kernel_size=(1, kernel_size, kernel_size),
            padding=padding,
        )
        # output shape (batch, X_1=channels, Y_1, conv_X_2, conv_Y_2)

    # Apply (n, k, k) convolutions, n is the size of a dimension
    def _conv3d_rod(self, in_channels, in_dim, kernel_size=1, scale=2, padding=1):
        assert in_dim is not None, "Input dimension is None"
        if in_channels is None:
            in_channels = self.input_Shape
        return self._conv3d_block(
            in_channels,
            in_channels * scale,
            kernel_size=(in_dim, kernel_size, kernel_size),
            padding=padding,
        ).squeeze(
            dim=-3
        )  # Y_1 is squeezed
        # output shape (X_1_conv_Y_1=channel, conv_X_2, conv_Y_2)

    # Apply (n, 1, 1) convolutions, n is the size of a dimension, 1 for compressing a dimension
    def _conv3d_axis(self, in_channels, in_dim, scale=2, padding=0):
        if in_channels is None:
            in_channels = self.input_Shape
        return self._conv3d_block(
            in_channels,
            in_channels * scale,
            kernel_size=(in_dim, 1, 1),
            padding=padding,
        ).squeeze(
            dim=-3
        )  # Y_1 is squeezed
        # output shape (X_1_conv_Y_1=channel, X_2, Y_2)

    def __init__(
        self,
        output_size=1,  # number of output channels, either 1, 3, or 4
        input_shape=16,  # size of input dimension, 16 for default
        feat=16,  # number of features in the first layer
        scale=2,  # scale factor for the number of features
        kernel_size=3,  # size of the kernel
        padding=1,  # padding of the kernel
        depth=2,  # depth of the network
        fcldepth=2,  # depth of the fully connected layer
    ):
        super().__init__()
        self.output_size = output_size
        self.input_shape = input_shape
        self.depth = depth
        self.feat = feat
        self.scale = scale
        self.kernel_size = kernel_size
        self.padding = padding
        self.fcldepth = fcldepth


# B13NetV1 uses the (3, 3, 3) convolution
class B13NetV1(B13Net):
    # Apply (k, k, k) convolutions
    def _conv3d_cube(self, in_channels, out_channels, kernel_size=3, padding=1):
        if in_channels is None:
            in_channels = self.input_Shape
        return self._conv3d_block(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        # output shape (X_1=channels, conv_Y_1, conv_X_2, conv_Y_2)

    def __init__(
        self,
        output_size=1,  # number of output channels, either 1, 3, or 4
        input_shape=16,  # size of input dimension, 16 for default
        feat=16,  # number of features in the first layer
        scale=2,  # scale factor for the number of features
        kernel_size=3,  # size of the kernel
        padding=1,  # padding of the kernel
        depth=2,  # depth of the network
        fcldepth=2,  # depth of the fully connected layer
        linear_input_size=None,
    ):
        super().__init__(
            output_size,
            input_shape,
            feat,
            scale,
            kernel_size,
            padding,
            depth,
            fcldepth,
        )
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        assert depth < 5, "Max supported depth is 4"
        assert depth >= 0, "Min supported depth is 0"
        assert fcldepth < 3, "Max supported depth is 2"

        self.output_size = output_size
        self.input_shape = input_shape
        self.depth = depth
        self.feat = feat
        self.scale = scale
        self.kernel_size = kernel_size
        self.padding = padding
        self.fcldepth = fcldepth

        self.encoder = nn.ModuleList(
            [
                self._conv3d_cube(
                    in_channels=input_shape,
                    out_channels=feat * (scale**1),
                    kernel_size=kernel_size,
                    padding=padding,
                ),
                self._conv3d_cube(
                    in_channels=feat * (scale**1),
                    out_channels=feat * (scale**2),
                    kernel_size=kernel_size,
                    padding=padding,
                ),
                self._conv3d_cube(
                    in_channels=feat * (scale**2),
                    out_channels=feat * (scale**3),
                    kernel_size=kernel_size,
                    padding=padding,
                ),
                self._conv3d_cube(
                    in_channels=feat * (scale**3),
                    out_channels=feat * (scale**4),
                    kernel_size=kernel_size,
                    padding=padding,
                ),
            ][:depth]
        )

        featB = feat * 2
        self.encoderB = nn.ModuleList(
            [
                self._conv3d_cube(
                    in_channels=input_shape * 2,
                    out_channels=featB * (scale**1),
                    kernel_size=kernel_size,
                    padding=padding,
                ),
                self._conv3d_cube(
                    in_channels=featB * (scale**1),
                    out_channels=featB * (scale**2),
                    kernel_size=kernel_size,
                    padding=padding,
                ),
                self._conv3d_cube(
                    in_channels=featB * (scale**2),
                    out_channels=featB * (scale**3),
                    kernel_size=kernel_size,
                    padding=padding,
                ),
                self._conv3d_cube(
                    in_channels=featB * (scale**3),
                    out_channels=featB * (scale**4),
                    kernel_size=kernel_size,
                    padding=padding,
                ),
            ][:depth]
        )

        self.flatten = nn.Flatten()

        if linear_input_size is None:
            linear_input_size = 5373952

        self.fully_connected = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 16),
            nn.LeakyReLU(),
            nn.Linear(16, output_size),
        )

    def forward(self, input):
        # input shape (batch, C, X_1, Y_1, X_2, Y_2)
        inputsA = self.swap_input(input, mode="swapD1D2")

        step1C0 = (
            []
        )  # [(X_1, Y_1, X_2, Y_2), (Y_1, X_1, X_2, Y_2), (X_2, Y_2, X_1, Y_1), (Y_2, X_2, X_1, Y_1)]
        step1C1 = []  # (batch, 1, X_1, Y_1, X_2, Y_2)
        setp1CA = []
        for idx, data in enumerate(inputsA):
            step1C0.append(self.encoder[0](data[0]))
            step1C1.append(self.encoder[0](data[1]))
            setp1CA.append(torch.cat((step1C0[idx], step1C1[idx]), dim=-4))

        step2CA = []
        for idx, data in enumerate(setp1CA):
            step2CA.append(self.encoderB[1](data))

        step3CA = self.flatten(torch.stack(step2CA, dim=-4))
        for idx, data in enumerate(step2CA):
            step3CA = torch.cat((step3CA, self.flatten(data)), dim=-1)

        for idx, data in enumerate(setp1CA):
            step3CA = torch.cat((step3CA, self.flatten(data)), dim=-1)

        step3CA = torch.cat((step3CA, self.flatten(input)), dim=-1)

        step4CA = self.fully_connected(step3CA)

        return step4CA


class B13NetV2(B13Net):
    # Apply (f, k, k) convolutions, f should be the channel dimension
    def _conv3d_square2C(
        self, in_channels=None, feature_channels=4, kernel_size=3, scale=2
    ):
        if in_channels is None:
            in_channels = self.input_Shape**2
        padding = (0, kernel_size // 2, kernel_size // 2)
        return nn.Sequential(
            nn.Conv3d(
                in_channels,
                in_channels * scale,
                kernel_size=(1, kernel_size, kernel_size),
                padding=padding,
            ),
            nn.ReLU(),
            nn.Conv3d(
                in_channels * scale,
                in_channels * scale,
                kernel_size=(feature_channels, kernel_size, kernel_size),
                padding=padding,
            ),
            nn.ReLU(),
        )
        # output shape (batch, X_1=channels, conv_Y_1, conv_X_2, conv_Y_2)

    def _resnet_block(
        self, in_channels, out_channels, kernel_size=3, increase_dim=False
    ):
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=2 if increase_dim else 1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def __init__(
        self,
        output_size=1,
        input_shape=16,
        feat=16,
        scale=2,
        kernel_size=3,
        padding=1,
        depth=16,
        block_size = 4,
        fcldepth=2,  # depth of the fully connected layer
        complex_mode=None,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        assert depth < 20, "Max supported depth is 20"
        assert depth >= 4, "Min supported depth is 4"
        assert block_size < 5, "Max supported block size is 4"
        assert fcldepth < 3, "Max supported depth is 2"

        self.output_size = output_size
        self.input_shape = input_shape
        self.feat = feat
        self.scale = scale
        self.kernel_size = kernel_size
        self.padding = padding
        self.depth = depth
        self.block_size = block_size
        self.fcldepth = fcldepth
        self.complex_mode = complex_mode
        
        if complex_mode is None or complex_mode == "ComplexComponents" or complex_mode == "ComplexNumbers":
            self.feature_channels = 2 
        elif complex_mode == "OrignalComplexStack":
            self.feature_channels = 4
        else:
            raise ValueError("Mode is not supported")

        self.flatten2D = nn.Flatten(-5, -4)
        self.flatten1D = nn.Flatten(-1)


        self.initialConv3D = self._conv3d_square2C(  # reduce the channel dimension to 1
            in_channels=input_shape**2,
            feature_channels=self.feature_channels,
            kernel_size=kernel_size,
            scale=scale,
        )

        in_channels = input_shape * input_shape * scale
        num_groups = depth // block_size  # 4 blocks per group

        self.resnet_layers = nn.ModuleList()
        for i in range(num_groups):
            # 4 blocks with the same number of channels
            for _ in range(block_size):
                self.resnet_layers.append(self._resnet_block(in_channels, in_channels))
            # 1 block that shrinks dimensions and doubles channels
            out_channels = in_channels * 2
            self.resnet_layers.append(
                self._resnet_block(in_channels, out_channels, increase_dim=True)
            )
            in_channels = out_channels
            
        self.linear = nn.Sequential(
            nn.Linear(out_channels*2, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 16),
            nn.LeakyReLU(),
            nn.Linear(16, output_size),
        )


    def forward(self, input):
        if self.complex_mode is not None:
            inputsA = self.complex_input(input, mode=self.complex_mode)
        inputsA = self.swap_input(inputsA, mode="flatten2D")

        step1Flatten = []
        for data in inputsA:
            step1Flatten.append(self.flatten2D(data))

        step2Conv3D = []
        for data in step1Flatten:
            step2Conv3D.append(self.initialConv3D(data).squeeze(dim=-3))

        step3Resnet = []
        for data in step2Conv3D:
            for layer in self.resnet_layers:
                residual = data
                data = layer(data)
                if (
                    data.shape == residual.shape
                ):  # Add residual only if the shapes match
                    data += residual
            step3Resnet.append(data.squeeze())
        
        step4Linear = self.linear(torch.cat((step3Resnet[0], step3Resnet[1]), 1))

        return step4Linear
