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
    def _conv2d_block(self, in_channels, out_channels, kernel_size=3, padding=0):
        assert in_channels is not None, "Input is None"
        assert out_channels is not None, "Output is None"
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

    # Swap the dimension of the input
    def swap_input(self, input, mode=None):
        assert input is not None, "Input is None"
        inputs = []
        if mode is None:  # return the exact input, (C, 0, 1, 2, 3) is the orignal order
            inputs.append(input)  # (C, 0, 1, 2, 3)
        elif mode == "swapD1D2":  # return 2 inputs with swaped XY_1 and XY_2
            inputs.append(input)  # (C, 0, 1, 2, 3)
            inputs.append(input.swapaxes(-3, -4))  # (C, 1, 0, 2, 3)
            inputs.append(input.swapaxes(-2, -4).swapaxes(-2, -4))  # (C, 2, 3, 1, 0)
            inputs.append(input.swapaxes(-3, -4))  # (C, 3, 2, 1, 0)
        elif mode == "swapX1":  # return n inputs with each dimension swapped to X_1
            inputs.append(input)  # (C, 0, 1, 2, 3)
            inputs.append(input.swapaxes(-3, -4))  # (C, 1, 0, 2, 3)
            inputs.append(input.swapaxes(-2, -4))  # (C, 2, 0, 1, 3)
            inputs.append(input.swapaxes(-1, -4))  # (C, 3, 0, 1, 2)
        elif mode == "flattenD1D2":
            inputs.append(input.swapaxes(-3, -5))  # (1, 0, C, 2, 3)
            inputs.append(input.swapaxes(-1, -5).swapaxes(-2, -4))  # (3, 2, C, 0, 1)
        else:
            raise ValueError("Mode is not supported")
        return inputs

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

    # Apply (2, k, k) convolutions, 2 should be the channel dimension
    def _conv3d_square2C(self, in_channels, kernel_size=3, scale=2, padding=1):
        if in_channels is None:
            in_channels = self.input_Shape
        return self._conv3d_block(
            in_channels,
            in_channels * scale,
            kernel_size=(2, kernel_size, kernel_size),
            padding=padding,
        )
        # output shape (batch, X_1=channels, conv_Y_1, conv_X_2, conv_Y_2)

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
        super().__init__()

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

        linear_input_size = 4194304 
        self.fully_connected = nn.Sequential(
            nn.Linear(linear_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, output_size),
        )

    def forward(self, input):
        inputsA = self.swap_input(input, mode="swapD1D2")

        step1C0 = []
        step1C1 = []
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
        step4CA = self.fully_connected(step3CA)

        return step4CA
    
    
   