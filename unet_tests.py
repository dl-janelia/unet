import numpy as np
import torch
import pytest


class TestDown:
    def __init__(self, down_module):
        self.down_module = down_module

    def test_shape_checker(self) -> None:
        down2 = self.down_module(2)
        msg = "Your `check_valid` function is not right yet."
        assert down2.check_valid((8, 8)), msg
        assert not down2.check_valid((9, 9)), msg
        assert not down2.check_valid((9, 8)), msg
        assert not down2.check_valid((8, 9)), msg
        down3 = self.down_module(3)
        assert down3.check_valid((9, 9)), msg
        assert not down3.check_valid((8, 8)), msg
        assert not down3.check_valid((9, 8)), msg
        assert not down3.check_valid((8, 9)), msg

    def test_shape_checker_error(self) -> None:
        down2 = self.down_module(2)
        with pytest.raises(RuntimeError):
            x = torch.zeros((2, 4, 7, 8))
            down2(x)
        with pytest.raises(RuntimeError):
            x = torch.zeros((2, 4, 8, 7))
            down2(x)
        with pytest.raises(RuntimeError):
            x = torch.zeros((4, 7, 8))
            down2(x)

    def test_shape(self) -> None:
        tensor2 = torch.arange(16).reshape((1, 4, 4))
        down2 = self.down_module(2)
        expected = torch.Tensor([5, 7, 13, 15]).reshape((1, 2, 2))
        msg = "The output shape of your Downsample module is not correct."
        assert expected.shape == down2(tensor2).shape, msg
        msg = "The ouput shape of your Downsample module is correct, but the values are not."
        assert torch.equal(expected, down2(tensor2)), msg

    def run(self):
        self.test_shape_checker()
        self.test_shape_checker_error()
        self.test_shape()
        print("TESTS PASSED")


class TestConvBlock:
    def __init__(self, conv_module):
        self.conv_module = conv_module

    def test_shape_valid(self) -> None:
        shape = [20, 30]
        channels = 4
        out_channels = 5
        kernel_size = 7

        tensor_in = torch.ones([channels, *shape])
        conv = self.conv_module(channels, out_channels, kernel_size, padding="valid")
        tensor_out = conv(tensor_in)

        shape_expected = list(np.array(shape) - 2 * (kernel_size - 1))
        shape_expected = [out_channels, *shape_expected]
        msg = "Output shape for valid padding is incorrect."
        assert tensor_out.shape == torch.Size(shape_expected), msg

    def test_shape_same(self) -> None:
        shape = [16, 39]
        channels = 4
        out_channels = 5
        kernel_size = 7

        tensor_in = torch.ones([channels, *shape])
        conv = self.conv_module(channels, out_channels, kernel_size, padding="same")
        tensor_out = conv(tensor_in)

        shape_expected = [out_channels, *shape]
        msg = "Output shape for same padding is incorrect."
        assert tensor_out.shape == torch.Size(shape_expected), msg

    def test_relu(self) -> None:
        shape = [1, 100, 100]
        tensor_in = torch.randn(shape) * 2

        conv = self.conv_module(1, 50, 5, padding="same")
        tensor_out = conv(tensor_in)
        msg = "Your activation function is incorrect."
        assert torch.all(tensor_out >= 0), msg

    def run(self):
        self.test_shape_valid()
        self.test_shape_same()
        for i in range(5):
            self.test_relu()
        print("TESTS PASSED")


class TestCropAndConcat:
    def __init__(self, ccmodule):
        self.ccmodule = ccmodule

    def test_crop(self) -> None:
        big_tensor = torch.ones((12, 14, 40, 50))
        small_tensor = torch.zeros((12, 5, 13, 18))
        ccmod = self.ccmodule()
        out_tensor = ccmod(big_tensor, small_tensor)
        expected_tensor_1 = torch.cat(
            [torch.ones(12, 14, 13, 18), torch.zeros(12, 5, 13, 18)], dim=1
        )
        expected_tensor_2 = torch.cat(
            [torch.ones(12, 14, 13, 18), torch.zeros(12, 5, 13, 18)], dim=1
        )
        msg = "Your CropAndConcat node does not give the expected output"
        assert torch.equal(out_tensor, expected_tensor_1) or torch.equal(
            out_tensor, expected_tensor_2
        ), msg

    def run(self):
        self.test_crop()
        print("TESTS PASSED")


class TestOutputConv:
    def __init__(self, outconvmodule):
        self.outconvmodule = outconvmodule

    def test_channels(self) -> None:
        outconv = self.outconvmodule(3, 30, activation=torch.nn.Softshrink())
        tensor = torch.ones((3, 24, 17))
        tensor_out = outconv(tensor)
        msg = "The output shape of your output conv is not right."
        assert tensor_out.shape == torch.Size((30, 24, 17)), msg

    def test_activation(self) -> None:
        outconv = self.outconvmodule(3, 30, activation=None)
        tensor = torch.ones((3, 24, 17))
        try:
            outconv(tensor)
        except TypeError as e:
            if str(e) != "'NoneType' object is not callable":
                raise e
            else:
                msg = "Make sure you cover the case when activation is None."
                assert False, msg

    def run(self):
        self.test_channels()
        self.test_activation()
        print("TESTS PASSED")


class TestFmaps:
    """Tests the `compute_fmaps_encoder` / `compute_fmaps_decoder` helpers
    of a UNet stub directly. Use this in Task 6A, before the full UNet is built.
    """

    def __init__(self, unetmodule):
        self.unetmodule = unetmodule

    def test_encoder_fmaps(self) -> None:
        unet = self.unetmodule(5, 1, 1, num_fmaps=17, fmap_inc_factor=4)
        msg = (
            "compute_fmaps_encoder(0) is incorrect. The first encoder block "
            "should map in_channels -> num_fmaps."
        )
        assert unet.compute_fmaps_encoder(0) == (1, 17), msg
        msg = (
            "compute_fmaps_encoder(3) is incorrect. The encoder feature maps "
            "should grow by fmap_inc_factor at every level."
        )
        assert unet.compute_fmaps_encoder(3) == (272, 1088), msg
        msg = (
            "compute_fmaps_encoder(4) (the bottom level) is incorrect. "
            "The same encoder formula should apply at the bottom of the U-Net."
        )
        assert unet.compute_fmaps_encoder(4) == (1088, 4352), msg

    def test_decoder_fmaps(self) -> None:
        unet = self.unetmodule(5, 1, 1, num_fmaps=17, fmap_inc_factor=4)
        msg = (
            "compute_fmaps_decoder(0) is incorrect. Did you remember to add "
            "the skip-connection channels to the input?"
        )
        assert unet.compute_fmaps_decoder(0) == (85, 17), msg
        msg = (
            "compute_fmaps_decoder(3) is incorrect. The decoder input is "
            "(skip connection) + (upsampled output of the level below)."
        )
        assert unet.compute_fmaps_decoder(3) == (5440, 1088), msg

    def run(self):
        self.test_encoder_fmaps()
        self.test_decoder_fmaps()
        print("TESTS PASSED")


class TestUNet:
    def __init__(self, unetmodule):
        self.unetmodule = unetmodule

    def test_fmaps(self) -> None:
        unet = self.unetmodule(5, 1, 1, num_fmaps=17, fmap_inc_factor=4)

        # Check encoder layer 3
        msg = "The number of feature maps in the encoder is incorrect"
        assert unet.left_convs[3].in_channels == 272, msg
        assert unet.left_convs[3].out_channels == 1088, msg

        # Check decoder layer 3
        msg = "The number of feature maps in the decoder is incorrect"
        assert unet.right_convs[3].in_channels == 5440, msg
        assert unet.right_convs[3].out_channels == 1088, msg

        # Check encoder layer 0
        msg = "The number of feature maps in the encoder is incorrect for level 0"
        assert unet.left_convs[0].in_channels == 1, msg
        assert unet.left_convs[0].out_channels == 17, msg

        # Check decoder layer 0
        msg = "The number of feature maps in the decoder is incorrect for level 0"
        assert unet.right_convs[0].in_channels == 85, msg
        assert unet.right_convs[0].out_channels == 17, msg

    def test_shape_valid(self) -> None:
        unetvalid = self.unetmodule(
            depth=4,
            in_channels=2,
            out_channels=7,
            num_fmaps=5,
            fmap_inc_factor=5,
            downsample_factor=3,
            kernel_size=5,
            padding="valid",
        )
        msg = "The output shape of your UNet is incorrect for valid padding."
        assert unetvalid(torch.ones((2, 2, 536, 536))).shape == torch.Size(
            (2, 7, 112, 112)
        ), msg

    def test_shape_valid_3d(self) -> None:
        unetvalid = self.unetmodule(
            depth=3,
            in_channels=2,
            out_channels=1,
            num_fmaps=5,
            fmap_inc_factor=5,
            downsample_factor=3,
            kernel_size=5,
            padding="valid",
            ndim=3,
        )
        msg = "The output shape of your UNet is incorrect for valid padding in 3D."
        assert unetvalid(torch.ones((2, 2, 140, 140, 140))).shape == torch.Size(
            (2, 1, 4, 4, 4)
        ), msg

    def test_shape_valid_3d_tiled(self) -> None:
        unetvalid = self.unetmodule(
            depth=3,
            in_channels=2,
            out_channels=1,
            num_fmaps=5,
            fmap_inc_factor=5,
            downsample_factor=3,
            kernel_size=5,
            padding="valid",
            ndim=3,
        )
        msg = "The output shape of your UNet is incorrect for valid padding in 3D."
        assert unetvalid(torch.ones((2, 2, 158, 158, 158))).shape == torch.Size(
            (2, 1, 18, 18, 18)
        ), msg

    def test_shape_same(self) -> None:
        unetsame = self.unetmodule(
            depth=4,
            in_channels=2,
            out_channels=7,
            num_fmaps=5,
            fmap_inc_factor=5,
            downsample_factor=3,
            kernel_size=5,
            padding="same",
        )
        msg = "The output shape of your Unet is incorrect for same padding."
        assert unetsame(torch.ones((2, 2, 243, 243))).shape == torch.Size(
            (2, 7, 243, 243)
        ), msg

    def test_shape_same_3d(self) -> None:
        unetsame = self.unetmodule(
            depth=3,
            in_channels=2,
            out_channels=1,
            num_fmaps=5,
            fmap_inc_factor=5,
            downsample_factor=3,
            kernel_size=5,
            padding="same",
            ndim=3,
        )
        msg = "The output shape of your Unet is incorrect for same padding in 3D."
        assert unetsame(torch.ones((2, 2, 27, 27, 27))).shape == torch.Size(
            (2, 1, 27, 27, 27)
        ), msg

    def run(self):
        self.test_fmaps()
        self.test_shape_valid()
        self.test_shape_same()
        print("TESTS PASSED")

    def run3d(self):
        self.test_fmaps()
        self.test_shape_valid_3d()
        self.test_shape_same_3d()
        print("TESTS PASSED")

    def run3d_tiled(self):
        self.test_fmaps()
        self.test_shape_same_3d()
        self.test_shape_valid_3d_tiled()
        print("TESTS PASSED")
