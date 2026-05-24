# %% tags=["solution", "task"]
# ruff: noqa: F811
# %% [markdown] tags=[]
# # Build Your Own U-Net
#
# <hr style="height:2px;">
#
# In this notebook, we will implement a U-Net architecture. Through this exercise you should gain an understanding of the U-Net architecture in particular as well as learn how to approach the implementation of an architecture in general and familiarize yourself a bit more with the inner workings of pytorch.
#
# The exercise is split into five parts:
#
# In part 1 you will implement the building blocks of the U-Net. That includes the convolutions, downsampling, upsampling and skip connections. We will go in the order of how difficult they are to implement.
#
# In part 2 you will combine the modules you've built in part 1 to implement the U-Net module.
#
# Parts 3 and 4 are light on coding tasks but you will learn about two important concepts: receptive fields and translational equivariance.
#
# Finally, in part 5 you will train your first U-Net of the course! This will just be a first flavor though since you will learn much more about that in the next exercise.
#
#
# Written by Larissa Heinrich, Caroline Malin-Mayor, and Morgan Schwartz, with inspiration from William Patton.

# %% [markdown] tags=[]
# <hr style="height:2px;">
#
# ## The libraries

# %% tags=[]
# %matplotlib inline
import numpy as np
import torch
import subprocess
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import unet_tests
from local import (
    NucleiDataset,
    apply_and_show_random_image,
    plot_receptive_field,
    show_random_dataset_image,
    pad_to_size,
    unnormalize,
)

# %% tags=[]
# make sure gpu is available. Please call a TA if this cell fails
assert torch.cuda.is_available()


# %% [markdown] tags=[]
# ## The Dataset
# For our segmentation exercises, we will be using a nucleus segmentation dataset from [Kaggle 2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018/data). We have downloaded the dataset during setup and we provided a pytorch Dataset called `NucleiDataset` which we will use for training later. In addition to training, we will use these images to visualize the output of the individual building blocks of the U-Net we will be implementing.
# Below, we create a dataset and then visualize a random image.

# %% tags=[]
dataset = NucleiDataset("nuclei_train_data")

# %% tags=[]
show_random_dataset_image(dataset)

# %% [markdown] tags=[]
# Rerun the cell above a few times to see different images.

# %% [markdown] tags=[]
# <hr style="height:2px;">
#
# ## The Components of a U-Net

# %% [markdown] tags=[]
# The [U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) architecture has proven to outperform the other architectures in segmenting biological and medical images. It is also commonly used for other tasks that require the output to be the same resolution as the input, such as style transfer and denoising. Below is an overview figure of the U-Net architecture from the original [paper](https://arxiv.org/pdf/1505.04597.pdf). We will go through each of the components first (hint: all of them can be found in the list of PyTorch modules [here](https://pytorch.org/docs/stable/nn.html#convolution-layers)), and then fit them all together to make our very own U-Net.
#
# <img src="static/unet.png" alt="UNet" style="width: 1500px;"/>

# %% [markdown] tags=[]
# ### Component 1: Upsampling
# %% [markdown] tags=[]
# <img src="./static/up_layer.png" alt="UNet_up_layer" style="width: 1500px;"/>
# %% [markdown] tags=[]
# We will start with the Upsample module that we will use in our U-Net. The right side of the U-Net contains upsampling between the levels. There are many ways to upsample: in the original U-Net, they used a transposed convolution, but this has since fallen a bit out of fashion so we will use the PyTorch Upsample Module [torch.nn.Upsample](https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html#torch.nn.Upsample) instead.


# %% [markdown] tags=[]
# #### Pytorch Modules
# Modules are the building blocks of PyTorch models, and contain lots of magic that makes training models easy. If you aren't familiar with PyTorch modules, take a look at the official documentation [here](https://pytorch.org/docs/stable/notes/modules.html). For our purposes, it is crucial to note how Modules can have submodules defined in the `__init__` function, and how the `forward` function is defined and called.

# %% tags=[]
# Here we make fake input to illustrate the upsampling techniques
# Pytorch expects a batch and channel dimension before the actual data,
# So this simulates a 1D input
sample_1d_input = torch.tensor([[[1, 2, 3, 4]]], dtype=torch.float64)
# And this simulates a 2D input
sample_2d_input = torch.tensor(
    [[[[1, 2], [3, 4]]]],
    dtype=torch.float64,
)
sample_2d_input.shape, sample_2d_input

# %% [markdown] tags=[]
# <div class="alert alert-block alert-info">
#     <h4>Task 1: Try out different upsampling techniques</h4>
#     <p>For our U-net, we will use the built-in PyTorch Upsample Module. Here we will practice declaring and calling an Upsample module with different parameters.</p>
#     <ol>
#         <li>Declare an instance of the pytorch Upsample module with <code style="color: black">scale_factor</code> 2 and mode <code style="color: black">"nearest"</code>.</li>
#         <li>Call the instance of Upsample on the <code style="color: black">sample_2d_input</code> to see what the nearest mode does.</li>
#         <li>Vary the scale factor and mode to see what changes. Check the documentation for possible modes and required input dimensions.</li>
#     </ol>
# </div>

# %% tags=["task"]
# TASK 1.1: initialize  an upsample module
up = ...  # YOUR CODE HERE

# TASK 1.2: apply your upsample module to `sample_2d_input`

# YOUR CODE HERE

# %% tags=["task"]
# TASK 1.3: vary scale factor and mode
# YOUR CODE HERE

# %% tags=["solution"]
# SOLUTION 1.1: initialize  an upsample module
up = torch.nn.Upsample(scale_factor=2, mode="nearest")

# SOLUTION 1.2: apply your upsample module to `sample_2d_input`
up(sample_2d_input)

# %% tags=["solution"]
# SOLUTION 1.3: vary scale factor and mode
up3 = torch.nn.Upsample(scale_factor=3, mode="bilinear")
up3(sample_2d_input)

# %% [markdown] tags=[]
# Here is an additional example on image data.

# %% tags=[]
apply_and_show_random_image(up, dataset)

# %% [markdown] tags=[]
# ### Component 2: Downsampling

# %% [markdown] tags=[]
# <img src="./static/down_layer.png" alt="UNet_down_layer" style="width: 1500px;"/>
# %% [markdown] tags=[]
# Between levels of the U-Net on the left side, there is a downsample step. Traditionally, this is done with a 2x2 max pooling operation. There are other ways to downsample, for example with average pooling, but we will stick with max pooling for this exercise.


# %% tags=[]
sample_2d_input = torch.tensor(np.arange(25, dtype=np.float64).reshape((1, 1, 5, 5)))
sample_2d_input = torch.randint(0, 10, (1, 1, 6, 6))
sample_2d_input

# %% [markdown] tags=[]
# <div class="alert alert-block alert-info">
#     <h4>Task 2A: Try out max pooling</h4>
#         <p>Using the docs for <a href=https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html>torch.nn.MaxPool2d</a>,
#         try initializing the module and applying it to the sample input. Try varying the parameters to understand the effect of <code style="color: black">kernel_size</code> and <code style="color: black">stride</code>.
#         </p>

# %% tags=["task"]
# TASK 2A: Initialize max pooling and apply to sample input
# YOUR CODE HERE

# %% tags=["solution"]
# SOLUTION 2A: Initialize max pooling and apply to sample input
max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
max_pool(sample_2d_input)


# %% [markdown] tags=[]
# <div class="alert alert-block alert-info">
#     <h4>Task 2B: Implement a Downsample Module</h4>
#     <p>This is very similar to the built in MaxPool2d, but additionally has to check if the downsample factor matches in the input size. Note that we provide the forward function for you - in future Modules, you will implement the forward yourself.</p>
#     <ol>
#         <li>Declare the submodules you want to use (in this case, <code style="color: black">torch.nn.MaxPool2d</code> with the correct arguments) in the <code style="color: black">__init__</code> function. In our Downsample Module, we do not want to use padding and the stride should match the input kernel size.</li>
#         <li>Write a function to check if the downsample factor is valid. If the downsample factor does not evenly divide the dimensions of the input to the layer, this function should return False.</li>
#     </ol>
# </div>


# %% tags=["task"]
class Downsample(torch.nn.Module):
    def __init__(self, downsample_factor: int):
        """Initialize a MaxPool2d module with the input downsample factor"""

        super().__init__()

        self.downsample_factor = downsample_factor
        # TASK 2B1: Initialize the maxpool module
        self.down = ...  # YOUR CODE HERE

    def check_valid(self, image_size: tuple[int, int]) -> bool:
        """Check if the downsample factor evenly divides each image dimension.
        Returns `True` for valid image sizes and `False` for invalid image sizes.
        Note: there are multiple ways to do this!
        """
        # TASK 2B2: Check that the image_size is valid to use with the downsample factor
        # YOUR CODE HERE

    def forward(self, x):
        if not self.check_valid(tuple(x.size()[-2:])):
            raise RuntimeError(
                "Can not downsample shape %s with factor %s"
                % (x.size(), self.downsample_factor)
            )

        return self.down(x)


# %% tags=["solution"]
class Downsample(torch.nn.Module):
    def __init__(self, downsample_factor: int):
        """Initialize a MaxPool2d module with the input downsample factor"""

        super().__init__()

        self.downsample_factor = downsample_factor
        # SOLUTION 2B1: Initialize the maxpool module
        self.down = torch.nn.MaxPool2d(downsample_factor)

    def check_valid(self, image_size: tuple[int, int]) -> bool:
        """Check if the downsample factor evenly divides each image dimension.
        Returns `True` for valid image sizes and `False` for invalid image sizes.
        Note: there are multiple ways to do this!
        """
        # SOLUTION 2B2: Check that the image_size is valid to use with the downsample factor
        for dim in image_size:
            if dim % self.downsample_factor != 0:
                return False
        return True

    def forward(self, x):
        if not self.check_valid(tuple(x.size()[-2:])):
            raise RuntimeError(
                "Can not downsample shape %s with factor %s"
                % (x.size(), self.downsample_factor)
            )

        return self.down(x)


# %% [markdown] tags=[]
# We wrote some rudimentary tests for each of the torch modules you are writing. If you get an error from your code or an AssertionError from the test, you should probably have another look at your implementation.

# %% tags=[]
unet_tests.TestDown(Downsample).run()

# %% tags=[]
down = Downsample(4)
apply_and_show_random_image(down, dataset)

# %% [markdown] tags=[]
# ### Component 3: Convolution Block

# %% [markdown] tags=[]
# <img src="./static/conv_layer.png" alt="UNet_conv_layer" style="width: 1500px;"/>
# %% [markdown] tags=[]
# #### Convolution
# A U-Net is a convolutional neural network, which means that the main type of operation is a convolution. Convolutions with defined kernels were covered briefly in the pre-course materials.
#
# <img src="./static/2D_Convolution_Animation.gif" width="400" height="300">


# %% [markdown] tags=[]
# Shown here is a 3x3 kernel being convolved with an input array to get an output array of the same size. For each pixel of the input, the value at that same pixel of the output is computed by multiplying the kernel element-wise with the surrounding 3x3 neighborhood around the input pixel, and then summing the result.


# %% [markdown] tags=[]
# #### Padding
#
# You will notice that at the edges of the input, this animation shows greyed out values that extend past the input. This is known as padding the input. This example uses "same" padding, which means the values at the edges are repeated. The other option we will use in this exercise is "valid" padding, which essentially means no padding. In the case of valid padding, the output will be smaller than the input, as values at the edges of the output will not be computed. "Same" padding can introduce edge artifacts, but "valid" padding reduces the output size at every convolution. Note that the amount of padding (for same) and the amount of size lost (for valid) depends on the size of the kernel - a 3x3 convolution would require padding of 1, a 5x5 convolution would require a padding of 2, and so on.
#
# Additionally, there are different modes of padding that determine what strategy is used to make up the values for padding. In the animation above the mode is re-using values from the border. Even more commonly, the image is simply padded with zeros.


# %% [markdown] tags=[]
# #### ReLU Activation
# The Rectified Linear Unit (ReLU) is a common activation function, which takes the max of a value and 0, shown below. It introduces a non-linearity into the neural network - without a non-linear activation function, a neural network could not learn non-linear functions.
#
# <img src="./static/ReLU.png" width="400" height="300">

# %% [markdown] tags=[]
# <div class="alert alert-block alert-info">
#     <h4>Task 3: Implement a ConvBlock module</h4>
#     <p>The convolution block (ConvBlock) of a standard U-Net has two 3x3 convolutions, each of which is followed by a ReLU activation. Our implementation will handle other sizes of convolutions as well. The first convolution in the block will handle changing the input number of feature maps/channels into the output, and the second convolution will have the same number of feature maps in and out.</p>
#     <ol>
#         <li>Declare the submodules you want to use in the <code style="color: black">__init__</code> function. Because you will always be calling four submodules in sequence (<a href=https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d>torch.nn.Conv2d</a>, <a href=https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU>torch.nn.ReLU</a>, Conv2d, ReLU), you can use <a href=https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html>torch.nn.Sequential</a> to hold the convolutions and ReLUs.</li>
#         <li>Call the modules in the forward function. If you used <code style="color: black">torch.nn.Sequential</code> in step 1, you only need to call the Sequential module, but if not, you can call the Conv2d and ReLU Modules explicitly.</li>
#         <li>Visualize the output and maybe rerun the cell to see how the output changes. Can you explain what you see?</li>
# </ol>
# </div>
#
# If you get stuck, refer back to the <a href=https://pytorch.org/docs/stable/notes/modules.html>Module</a> documentation for hints and examples of how to define a PyTorch Module.


# %% tags=["task"]
class ConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: str = "same",
    ):
        """A convolution block for a U-Net. Contains two convolutions, each followed by a ReLU.

        Args:
            in_channels (int): The number of input channels for this conv block. Depends on
                the layer and side of the U-Net and the hyperparameters.
            out_channels (int): The number of output channels for this conv block. Depends on
                the layer and side of the U-Net and the hyperparameters.
            kernel_size (int): The size of the kernel. A kernel size of N signifies an
                NxN square kernel.
            padding (str): The type of convolution padding to use. Either "same" or "valid".
                Defaults to "same".
        """
        super().__init__()

        if kernel_size % 2 == 0:
            msg = "Only allowing odd kernel sizes."
            raise ValueError(msg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        # TASK 3.1: Initialize your modules and define layers.
        # YOUR CODE HERE

        for _name, layer in self.named_modules():
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

    def forward(self, x):
        # TASK 3.2: Apply the modules you defined to the input x
        ...  # YOUR CODE HERE


# %% tags=["solution"]
class ConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: str = "same",
    ):
        """A convolution block for a U-Net. Contains two convolutions, each followed by a ReLU.

        Args:
            in_channels (int): The number of input channels for this conv block. Depends on
                the layer and side of the U-Net and the hyperparameters.
            out_channels (int): The number of output channels for this conv block. Depends on
                the layer and side of the U-Net and the hyperparameters.
            kernel_size (int): The size of the kernel. A kernel size of N signifies an
                NxN square kernel.
            padding (str): The type of convolution padding to use. Either "same" or "valid".
                Defaults to "same".
        """
        super().__init__()

        if kernel_size % 2 == 0:
            msg = "Only allowing odd kernel sizes."
            raise ValueError(msg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        # SOLUTION 3.1: Initialize your modules and define layers.
        self.conv_pass = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                out_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            torch.nn.ReLU(),
        )

        for _name, layer in self.named_modules():
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

    def forward(self, x):
        # SOLUTION 3.2: Apply the modules you defined to the input x
        return self.conv_pass(x)


# %% [markdown] tags=[]
# #### Test and Visualize Output of ConvBlock
# Try rerunning the visualization a few times. What do you observe? Can you explain it?


# %% tags=[]
unet_tests.TestConvBlock(ConvBlock).run()

# %% tags=[]
torch.manual_seed(26)
conv = ConvBlock(1, 2, 5, "same")
apply_and_show_random_image(conv, dataset)


# %% [markdown] tags=[]
# <div class="alert alert-warning">
#
# <h4>Question: Padding</h4>
# As you saw, the convolution modules in pytorch allow you to directly use the keywords <code style="color: black">"valid"</code> or <code style="color: black">"same"</code> for your padding mode. How would you go about calculating the amount of padding you need based on the kernel size?
#
# If you'd like, you can test your assumption by editing the `ConvBlock` to pass your own calculated value to the `padding` keyword in the conv module and rerun the test
# </div>

# %% [markdown] tags=["solution"]
# <div class="alert alert-block alert-warning">
#
# ```
# if padding == "same": `
#     pad = kernel_size // 2
# elif padding == "valid":
#     pad = 0
# else:
#     raise ValueError("Invalid padding mode")
# ```

# %% [markdown] tags=[]
# ### Component 4: Skip Connections and Concatenation
# %% [markdown] tags=[]
# <img src="./static/concat_layer.png" alt="UNet_concat_layer" style="width: 1500px;"/>
# %% [markdown] tags=[]
# The skip connections between the left and right side of the U-Net are central to successfully obtaining high-resolution output. At each layer, the output of the left conv block is concatenated to the output of the upsample block on the right side from the last layer below. Since upsampling, especially with the "nearest" algorithm, does not actually add high resolution information, the concatenation of the left side conv block output is crucial to generate high resolution segmentations.
#
# If the convolutions in the U-Net are valid, the right side will be smaller than the left side, so the left side output must be cropped before concatenation. We provide a helper function to do this cropping.


# %% [markdown] tags=[]
# <div class="alert alert-block alert-info">
#     <h4>Task 4: Implement a CropAndConcat module</h4>
#     <p>Below, you must implement the <code style="color: black">forward</code> method, including the cropping (using the provided helper function <code style="color: black">center_crop</code>) and the concatenation (using <a href=https://pytorch.org/docs/stable/generated/torch.cat.html#torch.cat>torch.cat</a>).
# </p>
# Hint: Use the <code style="color: black">dim</code> keyword argument of <a href=https://pytorch.org/docs/stable/generated/torch.cat.html#torch.cat>torch.cat</a> to choose along which axis to concatenate the tensors.
# </p>
# Hint: The tensors have the layout (batch, channel, x, y)
# </div>


# %% tags=["task"]
def center_crop(x, target_spatial_shape):
    """Center-crop ``x`` so its spatial dimensions match ``target_spatial_shape``.

    The first two dimensions of ``x`` (batch and channel) are kept as-is; only
    the trailing spatial dimensions are cropped. The crop is centered: any
    excess on each side is removed symmetrically (when the size difference is
    odd, the extra pixel is taken off the trailing side because we use
    floor-division).

    Args:
        x: Input tensor shaped ``(batch, channels, *spatial)``.
        target_spatial_shape: Target sizes for the spatial dimensions only
            (e.g. ``(H, W)`` for 2D or ``(D, H, W)`` for 3D). Every entry must
            be ``<=`` the corresponding spatial size of ``x``.

    Returns:
        A view of ``x`` cropped to ``(batch, channels, *target_spatial_shape)``.
    """

    x_target_size = x.size()[:2] + torch.Size(target_spatial_shape)

    offset = tuple((a - b) // 2 for a, b in zip(x.size(), x_target_size))

    slices = tuple(slice(o, o + s) for o, s in zip(offset, x_target_size))

    return x[slices]


class CropAndConcat(torch.nn.Module):
    def forward(self, encoder_output, upsample_output):
        upsample_spatial_shape = upsample_output.size()[2:]  # noqa: F841
        # TASK 4: Implement the forward function
        ...


# %% tags=["solution"]
def center_crop(x, target_spatial_shape):
    """Center-crop ``x`` so its spatial dimensions match ``target_spatial_shape``.

    The first two dimensions of ``x`` (batch and channel) are kept as-is; only
    the trailing spatial dimensions are cropped. The crop is centered: any
    excess on each side is removed symmetrically (when the size difference is
    odd, the extra pixel is taken off the trailing side because we use
    floor-division).

    Args:
        x: Input tensor shaped ``(batch, channels, *spatial)``.
        target_spatial_shape: Target sizes for the spatial dimensions only
            (e.g. ``(H, W)`` for 2D or ``(D, H, W)`` for 3D). Every entry must
            be ``<=`` the corresponding spatial size of ``x``.

    Returns:
        A view of ``x`` cropped to ``(batch, channels, *target_spatial_shape)``.
    """

    x_target_size = x.size()[:2] + torch.Size(target_spatial_shape)
    offset = tuple((a - b) // 2 for a, b in zip(x.size(), x_target_size))

    slices = tuple(slice(o, o + s) for o, s in zip(offset, x_target_size))

    return x[slices]


class CropAndConcat(torch.nn.Module):
    def forward(self, encoder_output, upsample_output):
        upsample_spatial_shape = upsample_output.size()[2:]
        # SOLUTION 4: Implement the forward function
        encoder_cropped = center_crop(encoder_output, upsample_spatial_shape)

        return torch.cat([encoder_cropped, upsample_output], dim=1)


# %% tags=[]
unet_tests.TestCropAndConcat(CropAndConcat).run()

# %% [markdown] tags=[]
# ### Component 5: Output Block
# %% [markdown] tags=[]
# <img src="./static/final_layer.png" alt="UNet_final_layer" style="width: 1500px;"/>
# %% [markdown] tags=[]
# The final block we need to write for our U-Net is the output convolution block. The exact format of output you want depends on your task, so our U-Net must be flexible enough to handle different numbers of out channels and different final activation functions.


# %% [markdown] tags=[]
# <div class="alert alert-block alert-info">
#     <h4>Task 5: Implement an OutputConv Module</h4>
#     <ol>
#         <li>Define the convolution module in the <code style="color: black">__init__</code> function. You can use a convolution with kernel size 1 to get the appropriate number of output channels. The activation submodule is provided for you.</li>
#         <li>Call the final convolution and activation modules in the <code style="color: black">forward</code> function</li>
#     </ol>
# </div>


# %% tags=["task"]
class OutputConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: torch.nn.Module | None = None,
    ):
        """
        A module that uses a convolution with kernel size 1 to get the appropriate
        number of output channels, and then optionally applies a final activation.

        Args:
            in_channels (int): The number of feature maps that will be input to the
                OutputConv block.
            out_channels (int): The number of channels that you want in the output
            activation (str | None, optional): Accepts the name of any torch activation
                function  (e.g., ``ReLU`` for ``torch.nn.ReLU``) or None for no final
                activation. Defaults to None.
        """
        super().__init__()

        # TASK 5.1: Define the convolution submodule
        # YOUR CODE HERE

        self.activation = activation

    def forward(self, x):
        # TASK 5.2: Implement the forward function
        # YOUR CODE HERE
        ...


# %% tags=["solution"]
class OutputConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: torch.nn.Module | None = None,
    ):
        """
        A module that uses a convolution with kernel size 1 to get the appropriate
        number of output channels, and then optionally applies a final activation.

        Args:
            in_channels (int): The number of feature maps that will be input to the
                OutputConv block.
            out_channels (int): The number of channels that you want in the output
            activation (str | None, optional): Accepts the name of any torch activation
                function  (e.g., ``ReLU`` for ``torch.nn.ReLU``) or None for no final
                activation. Defaults to None.
        """
        super().__init__()

        # SOLUTION 5.1: Define the convolution submodule
        self.final_conv = torch.nn.Conv2d(in_channels, out_channels, 1, padding=0)
        self.activation = activation

    def forward(self, x):
        # SOLUTION 5.2: Implement the forward function
        x = self.final_conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


# %% tags=[]
unet_tests.TestOutputConv(OutputConv).run()

# %% tags=[]
out_conv = OutputConv(in_channels=1, out_channels=1, activation=torch.nn.ReLU())
apply_and_show_random_image(out_conv, dataset)

# %% [markdown] tags=[]
# <div class="alert alert-block alert-success">
#     <h2>Checkpoint 1</h2>
#
# Congratulations! You have implemented most of a U-Net!
# We will go over this portion together and answer any questions soon, but feel free to start on the next section where we put it all together.
#
# </div>
#
# <hr style="height:2px;">

# %% [markdown] tags=[]
# ## Putting the U-Net together
#
# Now we will make a U-Net class that combines all of these components as shown in the image. This image shows a U-Net of depth 4 with specific input channels, feature maps, upsampling, and final activation. Ours will be configurable with regards to depth and other features.
#
# <img src="static/unet.png" alt="UNet" style="width: 1500px;"/>


# %% [markdown] tags=[]
# Building a configurable U-Net all at once is a lot to juggle, so we'll split this into two steps:
# - **Task 6A**: write two small helper methods that compute the number of input/output feature maps at every level of the U-Net.
# - **Task 6B**: use those helpers to assemble the full U-Net (`__init__` + `forward`).

# %% [markdown] tags=[]
# ### Task 6A: Feature map arithmetic
#
# Before we can build the encoder and decoder, we need to figure out how many input and output channels each `ConvBlock` should have at every level. This is straightforward on the encoder side, but a bit trickier on the decoder side, where the input to each `ConvBlock` is the **concatenation** of two tensors: the skip connection from the same encoder level, and the upsampled output of the level below.
#
# As a concrete example, here are the channel counts at every level for `num_fmaps=64`, `fmap_inc_factor=2`, `in_channels=1`, `depth=4`:
#
# ```
# Encoder (left side)                    Decoder (right side)
#   level 0:  in=  1, out= 64              level 0:  in= 64+128=192, out= 64   <- output
#   level 1:  in= 64, out=128              level 1:  in=128+256=384, out=128
#   level 2:  in=128, out=256              level 2:  in=256+512=768, out=256
#   level 3:  in=256, out=512  <- bottom   (no decoder block at the bottom)
# ```
#
# Study the table above. Can you spot the pattern in how the encoder grows, and where the two numbers in each decoder `in` come from?

# %% [markdown] tags=[]
# <div class="alert alert-block alert-info">
#     <h4>Task 6A: Implement the feature-map helpers</h4>
#     <p>Below is a U-Net stub that stores the hyperparameters but does <strong>not</strong> build any submodules yet — that's Task 6B. For now, just implement these two methods:</p>
#     <ol>
#         <li><code style="color: black">compute_fmaps_encoder(level)</code>: return <code style="color: black">(fmaps_in, fmaps_out)</code> for the encoder <code style="color: black">ConvBlock</code> at the given level.</li>
#         <li><code style="color: black">compute_fmaps_decoder(level)</code>: return <code style="color: black">(fmaps_in, fmaps_out)</code> for the decoder <code style="color: black">ConvBlock</code> at the given level. Remember the skip-connection concat — calling <code style="color: black">self.compute_fmaps_encoder</code> is useful here.</li>
#     </ol>
# </div>


# %% tags=["task"]
class UNet(torch.nn.Module):
    def __init__(
        self,
        depth: int,
        in_channels: int,
        out_channels: int = 1,
        final_activation: torch.nn.Module | None = None,
        num_fmaps: int = 64,
        fmap_inc_factor: int = 2,
        downsample_factor: int = 2,
        kernel_size: int = 3,
        padding: str = "same",
        upsample_mode: str = "nearest",
    ):
        """A U-Net stub for Task 6A. The submodules will be built in Task 6B —
        for now this only stores the hyperparameters so the helper methods can use them.

        Args:
            depth:
                The number of levels in the U-Net. 2 is the smallest that really
                makes sense for the U-Net architecture, as a one layer U-Net is
                basically just 2 conv blocks.
            in_channels:
                The number of input channels in your dataset.
            out_channels (optional):
                How many output channels you want. Depends on your task. Defaults to 1.
            final_activation (optional):
                What activation to use in your final output block. Depends on your task.
                Defaults to None.
            num_fmaps (optional):
                The number of feature maps in the first layer. Defaults to 64.
            fmap_inc_factor (optional):
                By how much to multiply the number of feature maps between
                layers. Encoder layer ``l`` will have ``num_fmaps*fmap_inc_factor**l``
                output feature maps. Defaults to 2.
            downsample_factor (optional):
                Factor to use for down- and up-sampling the feature maps between layers.
                Defaults to 2.
            kernel_size (optional):
                Kernel size to use in convolutions on both sides of the UNet.
                Defaults to 3.
            padding (optional):
                How to pad convolutions. Either 'same' or 'valid'. Defaults to "same."
            upsample_mode (optional):
                The upsampling mode to pass to torch.nn.Upsample. Usually "nearest"
                or "bilinear." Defaults to "nearest."
        """
        super().__init__()

        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.final_activation = final_activation
        self.num_fmaps = num_fmaps
        self.fmap_inc_factor = fmap_inc_factor
        self.downsample_factor = downsample_factor
        self.kernel_size = kernel_size
        self.padding = padding
        self.upsample_mode = upsample_mode

    def compute_fmaps_encoder(self, level: int) -> tuple[int, int]:
        """Compute the number of input and output feature maps for
        a conv block at a given level of the UNet encoder (left side).

        Args:
            level (int): The level of the U-Net which we are computing
            the feature maps for. Level 0 is the input level, level 1 is
            the first downsampled layer, and level=depth - 1 is the bottom layer.

        Output (tuple[int, int]): The number of input and output feature maps
            of the encoder convolutional pass in the given level.
        """
        # TASK 6A.1: Implement this function
        ...  # YOUR CODE HERE

    def compute_fmaps_decoder(self, level: int) -> tuple[int, int]:
        """Compute the number of input and output feature maps for a conv block
        at a given level of the UNet decoder (right side). Note:
        The bottom layer (depth - 1) is considered an "encoder" conv pass,
        so this function is only valid up to depth - 2.

        Args:
            level (int): The level of the U-Net which we are computing
            the feature maps for. Level 0 is the input level, level 1 is
            the first downsampled layer, and level=depth - 1 is the bottom layer.

        Output (tuple[int, int]): The number of input and output feature maps
            of the decoder convolutional pass in the given level.
        """
        # TASK 6A.2: Implement this function.
        # Hint: `self.compute_fmaps_encoder` is useful here.
        ...  # YOUR CODE HERE


# %% tags=["solution"]
class UNet(torch.nn.Module):
    def __init__(
        self,
        depth: int,
        in_channels: int,
        out_channels: int = 1,
        final_activation: torch.nn.Module | None = None,
        num_fmaps: int = 64,
        fmap_inc_factor: int = 2,
        downsample_factor: int = 2,
        kernel_size: int = 3,
        padding: str = "same",
        upsample_mode: str = "nearest",
    ):
        """A U-Net stub for Task 6A. The submodules will be built in Task 6B —
        for now this only stores the hyperparameters so the helper methods can use them.

        Args:
            depth:
                The number of levels in the U-Net. 2 is the smallest that really
                makes sense for the U-Net architecture, as a one layer U-Net is
                basically just 2 conv blocks.
            in_channels:
                The number of input channels in your dataset.
            out_channels (optional):
                How many output channels you want. Depends on your task. Defaults to 1.
            final_activation (optional):
                What activation to use in your final output block. Depends on your task.
                Defaults to None.
            num_fmaps (optional):
                The number of feature maps in the first layer. Defaults to 64.
            fmap_inc_factor (optional):
                By how much to multiply the number of feature maps between
                layers. Encoder layer ``l`` will have ``num_fmaps*fmap_inc_factor**l``
                output feature maps. Defaults to 2.
            downsample_factor (optional):
                Factor to use for down- and up-sampling the feature maps between layers.
                Defaults to 2.
            kernel_size (optional):
                Kernel size to use in convolutions on both sides of the UNet.
                Defaults to 3.
            padding (optional):
                How to pad convolutions. Either 'same' or 'valid'. Defaults to "same."
            upsample_mode (optional):
                The upsampling mode to pass to torch.nn.Upsample. Usually "nearest"
                or "bilinear." Defaults to "nearest."
        """
        super().__init__()

        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.final_activation = final_activation
        self.num_fmaps = num_fmaps
        self.fmap_inc_factor = fmap_inc_factor
        self.downsample_factor = downsample_factor
        self.kernel_size = kernel_size
        self.padding = padding
        self.upsample_mode = upsample_mode

    def compute_fmaps_encoder(self, level: int) -> tuple[int, int]:
        """Compute the number of input and output feature maps for
        a conv block at a given level of the UNet encoder (left side).

        Args:
            level (int): The level of the U-Net which we are computing
            the feature maps for. Level 0 is the input level, level 1 is
            the first downsampled layer, and level=depth - 1 is the bottom layer.

        Output (tuple[int, int]): The number of input and output feature maps
            of the encoder convolutional pass in the given level.
        """
        # SOLUTION 6A.1: Implement this function
        if level == 0:
            fmaps_in = self.in_channels
        else:
            fmaps_in = self.num_fmaps * self.fmap_inc_factor ** (level - 1)

        fmaps_out = self.num_fmaps * self.fmap_inc_factor**level
        return fmaps_in, fmaps_out

    def compute_fmaps_decoder(self, level: int) -> tuple[int, int]:
        """Compute the number of input and output feature maps for a conv block
        at a given level of the UNet decoder (right side). Note:
        The bottom layer (depth - 1) is considered an "encoder" conv pass,
        so this function is only valid up to depth - 2.

        Args:
            level (int): The level of the U-Net which we are computing
            the feature maps for. Level 0 is the input level, level 1 is
            the first downsampled layer, and level=depth - 1 is the bottom layer.

        Output (tuple[int, int]): The number of input and output feature maps
            of the decoder convolutional pass in the given level.
        """
        # SOLUTION 6A.2: Implement this function
        fmaps_out = self.num_fmaps * self.fmap_inc_factor ** (level)
        concat_fmaps = self.compute_fmaps_encoder(level)[
            1
        ]  # The channels that come from the skip connection
        fmaps_in = concat_fmaps + self.num_fmaps * self.fmap_inc_factor ** (level + 1)

        return fmaps_in, fmaps_out


# %% tags=[]
unet_tests.TestFmaps(UNet).run()


# %% [markdown] tags=[]
# ### Task 6B: Assemble the U-Net
#
# Now that you can compute the channel counts at every level, you can build the rest of the U-Net. You need to:
# - Build the encoder `ConvBlock`s (one per level, `0` through `depth-1`) into `self.left_convs`
# - Build the decoder `ConvBlock`s (one per level, `0` through `depth-2`) into `self.right_convs`
# - Build the `Downsample`, `Upsample`, `CropAndConcat`, and `OutputConv` modules. **Reuse the classes you already wrote** in Part 1 — you do not need to re-implement them. Use `torch.nn.Upsample` directly for the upsampling module.
# - Implement `forward` to wire everything together
#
# You'll also need to **copy your helper methods from Task 6A** into the placeholders inside the class — `__init__` and `forward` both call them. Make sure your Task 6A tests pass before you copy them over!

# %% [markdown] tags=[]
# <div class="alert alert-block alert-info">
#     <h4>Task 6B: U-Net Implementation</h4>
#     <ol>
#         <li>Copy your <code style="color: black">compute_fmaps_encoder</code> and <code style="color: black">compute_fmaps_decoder</code> implementations from Task 6A into the placeholders below.</li>
#         <li>Fill in <code style="color: black">__init__</code>: build the ConvBlock lists using your helpers, then declare an <code style="color: black">Upsample</code>, <code style="color: black">Downsample</code>, <code style="color: black">CropAndConcat</code>, and <code style="color: black">OutputConv</code> module — reuse the <code style="color: black">Downsample</code>, <code style="color: black">CropAndConcat</code>, and <code style="color: black">OutputConv</code> classes you implemented earlier.</li>
#         <li>Implement the <code style="color: black">forward</code> function, applying the modules you declared above in the proper order. Hint: you'll need to hold on to each encoder output for the skip connections.</li>
#     </ol>
# </div>


# %% tags=["task"]
class UNet(torch.nn.Module):
    def __init__(
        self,
        depth: int,
        in_channels: int,
        out_channels: int = 1,
        final_activation: torch.nn.Module | None = None,
        num_fmaps: int = 64,
        fmap_inc_factor: int = 2,
        downsample_factor: int = 2,
        kernel_size: int = 3,
        padding: str = "same",
        upsample_mode: str = "nearest",
    ):
        """A U-Net for 2D input that expects tensors shaped like::
            ``(batch, channels, height, width)``.
        Args:
            depth:
                The number of levels in the U-Net. 2 is the smallest that really
                makes sense for the U-Net architecture, as a one layer U-Net is
                basically just 2 conv blocks.
            in_channels:
                The number of input channels in your dataset.
            out_channels (optional):
                How many output channels you want. Depends on your task. Defaults to 1.
            final_activation (optional):
                What activation to use in your final output block. Depends on your task.
                Defaults to None.
            num_fmaps (optional):
                The number of feature maps in the first layer. Defaults to 64.
            fmap_inc_factor (optional):
                By how much to multiply the number of feature maps between
                layers. Encoder layer ``l`` will have ``num_fmaps*fmap_inc_factor**l``
                output feature maps. Defaults to 2.
            downsample_factor (optional):
                Factor to use for down- and up-sampling the feature maps between layers.
                Defaults to 2.
            kernel_size (optional):
                Kernel size to use in convolutions on both sides of the UNet.
                Defaults to 3.
            padding (optional):
                How to pad convolutions. Either 'same' or 'valid'. Defaults to "same."
            upsample_mode (optional):
                The upsampling mode to pass to torch.nn.Upsample. Usually "nearest"
                or "bilinear." Defaults to "nearest."
        """

        super().__init__()

        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.final_activation = final_activation
        self.num_fmaps = num_fmaps
        self.fmap_inc_factor = fmap_inc_factor
        self.downsample_factor = downsample_factor
        self.kernel_size = kernel_size
        self.padding = padding
        self.upsample_mode = upsample_mode

        # left convolutional passes
        self.left_convs = torch.nn.ModuleList()
        # TASK 6B.1A: Build encoder ConvBlocks
        # Loop through each level of the encoder from top (level=0) to bottom (level=self.depth - 1)
        for level in range(self.depth):
            fmaps_in, fmaps_out = self.compute_fmaps_encoder(level)
            conv = ...  # YOUR CODE HERE
            # Adding conv module to the list
            self.left_convs.append(conv)

        # right convolutional passes
        self.right_convs = torch.nn.ModuleList()
        # TASK 6B.1B: Build decoder ConvBlocks
        # Loop through each level of the decoder from top (level=0) to one above bottom (level=self.depth - 2)
        for level in range(self.depth - 1):
            fmaps_in, fmaps_out = self.compute_fmaps_decoder(level)
            conv = ...  # YOUR CODE HERE
            # Adding conv module to the list
            self.right_convs.append(conv)

        # TASK 6B.2: Initialize the Downsample, Upsample, CropAndConcat and OutputConv modules
        # YOUR CODE HERE

    def compute_fmaps_encoder(self, level: int) -> tuple[int, int]:
        """Compute the number of input and output feature maps for
        a conv block at a given level of the UNet encoder (left side).
        """
        # TASK 6B.0A: Copy your implementation from Task 6A
        ...  # YOUR CODE HERE

    def compute_fmaps_decoder(self, level: int) -> tuple[int, int]:
        """Compute the number of input and output feature maps for a conv block
        at a given level of the UNet decoder (right side).
        """
        # TASK 6B.0B: Copy your implementation from Task 6A
        ...  # YOUR CODE HERE

    def forward(self, x):
        # left side
        # Hint - you will need the outputs of each convolutional block in the encoder for the skip connection, so you need to hold on to those output tensors
        for i in range(self.depth - 1):
            # TASK 6B.3A: Implement encoder here
            ...

        # bottom
        # TASK 6B.3B: Implement bottom of U-Net here

        # right
        for i in range(0, self.depth - 1)[::-1]:
            # TASK 6B.3C: Implement decoder here
            ...
        # TASK 6B.3D: Apply the final convolution and return the output
        return


# %% tags=["solution"]
class UNet(torch.nn.Module):
    def __init__(
        self,
        depth: int,
        in_channels: int,
        out_channels: int = 1,
        final_activation: torch.nn.Module | None = None,
        num_fmaps: int = 64,
        fmap_inc_factor: int = 2,
        downsample_factor: int = 2,
        kernel_size: int = 3,
        padding: str = "same",
        upsample_mode: str = "nearest",
    ):
        """A U-Net for 2D input that expects tensors shaped like::
            ``(batch, channels, height, width)``.
        Args:
            depth:
                The number of levels in the U-Net. 2 is the smallest that really
                makes sense for the U-Net architecture, as a one level U-Net is
                basically just 2 conv blocks.
            in_channels:
                The number of input channels in your dataset.
            out_channels (optional):
                How many output channels you want. Depends on your task. Defaults to 1.
            final_activation (optional):
                What activation to use in your final output block. Depends on your task.
                Defaults to None.
            num_fmaps (optional):
                The number of feature maps in the first layer. Defaults to 64.
            fmap_inc_factor (optional):
                By how much to multiply the number of feature maps between
                layers. Encoder layer ``l`` will have ``num_fmaps*fmap_inc_factor**l``
                output feature maps. Defaults to 2.
            downsample_factor (optional):
                Factor to use for down- and up-sampling the feature maps between layers.
                Defaults to 2.
            kernel_size (optional):
                Kernel size to use in convolutions on both sides of the UNet.
                Defaults to 3.
            padding (optional):
                How to pad convolutions. Either 'same' or 'valid'. Defaults to "same."
            upsample_mode (optional):
                The upsampling mode to pass to torch.nn.Upsample. Usually "nearest"
                or "bilinear." Defaults to "nearest."
        """

        super().__init__()

        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.final_activation = final_activation
        self.num_fmaps = num_fmaps
        self.fmap_inc_factor = fmap_inc_factor
        self.downsample_factor = downsample_factor
        self.kernel_size = kernel_size
        self.padding = padding
        self.upsample_mode = upsample_mode

        # left convolutional passes
        self.left_convs = torch.nn.ModuleList()
        # SOLUTION 6B.1A: Build encoder ConvBlocks
        # Loop through each level of the encoder from top (level=0) to bottom (level=self.depth - 1)
        for level in range(self.depth):
            fmaps_in, fmaps_out = self.compute_fmaps_encoder(level)
            conv = ConvBlock(fmaps_in, fmaps_out, self.kernel_size, self.padding)
            # Adding conv module to the list
            self.left_convs.append(conv)

        # right convolutional passes
        self.right_convs = torch.nn.ModuleList()
        # SOLUTION 6B.1B: Build decoder ConvBlocks
        # Loop through each level of the decoder from top (level=0) to one above bottom (level=self.depth - 2)
        for level in range(self.depth - 1):
            fmaps_in, fmaps_out = self.compute_fmaps_decoder(level)
            # Initialize conv module
            conv = ConvBlock(
                fmaps_in,
                fmaps_out,
                self.kernel_size,
                self.padding,
            )
            # Adding conv module to the list
            self.right_convs.append(conv)

        # SOLUTION 6B.2: Initialize other modules here
        self.downsample = Downsample(self.downsample_factor)
        self.upsample = torch.nn.Upsample(
            scale_factor=self.downsample_factor,
            mode=self.upsample_mode,
        )
        self.crop_and_concat = CropAndConcat()
        self.final_conv = OutputConv(
            self.compute_fmaps_decoder(0)[1], self.out_channels, self.final_activation
        )

    def compute_fmaps_encoder(self, level: int) -> tuple[int, int]:
        """Compute the number of input and output feature maps for
        a conv block at a given level of the UNet encoder (left side).

        Args:
            level (int): The level of the U-Net which we are computing
            the feature maps for. Level 0 is the input level, level 1 is
            the first downsampled layer, and level=depth - 1 is the bottom layer.

        Output (tuple[int, int]): The number of input and output feature maps
            of the encoder convolutional pass in the given level.
        """
        # From Task 6A
        if level == 0:
            fmaps_in = self.in_channels
        else:
            fmaps_in = self.num_fmaps * self.fmap_inc_factor ** (level - 1)

        fmaps_out = self.num_fmaps * self.fmap_inc_factor**level
        return fmaps_in, fmaps_out

    def compute_fmaps_decoder(self, level: int) -> tuple[int, int]:
        """Compute the number of input and output feature maps for a conv block
        at a given level of the UNet decoder (right side). Note:
        The bottom layer (depth - 1) is considered an "encoder" conv pass,
        so this function is only valid up to depth - 2.

        Args:
            level (int): The level of the U-Net which we are computing
            the feature maps for. Level 0 is the input level, level 1 is
            the first downsampled layer, and level=depth - 1 is the bottom layer.

        Output (tuple[int, int]): The number of input and output feature maps
            of the decoder convolutional pass in the given level.
        """
        # From Task 6A
        fmaps_out = self.num_fmaps * self.fmap_inc_factor ** (level)
        concat_fmaps = self.compute_fmaps_encoder(level)[
            1
        ]  # The channels that come from the skip connection
        fmaps_in = concat_fmaps + self.num_fmaps * self.fmap_inc_factor ** (level + 1)

        return fmaps_in, fmaps_out

    def forward(self, x):
        # left side
        convolution_outputs = []
        layer_input = x
        for i in range(self.depth - 1):
            # SOLUTION 6B.3A: Implement encoder here
            conv_out = self.left_convs[i](layer_input)
            convolution_outputs.append(conv_out)
            downsampled = self.downsample(conv_out)
            layer_input = downsampled

        # bottom
        # SOLUTION 6B.3B: Implement bottom of U-Net here
        conv_out = self.left_convs[-1](layer_input)
        layer_input = conv_out

        # right
        for i in range(0, self.depth - 1)[::-1]:
            # SOLUTION 6B.3C: Implement decoder here
            upsampled = self.upsample(layer_input)
            concat = self.crop_and_concat(convolution_outputs[i], upsampled)
            conv_output = self.right_convs[i](concat)
            layer_input = conv_output

        # SOLUTION 6B.3D: Apply the final convolution and return the output
        return self.final_conv(layer_input)


# %% [markdown] tags=[]
# Below we declare a very simple U-Net and then apply it to a random image. Because we have not trained the U-Net the output should look similar to the output of random convolutions. If you get errors here, go back and fix your U-Net implementation!

# %% tags=[]
unet_tests.TestUNet(UNet).run()

# %% tags=[]
simple_net = UNet(depth=2, in_channels=1)

# %% tags=[]
apply_and_show_random_image(simple_net, dataset)

# %% [markdown] tags=[]
# <div class="alert alert-block alert-success">
#     <h2>Checkpoint 2</h2>
#
# Congratulations! You have implemented a UNet architecture.
#
# Next we'll learn about receptive fields which should demistify how to choose the UNet hyperparameters a little bit.
#

# %% [markdown] tags=[]
# <hr style="height:2px;">

# %% [markdown] tags=[]
# ## Receptive Field
#
# The receptive field of an output value is the set of input values that can change the output value. The size of the receptive field is an important property of network architectures for image processing. Let's consider the receptive field size of the U-Net building blocks.

# %% [markdown] tags=[]
# <div class="alert alert-warning">
#
# <h4>Question: Receptive Field Size</h4>
# What are the receptive field sizes of the following operations?
#
# <ol>
# <li> <code style="color: black">torch.nn.Conv2d(1, 5, 3)</code></li>
# <li> <code style="color: black">torch.nn.Sequential(torch.nn.Conv2d(1, 5, 3), torch.nn.Conv2d(5, 5, 3))</code></li>
# <li> <code style="color: black">torch.nn.Sequential(torch.nn.Conv2d(1, 5, 3), torch.nn.Conv2d(5, 5, 5))</code></li>
# <li> <code style="color: black">Downsample(3)</code></li>
# <li> <code style="color: black">torch.nn.Sequential(ConvBlock(1, 5, 3), Downsample(2), ConvBlock(5, 5, 3))</code></li>
# <li> <code style="color: black">torch.nn.Upsample(2)</code></li>
# <li> <code style="color: black">torch.nn.Sequential(ConvBlock(1, 5, 3), Upsample(2), ConvBlock(5, 5, 3))</code></li>
# <li> <code style="color: black">torch.nn.Sequential(ConvBlock(1, 5, 3), Downsample(3), ConvBlock(5, 5, 3), Upsample(3), ConvBlock(5, 5, 3))</code></li>
# <li> <code style="color: black">UNet(depth=2, in_channels=1, downsample_factor=3, kernel_size=3)</code></li>
#
# </div>

# %% [markdown] tags=["solution"]
# <div class="alert alert-block alert-warning">
# <ol>
# <li> 3x3 </li>
# <li> 5x5 </li>
# <li> 7x7 </li>
# <li> 3x3 </li>
# <li> 14x14 </li>
# <li> 1x1 </li>
# <li> 8x8 </li>
# <li> 22x22 </li>
# <li> 22x22 </li>
# </div>

# %% [markdown] tags=[]
# <div class="alert alert-block alert-info">
#     <h4>Task 7: Receptive Field</h4>
#     <p>The <code style="color: black">plot_receptive_field</code> function visualizes the receptive field of a given U-Net - the square shows how many input pixels contribute to the output at the center pixel. Try it out with different U-Nets to get a sense of how varying the depth, kernel size, and downsample factor affect the receptive field of a U-Net.</p>
# </div>

# %% tags=["task"]
new_net = ...  # TASK 7: declare your U-Net here
if isinstance(new_net, UNet):
    plot_receptive_field(new_net)

# %% tags=["solution"]
# SOLUTION 7: declare your U-Net here
new_net = UNet(
    depth=2,
    in_channels=1,
    downsample_factor=2,
    kernel_size=3,
)
if isinstance(new_net, UNet):
    plot_receptive_field(new_net)

# %% [markdown] tags=[]
# <div class="alert alert-block alert-success">
#     <h2>Checkpoint 3</h2>
#
# Looking at the receptive field of your network is one of the most important aspects to consider when you choose the hyperparameters for your network.
#
# Questions to consider:
# <ol>
#     <li>Which hyperparameter of the U-Net has most effect on the receptive field?</li>
#     <li>If two sets of hyperparameters result in the same receptive field size are those networks equally good choices?</li>
#     <li>What's the relation between the receptive field size and the size of images you feed to your UNet?</li>
#     <li>Do you see a connection between padding and receptive field size?</li>
#     <li>For each hyperparameter: Can you think of scenarios in which you would consider changing this parameter? Why? </li>
# </ol>
#
# </div>
#
# <hr style="height:2px;">

# %% [markdown] tags=["solution"]
# <div class="alert alert-block alert-success">
# <ol>
#   <li>The depth and the downsample factor have the most effect on the receptive field size.</li>
#   <li>No - other factors such as the number of feature maps also affect the capacity of the network to learn complex functions. Additionally, not all receptive fields are created equal - for example, you can have a large receptive field by choosing a very large downsample factor, but this will result in a very coarse receptive field that may not be ideal for your task.</li>
#   <li>The receptive field size is independent of the size of images you feed to your UNet. You want the images to be larger than the receptive field size - otherwise you need to pad with zeros and learn on the padded pixels instead of real data. But once your images are as big as the receptive field size, increasing the image size does not allow your network to look at more pixels at once. It does however increase the number of pixels that contribute to the loss which can make training more stable, similar to increasing the batch size.</li>
#   <li>The receptive field size is equivalent ot the amount of padding you would need for "same" instead of "valid".</li>
#   <li>Open discussion!</li>
# </ol>
# </div>
#
# <hr style="height:2px;">

# %% [markdown] tags=[]
# ## Translational equivariance
#
# Depending on the task you're trying to solve you may care about translational (shift) invariance or equivariance.
#
# Let's first define what these invariance and equivariance mean in mathematical notation.
#
# Let $T$ be a transformation and $F$ the function whose properties we're considering.
#
# $F$ is invariant under transformation $T$ if: $F(T(x)) = F(x)$. The output of the function remains the same whether the input was transformed or not.
#
# $F$ is equivariant under transformation $T$ if: $F(T(x)) = T(F(x))$. Applying the function on the transformed input is the same as applying the transformation on the output of the original input.

# %% [markdown] tags=[]
# If math isn't your thing hopefully this picture helps to convey the concept, now specifically for translations.

# %% [markdown] tags=[]
# <img src="static/equivariance.png" alt="Invariance and Equivariance" style="width: 1500px;"/>

# %% [markdown] tags=[]
# <div class="alert alert-warning">
#
# <h4>Question: Translational invariance and equivariance</h4>
# For what types of deep learning tasks would you want your network to be translationally invariant and equivariant, respectively? Where does the U-Net fit in?
# </div>

# %% [markdown] tags=[]
# <div class="alert alert-warning">
#
# <h4>Question: Translational properties of U-Net building blocks</h4>
# For each of these building blocks of the U-Net: Is it translationally equivariant or invariant?
# <ol>
#     <li>ConvBlock</li>
#     <li>Downsample</li>
#     <li>Upsample</li>
# </ol>
# </div>

# %% [markdown] tags=["solution"]
# <div class="alert alert-block alert-warning">
#
# <ol>
#   <li>ConvBlock: Translationally equivariant for any shift.</li>
#   <li>Downsample: Translationally equivariant for shifts that are multiples of the downsampling factor.</li>
#   <li>Upsample: Translationally equivariant for any shift of the lower-resolution input.</li>
# </ol>
# A U-Net with valid padding can be translationally equivariant under shifts that are multiples of the product of the downsample factors. However only if the output size is a multiple of that product of downsample factors (see Bonus Exercise).
# </div>

# %% [markdown] tags=[]
# <div class="alert alert-block alert-success">
#     <h2>Checkpoint 4</h2>
#
#
# Questions and points to consider:
# <ol>
#     <li>How the output of your network behaves under translations becomes particularly important when your images are too large to be fed into your network as a whole. To solve this the input image is split into tiles and ideally their output should fit together seamlessly. Do you see how seamless stitching is equivalent to translational equivariance under shifts under your tile size and multiples thereof?</li>
#     <li>How does padding affect the translational equivariance of your network?</li>
#     <li>Can you think of ways to design an architecture to achieve translation invariance?</li>
# </ol>
#
# </div>
#
# <hr style="height:2px;">

# %% [markdown] tags=["solution"]
# <div class="alert alert-block alert-success">
# <ol>
#   <li>Seamless stitching is equivalent to translational equivariance under shifts of your tile size. If you want to guarantee no stitching artifacts, you need to ensure that your network is translationally equivariant under shifts of your output tile size.</li>
#   <li>Padding breaks translational equivariance. The zeros do affect the output and they appear at different locations in the unshifted and shifted inputs.</li>
#   <li>One simple way to design an architecture to achieve translation invariance is to use an equivariant architecture and then apply a global pooling operation at the end.</li>
# </ol>
# </div>
#
# <hr style="height:2px;">
# %% [markdown] tags=[]
# ## Train a U-Net!
# We will get more into the details of evaluating semantic segmentation models in the next exercise. For now, we will provide an example pipeline that will train a U-Net to classify each pixel in an image of cells as foreground or background. You should have seen training loops like this in the pre-course exercises, so there is no implementation task here.
#
# We will use Tensorboard to log our training runs, so there is code in the train loop to log aspects of the training to Tensorboard.
#
# **A note on the loss:** below we use `MSELoss` together with a `Sigmoid` final activation. For foreground/background segmentation, `BCELoss` (or `BCEWithLogitsLoss` without a final sigmoid) is the more standard choice. We use MSE here purely for simplicity — in the next exercise you'll see why BCE/Dice are the better default for segmentation tasks.

# %% tags=[]
train_dataset = NucleiDataset("nuclei_train_data", transforms.RandomCrop(256))
train_loader = DataLoader(train_dataset)

# %% tags=[]
loss_function: torch.nn.Module = torch.nn.MSELoss()


# %% tags=[]
# apply training for one epoch
def train(
    model,
    loader,
    optimizer,
    loss_function,
    epoch,
    log_interval=100,
    log_image_interval=20,
    tb_logger=None,
    device=None,
    early_stop=False,
):
    if device is None:
        # You can pass in a device or we will default to using
        # the gpu. Feel free to try training on the cpu to see
        # what sort of performance difference there is
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # set the model to train mode
    model.train()

    # move model to device
    model = model.to(device)

    # iterate over the batches of this epoch
    for batch_id, (x, y) in enumerate(loader):
        # move input and target to the active device (either cpu or gpu)
        x, y = x.to(device), y.to(device)

        # zero the gradients for this iteration
        optimizer.zero_grad()

        # apply model and calculate loss
        prediction = model(x)
        # if necessary, crop the masks to match the model output shape
        if prediction.shape != y.shape:
            y = center_crop(y, prediction.size()[2:])
        if y.dtype != prediction.dtype:
            y = y.type(prediction.dtype)
        loss = loss_function(prediction, y)

        # backpropagate the loss and adjust the parameters
        loss.backward()
        optimizer.step()

        # log to console
        if batch_id % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_id * len(x),
                    len(loader.dataset),
                    100.0 * batch_id / len(loader),
                    loss.item(),
                )
            )

        # log to tensorboard
        if tb_logger is not None:
            step = epoch * len(loader) + batch_id
            tb_logger.add_scalar(
                tag="train_loss", scalar_value=loss.item(), global_step=step
            )
            # check if we log images in this iteration
            if step % log_image_interval == 0:
                x = unnormalize(x)
                tb_logger.add_images(
                    tag="input", img_tensor=x.to("cpu"), global_step=step
                )
                tb_logger.add_images(
                    tag="target", img_tensor=y.to("cpu"), global_step=step
                )
                tb_logger.add_images(
                    tag="prediction",
                    img_tensor=prediction.to("cpu").detach(),
                    global_step=step,
                )
                combined_image = torch.cat(
                    [x, pad_to_size(y, x.size()), pad_to_size(prediction, x.size())],
                    dim=3,
                )

                tb_logger.add_images(
                    tag="input_target_prediction",
                    img_tensor=combined_image,
                    global_step=step,
                )

        if early_stop and batch_id > 5:
            print("Stopping test early!")
            break


# %% [markdown]
# The next two cells start tensorboard.


# %% tags=[]
# Function to find an available port
def find_free_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# Launch TensorBoard on the browser
def launch_tensorboard(log_dir):
    port = find_free_port()
    tensorboard_cmd = f"tensorboard --logdir={log_dir} --port={port}"
    process = subprocess.Popen(tensorboard_cmd, shell=True)
    print(
        f"TensorBoard started at http://localhost:{port}. \n"
        "If you are using VSCode remote session, forward the port using the PORTS tab next to TERMINAL."
    )
    return process


# %% tags=[]
# Launch tensorboard and click on the link to view the logs.
tensorboard_process = launch_tensorboard("unet_runs")

# %% [markdown] tags=[]
"""
<div class="alert alert-warning">
If you are using VSCode and a remote server, you will need to forward the port to view the tensorboard. <br>
Take note of the port number was assigned in the previous cell.(i.e <code style="color: black"> http://localhost:{port_number_assigned}</code>) <br>

Locate the your VSCode terminal and select the <code style="color: black">Ports</code> tab <br>
<ul>
<li>Add a new port with the <code style="color: black">port_number_assigned</code>
</ul>
Click on the link to view the tensorboard and it should open in your browser.
</div>
<div class="alert alert-warning">
If you launched jupyter lab from ssh terminal, add <code style="color: black">--host &lt;your-server-name&gt;</code> to the tensorboard command below. <code style="color: black">&lt;your-server-name&gt;</code> is the address of your compute node that ends in amazonaws.com.

</div>
"""

# %% [markdown] tags=[]
# <div class="alert alert-block alert-info">
#     <h3>Task 8: Declare and train a U-Net</h3>
#     <ol>
#         <li>Declare a U-Net with your favorite parameters!</li>
#         <li>Train the U-Net for 5 epochs, and inspect the training loss and the output predictions in Tensorboard.</li>
#         </ol>
# </div>

# %% tags=["task"]
model = ...  # TASK 8.1: Declare your U-Net here and name it below
model_name = "my_fav_unet"  # This name will be used in the tensorboard logs
logger = SummaryWriter(f"unet_runs/{model_name}")

# %% tags=["solution"]
model = UNet(
    depth=3,
    in_channels=1,
    upsample_mode="bilinear",
    fmap_inc_factor=3,
    final_activation=torch.nn.Sigmoid(),
)  # SOLUTION 8.1: Declare your U-Net here and name it below
model_name = "my_fav_unet"  # This name will be used in the tensorboard logs
logger = SummaryWriter(f"unet_runs/{model_name}")

# %% tags=[]
# use adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# train for $5$ epochs
# TASK 8.2: Run this cell and examine outputs in Tensorboard above!
n_epochs = 5
for epoch in range(n_epochs):
    # train
    train(
        model,
        train_loader,
        optimizer=optimizer,
        loss_function=loss_function,
        epoch=epoch,
        tb_logger=logger,
    )


# %% [markdown] tags=[]
# <div class="alert alert-block alert-info">
#     <h3>Task 9: Train more U-Nets! </h3>
#     <ol>
#         <li>Declare a new U-Net with different parameters. If you use a different name in the Tensorboard logger, you will be able to know which logs are from which run.</li>
#         <li>Train your new U-Net - does it perform better or worse? How can you tell?</li>
#         <li>Keep tinkering with different architectures until the end of the exercise!</li>
#         </ol>
# </div>

# %% [markdown] tags=[]
# <div class="alert alert-block alert-success">
#     <h2>Checkpoint 5</h2>
#
# Congratulations! You trained your first UNet that you implemented all by yourself!
#
# We will keep using this U-Net throughout the rest of the exercises. Whenever you see an import like `import dlmbl_unet` or
# `from dlmbl_unet import UNet` it will be importing from [this repository](https://github.com/dlmbl/dlmbl-unet) which contains the solution to this notebook as a package (including the bonus exercises so don't peek just yet if you wanna solve the bonus too).
# </div>

# %% [markdown] tags=[]
# ## Bonus 1: 3D UNet
# The UNet you implemented so far only works for 2D images, but in microscopy we often have 3D data that also needs to be processed as such, i.e. for some tasks it is important that the network's receptive field is 3D. So in this bonus exercise we will change our implementation to make the number of dimensions configurable.
#

# %% [markdown] tags=[]
# <div class="alert alert-block alert-info">
#     <h3>Task 10: Make U-Net building blocks configurable for 2D and 3D data! </h3>
#     To make the same class usable for 2D and 3D data we will add an argument `ndim` to each building block.
# </div>


# %% tags=["task"]
class Downsample(torch.nn.Module):
    def __init__(self, downsample_factor: int, ndim: int = 2):
        """Initialize a MaxPool2d module with the input downsample factor"""

        super().__init__()
        if ndim not in (2, 3):
            msg = f"Invalid number of dimensions: {ndim=}. Options are 2 or 3."
            raise ValueError(msg)
        self.ndim = ndim
        self.downsample_factor = downsample_factor
        # TASK 10A: Initialize the maxpool module
        # Define what the downop should be based on `ndim`.
        self.down = ...  # YOUR CODE HERE

    def check_valid(self, image_size: tuple[int, ...]) -> bool:
        """Check if the downsample factor evenly divides each image dimension.
        Returns `True` for valid image sizes and `False` for invalid image sizes.
        Note: there are multiple ways to do this!
        """
        # TASK 10B: Check that the image_size is valid to use with the downsample factor
        # YOUR CODE HERE
        # You can likely copy this from Task 2B2

    def forward(self, x):
        if not self.check_valid(tuple(x.size()[-self.ndim :])):
            raise RuntimeError(
                "Can not downsample shape %s with factor %s"
                % (x.size(), self.downsample_factor)
            )

        return self.down(x)


# %% tags=["task"]
class ConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: str = "same",
        ndim: int = 2,
    ):
        """A convolution block for a U-Net. Contains two convolutions, each followed by a ReLU.

        Args:
            in_channels (int): The number of input channels for this conv block. Depends on
                the layer and side of the U-Net and the hyperparameters.
            out_channels (int): The number of output channels for this conv block. Depends on
                the layer and side of the U-Net and the hyperparameters.
            kernel_size (int): The size of the kernel. A kernel size of N signifies an
                NxN square kernel.
            padding (str): The type of convolution padding to use. Either "same" or "valid".
                Defaults to "same".
            ndim (int): Number of dimensions for the convolution operation. Use 2 for 2D
                convolutions and 3 for 3D convolutions. Defaults to 2.
        """
        super().__init__()
        if ndim not in (2, 3):
            msg = f"Invalid number of dimensions: {ndim=}. Options are 2 or 3."
            raise ValueError(msg)
        if kernel_size % 2 == 0:
            msg = "Only allowing odd kernel sizes."
            raise ValueError(msg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        # TASK 10C: Initialize your modules and define layers.
        # Use the convolution module matching `ndim`.
        # YOUR CODE HERE

        for _name, layer in self.named_modules():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Conv3d)):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

    def forward(self, x):
        # TASK 10D: Apply the modules you defined to the input x
        ...  # YOUR CODE HERE


# %% tags=["task"]
class OutputConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: torch.nn.Module | None = None,
        ndim: int = 2,
    ):
        """
        A module that uses a convolution with kernel size 1 to get the appropriate
        number of output channels, and then optionally applies a final activation.

        Args:
            in_channels (int): The number of feature maps that will be input to the
                OutputConv block.
            out_channels (int): The number of channels that you want in the output
            activation (str | None, optional): Accepts the name of any torch activation
                function  (e.g., ``ReLU`` for ``torch.nn.ReLU``) or None for no final
                activation. Defaults to None.
            ndim (int): Number of dimensions for convolution operation. Use 2 for 2D
                convolutions and 3 for 3D convolutions. Defaults to 2.
        """
        super().__init__()
        if ndim not in (2, 3):
            msg = f"Invalid number of dimensions: {ndim=}. Options are 2 or 3."
            raise ValueError(msg)

        # TASK 10E: Define the convolution submodule.
        # Use the convolution module matching `ndim`.
        # YOUR CODE HERE

        self.activation = activation

    def forward(self, x):
        # TASK 10F: Implement the forward function
        # YOUR CODE HERE
        ...


# %% tags=["task"]
class UNet(torch.nn.Module):
    def __init__(
        self,
        depth: int,
        in_channels: int,
        out_channels: int = 1,
        final_activation: torch.nn.Module | None = None,
        num_fmaps: int = 64,
        fmap_inc_factor: int = 2,
        downsample_factor: int = 2,
        kernel_size: int = 3,
        padding: str = "same",
        upsample_mode: str = "nearest",
        ndim: int = 2,
    ):
        """A U-Net for 2D or 3D input that expects tensors shaped like:
            ``(batch, channels, height, width)`` or ``(batch, channels, depth, height, width)``,
            respectively.

        Args:
            depth:
                The number of levels in the U-Net. 2 is the smallest that really
                makes sense for the U-Net architecture, as a one layer U-Net is
                basically just 2 conv blocks.
            in_channels:
                The number of input channels in your dataset.
            out_channels (optional):
                How many output channels you want. Depends on your task. Defaults to 1.
            final_activation (optional):
                What activation to use in your final output block. Depends on your task.
                Defaults to None.
            num_fmaps (optional):
                The number of feature maps in the first layer. Defaults to 64.
            fmap_inc_factor (optional):
                By how much to multiply the number of feature maps between
                layers. Encoder layer ``l`` will have ``num_fmaps*fmap_inc_factor**l``
                output feature maps. Defaults to 2.
            downsample_factor (optional):
                Factor to use for down- and up-sampling the feature maps between layers.
                Defaults to 2.
            kernel_size (int, optional):
                Kernel size to use in convolutions on both sides of the UNet.
                Defaults to 3.
            padding (str, optional):
                How to pad convolutions. Either 'same' or 'valid'. Defaults to "same."
            upsample_mode (str, optional):
                The upsampling mode to pass to torch.nn.Upsample. Usually "nearest"
                or "bilinear." Defaults to "nearest."
            ndim (int, optional): Number of dimensions for the U-Net. Use 2 for 2D U-Net and
                3 for 3D U-Net. Defaults to 2.
        """

        super().__init__()
        if ndim not in (2, 3):
            msg = f"Invalid number of dimensions: {ndim=}. Options are 2 or 3."
            raise ValueError(msg)
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.final_activation = final_activation
        self.num_fmaps = num_fmaps
        self.fmap_inc_factor = fmap_inc_factor
        self.downsample_factor = downsample_factor
        self.kernel_size = kernel_size
        self.padding = padding
        self.upsample_mode = upsample_mode

        # left convolutional passes
        self.left_convs = torch.nn.ModuleList()
        # TASK 10G: Initialize list here
        # Loop through each level of the encoder from top (level=0) to bottom (level=self.depth - 1)
        for level in range(self.depth):
            # conv =
            # Adding conv module to the list
            self.left_convs.append(conv)

        # right convolutional passes
        self.right_convs = torch.nn.ModuleList()
        # TASK 10H: Initialize list here
        # Loop through each level of the decoder from top (level=0) to one above bottom (level=self.depth - 2)
        for level in range(self.depth - 1):
            # Initialize conv module
            # conv =
            # Adding conv module to the list
            self.right_convs.append(conv)

        # TASK 10I: Initialize other modules here
        # Same here, copy over from TASK 6.2, but make sure to add the ndim argument
        # as needed.

    def forward(self, x):
        # left side
        # Hint - you will need the outputs of each convolutional block in the encoder for the skip connection, so you need to hold on to those output tensors
        for i in range(self.depth - 1):
            # TASK 10L: Implement encoder here
            # Copy from TASK 6.3A
            ...

        # bottom
        # TASK 10M: Implement bottom of U-Net here
        # Copy from TASK 6.3B

        # right
        for i in range(0, self.depth - 1)[::-1]:
            # TASK 10N: Implement decoder here
            # Copy from TASK 6.3C
            ...
        # TASK 10O: Apply the final convolution and return the output
        # Copy from TASK 6.3D
        return


# %% tags=["solution"]
class Downsample(torch.nn.Module):
    def __init__(self, downsample_factor: int, ndim: int = 2):
        """Initialize a MaxPool2d module with the input downsample factor"""

        super().__init__()
        if ndim not in (2, 3):
            msg = f"Invalid number of dimensions: {ndim=}. Options are 2 or 3."
            raise ValueError(msg)
        self.ndim = ndim
        self.downsample_factor = downsample_factor
        # SOLUTION 10A: Initialize the maxpool module
        # Define what the downop should be based on `ndim`.
        downops = {2: torch.nn.MaxPool2d, 3: torch.nn.MaxPool3d}
        self.down = downops[ndim](downsample_factor)

    def check_valid(self, image_size: tuple[int, ...]) -> bool:
        """Check if the downsample factor evenly divides each image dimension.
        Returns `True` for valid image sizes and `False` for invalid image sizes.
        Note: there are multiple ways to do this!
        """
        # SOLUTION 10B: Check that the image_size is valid to use with the downsample factor
        # YOUR CODE HERE
        # You can likely copy this from Task 2B2
        for dim in image_size:
            if dim % self.downsample_factor != 0:
                return False
        return True

    def forward(self, x):
        if not self.check_valid(tuple(x.size()[-self.ndim :])):
            raise RuntimeError(
                "Can not downsample shape %s with factor %s"
                % (x.size(), self.downsample_factor)
            )

        return self.down(x)


# %% tags=["solution"]
class ConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: str = "same",
        ndim: int = 2,
    ):
        """A convolution block for a U-Net. Contains two convolutions, each followed by a ReLU.

        Args:
            in_channels (int): The number of input channels for this conv block. Depends on
                the layer and side of the U-Net and the hyperparameters.
            out_channels (int): The number of output channels for this conv block. Depends on
                the layer and side of the U-Net and the hyperparameters.
            kernel_size (int): The size of the kernel. A kernel size of N signifies an
                NxN square kernel.
            padding (str): The type of convolution padding to use. Either "same" or "valid".
                Defaults to "same".
            ndim (int): Number of dimensions for the convolution operation. Use 2 for 2D
                convolutions and 3 for 3D convolutions. Defaults to 2.
        """
        super().__init__()
        if ndim not in (2, 3):
            msg = f"Invalid number of dimensions: {ndim=}. Options are 2 or 3."
            raise ValueError(msg)
        if kernel_size % 2 == 0:
            msg = "Only allowing odd kernel sizes."
            raise ValueError(msg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        # SOLUTION 10C: Initialize your modules and define layers.
        # Use the convolution module matching `ndim`.
        # YOUR CODE HERE
        convops = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d}
        self.conv_pass = torch.nn.Sequential(
            convops[ndim](
                in_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            torch.nn.ReLU(),
            convops[ndim](
                out_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            torch.nn.ReLU(),
        )
        for _name, layer in self.named_modules():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Conv3d)):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

    def forward(self, x):
        # SOLUTION 10D: Apply the modules you defined to the input x
        output = self.conv_pass(x)
        return output


# %% tags=["solution"]
class OutputConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: torch.nn.Module | None = None,
        ndim: int = 2,
    ):
        """
        A module that uses a convolution with kernel size 1 to get the appropriate
        number of output channels, and then optionally applies a final activation.

        Args:
            in_channels (int): The number of feature maps that will be input to the
                OutputConv block.
            out_channels (int): The number of channels that you want in the output
            activation (str | None, optional): Accepts the name of any torch activation
                function  (e.g., ``ReLU`` for ``torch.nn.ReLU``) or None for no final
                activation. Defaults to None.
            ndim (int): Number of dimensions for convolution operation. Use 2 for 2D
                convolutions and 3 for 3D convolutions. Defaults to 2.
        """
        super().__init__()
        if ndim not in (2, 3):
            msg = f"Invalid number of dimensions: {ndim=}. Options are 2 or 3."
            raise ValueError(msg)
        # SOLUTION 10E: Define the convolution submodule.
        # Use the convolution module matching `ndim`.
        convops = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d}
        self.final_conv = convops[ndim](in_channels, out_channels, 1, padding=0)

        self.activation = activation

    def forward(self, x):
        # SOLUTION 10F: Implement the forward function
        x = self.final_conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


# %% tags=["solution"]
class UNet(torch.nn.Module):
    def __init__(
        self,
        depth: int,
        in_channels: int,
        out_channels: int = 1,
        final_activation: torch.nn.Module | None = None,
        num_fmaps: int = 64,
        fmap_inc_factor: int = 2,
        downsample_factor: int = 2,
        kernel_size: int = 3,
        padding: str = "same",
        upsample_mode: str = "nearest",
        ndim: int = 2,
    ):
        """A U-Net for 2D or 3D input that expects tensors shaped like:
            ``(batch, channels, height, width)`` or ``(batch, channels, depth, height, width)``,
            respectively.

        Args:
            depth:
                The number of levels in the U-Net. 2 is the smallest that really
                makes sense for the U-Net architecture, as a one layer U-Net is
                basically just 2 conv blocks.
            in_channels:
                The number of input channels in your dataset.
            out_channels (optional):
                How many output channels you want. Depends on your task. Defaults to 1.
            final_activation (optional):
                What activation to use in your final output block. Depends on your task.
                Defaults to None.
            num_fmaps (optional):
                The number of feature maps in the first layer. Defaults to 64.
            fmap_inc_factor (optional):
                By how much to multiply the number of feature maps between
                layers. Encoder layer ``l`` will have ``num_fmaps*fmap_inc_factor**l``
                output feature maps. Defaults to 2.
            downsample_factor (optional):
                Factor to use for down- and up-sampling the feature maps between layers.
                Defaults to 2.
            kernel_size (int, optional):
                Kernel size to use in convolutions on both sides of the UNet.
                Defaults to 3.
            padding (str, optional):
                How to pad convolutions. Either 'same' or 'valid'. Defaults to "same."
            upsample_mode (str, optional):
                The upsampling mode to pass to torch.nn.Upsample. Usually "nearest",
                "bilinear" (2D) or "trilinear" (3D). Defaults to "nearest".
            ndim (int, optional): Number of dimensions for the U-Net. Use 2 for 2D U-Net and
                3 for 3D U-Net. Defaults to 2.
        """

        super().__init__()
        if ndim not in (2, 3):
            msg = f"Invalid number of dimensions: {ndim=}. Options are 2 or 3."
            raise ValueError(msg)
        if ndim == 2 and upsample_mode == "trilinear":
            msg = "Trilinear interpolation is not valid for a 2D UNet. Please choose `nearest` or `bilinear` instead"
            raise ValueError(msg)
        if ndim == 3 and upsample_mode == "bilinear":
            msg = "Bilinear interpolation is not valid for a 3D UNet. Please choose `nearest` or `trilinear` instead"
            raise ValueError(msg)
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.final_activation = final_activation
        self.num_fmaps = num_fmaps
        self.fmap_inc_factor = fmap_inc_factor
        self.downsample_factor = downsample_factor
        self.kernel_size = kernel_size
        self.padding = padding
        self.upsample_mode = upsample_mode

        # left convolutional passes
        self.left_convs = torch.nn.ModuleList()
        # SOLUTION 10G: Initialize list here
        # After you implemented the conv pass you can copy this from TASK 6.1A,
        # but make sure to pass the ndim argument
        # Loop through each level of the encoder from top (level=0) to bottom (level=self.depth - 1)
        for level in range(self.depth):
            fmaps_in, fmaps_out = self.compute_fmaps_encoder(level)
            conv = ConvBlock(
                fmaps_in, fmaps_out, self.kernel_size, self.padding, ndim=ndim
            )
            # Adding conv module to the list
            self.left_convs.append(conv)
        # right convolutional passes
        self.right_convs = torch.nn.ModuleList()
        # SOLUTION 10H: Initialize list here
        # After you implemented the conv pass you can copy this from TASK 6.1B,
        # but make sure to pass the ndim argument
        # Loop through each level of the decoder from top (level=0) to one above bottom (level=self.depth - 2)
        for level in range(self.depth - 1):
            fmaps_in, fmaps_out = self.compute_fmaps_decoder(level)
            # Initialize conv module
            conv = ConvBlock(
                fmaps_in, fmaps_out, self.kernel_size, self.padding, ndim=ndim
            )
            # Adding conv module to the list
            self.right_convs.append(conv)

        # SOLUTION 10I: Initialize other modules here
        # Same here, copy over from TASK 6.2, but make sure to add the ndim argument
        # as needed.
        self.downsample = Downsample(self.downsample_factor, ndim=ndim)
        self.upsample = torch.nn.Upsample(
            scale_factor=self.downsample_factor,
            mode=self.upsample_mode,
        )
        self.crop_and_concat = CropAndConcat()
        self.final_conv = OutputConv(
            self.compute_fmaps_decoder(0)[1],
            self.out_channels,
            self.final_activation,
            ndim=ndim,
        )

    def compute_fmaps_encoder(self, level: int) -> tuple[int, int]:
        """Compute the number of input and output feature maps for
        a conv block at a given level of the UNet encoder (left side).

        Args:
            level (int): The level of the U-Net which we are computing
            the feature maps for. Level 0 is the input level, level 1 is
            the first downsampled layer, and level=depth - 1 is the bottom layer.

        Output (tuple[int, int]): The number of input and output feature maps
            of the encoder convolutional pass in the given level.
        """
        # SOLUTION 10J: Implement this function.
        if level == 0:
            fmaps_in = self.in_channels
        else:
            fmaps_in = self.num_fmaps * self.fmap_inc_factor ** (level - 1)

        fmaps_out = self.num_fmaps * self.fmap_inc_factor**level
        return fmaps_in, fmaps_out

    def compute_fmaps_decoder(self, level: int) -> tuple[int, int]:
        """Compute the number of input and output feature maps for a conv block
        at a given level of the UNet decoder (right side). Note:
        The bottom layer (depth - 1) is considered an "encoder" conv pass,
        so this function is only valid up to depth - 2.

        Args:
            level (int): The level of the U-Net which we are computing
            the feature maps for. Level 0 is the input level, level 1 is
            the first downsampled layer, and level=depth - 1 is the bottom layer.

        Output (tuple[int, int]): The number of input and output feature maps
            of the decoder convolutional pass in the given level.
        """
        # SOLUTION 10K: Implement this function.
        fmaps_out = self.num_fmaps * self.fmap_inc_factor ** (level)
        concat_fmaps = self.compute_fmaps_encoder(level)[
            1
        ]  # The channels that come from the skip connection
        fmaps_in = concat_fmaps + self.num_fmaps * self.fmap_inc_factor ** (level + 1)

        return fmaps_in, fmaps_out

    def forward(self, x):
        # left side
        # Hint - you will need the outputs of each convolutional block in the encoder for the skip connection, so you need to hold on to those output tensors
        convolution_outputs = []
        layer_input = x
        for i in range(self.depth - 1):
            # SOLUTION 10L: Implement encoder here
            # Copy from TASK 6.3A
            conv_out = self.left_convs[i](layer_input)
            convolution_outputs.append(conv_out)
            downsampled = self.downsample(conv_out)
            layer_input = downsampled

        # bottom
        # SOLUTION 10M: Implement bottom of U-Net here
        # Copy from TASK 6.3B
        conv_out = self.left_convs[-1](layer_input)
        layer_input = conv_out

        # right
        for i in range(0, self.depth - 1)[::-1]:
            # SOLUTION 10N: Implement decoder here
            # Copy from TASK 6.3C
            upsampled = self.upsample(layer_input)
            concat = self.crop_and_concat(convolution_outputs[i], upsampled)
            conv_output = self.right_convs[i](concat)
            layer_input = conv_output

        # SOLUTION 10O: Apply the final convolution and return the output
        # Copy from TASK 6.3D
        return self.final_conv(layer_input)


# %% [markdown] tags=[]
# Run the 3d test for your implementation of the UNet.

# %% tags=[]
unet_tests.TestUNet(UNet).run3d()

# %% [markdown] tags=[]
# ## Bonus 2: Tilable UNet
# In order to process arbitrarily large images, we can divide the image into smaller tiles and process each tile separately. To avoid artifacts at tile boundaries, the U-Net needs to be equivariant w.r.t. translations by the tile size. Add a cropping layer at the end of the U-Net that crops the output to a size for which the U-Net is equivariant. Can tranlsational equivariance be achieved for any parameter setting?

# %% [markdown] tags=[]
# <div class="alert alert-block alert-info">
#     <h3>Task 11: Implement a Tilable U-Net</h3>
#     <p>Modify the U-Net to make it suitable for processing large images by tiling. You'll need to add a cropping
#          mechanism to ensure translational equivariance.</p>
#     <ol>
#         <li>Copy over your UNet implemntation from Task 10.</li>
#         <li>Crop the output of the U-Net to an appropriate tile size</li>
#         <li>Add a conditional to only crop the outputs for U-Nets that can be translation equivariant.</li>
#     </ol>
# </div>


# %% tags=["task"]
class UNet(torch.nn.Module):
    def __init__(
        self,
        depth: int,
        in_channels: int,
        out_channels: int = 1,
        final_activation: torch.nn.Module | None = None,
        num_fmaps: int = 64,
        fmap_inc_factor: int = 2,
        downsample_factor: int = 2,
        kernel_size: int = 3,
        padding: str = "same",
        upsample_mode: str = "nearest",
        ndim: int = 2,
    ):
        """A U-Net for 2D or 3D input that expects tensors shaped like:
            ``(batch, channels, height, width)`` or ``(batch, channels, depth, height, width)``,
            respectively.
        Args:
            depth:
                The number of levels in the U-Net. 2 is the smallest that really
                makes sense for the U-Net architecture, as a one layer U-Net is
                basically just 2 conv blocks.
            in_channels:
                The number of input channels in your dataset.
            out_channels (optional):
                How many output channels you want. Depends on your task. Defaults to 1.
            final_activation (optional):
                What activation to use in your final output block. Depends on your task.
                Defaults to None.
            num_fmaps (optional):
                The number of feature maps in the first layer. Defaults to 64.
            fmap_inc_factor (optional):
                By how much to multiply the number of feature maps between
                layers. Encoder layer ``l`` will have ``num_fmaps*fmap_inc_factor**l``
                output feature maps. Defaults to 2.
            downsample_factor (optional):
                Factor to use for down- and up-sampling the feature maps between layers.
                Defaults to 2.
            kernel_size (int, optional):
                Kernel size to use in convolutions on both sides of the UNet.
                Defaults to 3.
            padding (str, optional):
                How to pad convolutions. Either 'same' or 'valid'. Defaults to "same."
            upsample_mode (str, optional):
                The upsampling mode to pass to torch.nn.Upsample. Usually "nearest"
                or "bilinear." Defaults to "nearest."
            ndim (int, optional): Number of dimensions for the U-Net. Use 2 for 2D U-Net and
                3 for 3D U-Net. Defaults to 2.
        """
        super().__init__()
        # TASK 11.1: Initialize the modules of the U-Net
        # Copy from TASK 10

    def forward(self, x):
        # TASK 11.1: Apply the modules of the U-Net
        # Copy from TASK 10

        # TASK 11.3: Add conditional

        # TASK 11.2: Crop the output of the U-Net to an appropriate tile size

        return


# %% tags=["solution"]
class UNet(torch.nn.Module):
    def __init__(
        self,
        depth: int,
        in_channels: int,
        out_channels: int = 1,
        final_activation: torch.nn.Module | None = None,
        num_fmaps: int = 64,
        fmap_inc_factor: int = 2,
        downsample_factor: int = 2,
        kernel_size: int = 3,
        padding: str = "same",
        upsample_mode: str = "nearest",
        ndim: int = 2,
    ):
        """A U-Net for 2D or 3D input that expects tensors shaped like:
            ``(batch, channels, height, width)`` or ``(batch, channels, depth, height, width)``,
            respectively.
        Args:
            depth:
                The number of levels in the U-Net. 2 is the smallest that really
                makes sense for the U-Net architecture, as a one layer U-Net is
                basically just 2 conv blocks.
            in_channels:
                The number of input channels in your dataset.
            out_channels (optional):
                How many output channels you want. Depends on your task. Defaults to 1.
            final_activation (optional):
                What activation to use in your final output block. Depends on your task.
                Defaults to None.
            num_fmaps (optional):
                The number of feature maps in the first layer. Defaults to 64.
            fmap_inc_factor (optional):
                By how much to multiply the number of feature maps between
                layers. Encoder layer ``l`` will have ``num_fmaps*fmap_inc_factor**l``
                output feature maps. Defaults to 2.
            downsample_factor (optional):
                Factor to use for down- and up-sampling the feature maps between layers.
                Defaults to 2.
            kernel_size (int, optional):
                Kernel size to use in convolutions on both sides of the UNet.
                Defaults to 3.
            padding (str, optional):
                How to pad convolutions. Either 'same' or 'valid'. Defaults to "same."
            upsample_mode (str, optional):
                The upsampling mode to pass to torch.nn.Upsample. Usually "nearest"
                or "bilinear." Defaults to "nearest."
            ndim (int, optional): Number of dimensions for the U-Net. Use 2 for 2D U-Net and
                3 for 3D U-Net. Defaults to 2.
        """
        super().__init__()
        # SOLUTION 11.1: Initialize the modules of the U-Net
        if ndim not in (2, 3):
            msg = f"Invalid number of dimensions: {ndim=}. Options are 2 or 3."
            raise ValueError(msg)
        if ndim == 2 and upsample_mode == "trilinear":
            msg = "Trilinear interpolation is not valid for a 2D UNet. Please choose `nearest` or `bilinear` instead"
            raise ValueError(msg)
        if ndim == 3 and upsample_mode == "bilinear":
            msg = "Bilinear interpolation is not valid for a 3D UNet. Please choose `nearest` or `trilinear` instead"
            raise ValueError(msg)
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.final_activation = final_activation
        self.num_fmaps = num_fmaps
        self.fmap_inc_factor = fmap_inc_factor
        self.downsample_factor = downsample_factor
        self.kernel_size = kernel_size
        self.padding = padding
        self.upsample_mode = upsample_mode
        self.ndim = ndim

        # left convolutional passes
        self.left_convs = torch.nn.ModuleList()
        # Loop through each level of the encoder from top (level=0) to bottom (level=self.depth - 1)
        for level in range(self.depth):
            fmaps_in, fmaps_out = self.compute_fmaps_encoder(level)
            conv = ConvBlock(
                fmaps_in, fmaps_out, self.kernel_size, self.padding, ndim=ndim
            )
            # Adding conv module to the list
            self.left_convs.append(conv)
        # right convolutional passes
        self.right_convs = torch.nn.ModuleList()
        # Loop through each level of the decoder from top (level=0) to one above bottom (level=self.depth - 2)
        for level in range(self.depth - 1):
            fmaps_in, fmaps_out = self.compute_fmaps_decoder(level)
            # Initialize conv module
            conv = ConvBlock(
                fmaps_in, fmaps_out, self.kernel_size, self.padding, ndim=ndim
            )
            # Adding conv module to the list
            self.right_convs.append(conv)

        self.downsample = Downsample(self.downsample_factor, ndim=ndim)
        self.upsample = torch.nn.Upsample(
            scale_factor=self.downsample_factor,
            mode=self.upsample_mode,
        )
        self.crop_and_concat = CropAndConcat()
        self.final_conv = OutputConv(
            self.compute_fmaps_decoder(0)[1],
            self.out_channels,
            self.final_activation,
            ndim=ndim,
        )

    def compute_fmaps_encoder(self, level: int) -> tuple[int, int]:
        """Compute the number of input and output feature maps for
        a conv block at a given level of the UNet encoder (left side).

        Args:
            level (int): The level of the U-Net which we are computing
            the feature maps for. Level 0 is the input level, level 1 is
            the first downsampled layer, and level=depth - 1 is the bottom layer.

        Output (tuple[int, int]): The number of input and output feature maps
            of the encoder convolutional pass in the given level.
        """
        # SOLUTION 10J: Implement this function.
        if level == 0:
            fmaps_in = self.in_channels
        else:
            fmaps_in = self.num_fmaps * self.fmap_inc_factor ** (level - 1)

        fmaps_out = self.num_fmaps * self.fmap_inc_factor**level
        return fmaps_in, fmaps_out

    def compute_fmaps_decoder(self, level: int) -> tuple[int, int]:
        """Compute the number of input and output feature maps for a conv block
        at a given level of the UNet decoder (right side). Note:
        The bottom layer (depth - 1) is considered an "encoder" conv pass,
        so this function is only valid up to depth - 2.

        Args:
            level (int): The level of the U-Net which we are computing
            the feature maps for. Level 0 is the input level, level 1 is
            the first downsampled layer, and level=depth - 1 is the bottom layer.

        Output (tuple[int, int]): The number of input and output feature maps
            of the decoder convolutional pass in the given level.
        """
        # SOLUTION 10K: Implement this function.
        fmaps_out = self.num_fmaps * self.fmap_inc_factor ** (level)
        concat_fmaps = self.compute_fmaps_encoder(level)[
            1
        ]  # The channels that come from the skip connection
        fmaps_in = concat_fmaps + self.num_fmaps * self.fmap_inc_factor ** (level + 1)

        return fmaps_in, fmaps_out

    def forward(self, x):
        # SOLUTION 11.1: Apply the modules of the U-Net
        # left side
        # Hint - you will need the outputs of each convolutional block in the encoder for the skip connection, so you need to hold on to those output tensors
        convolution_outputs = []
        layer_input = x
        for i in range(self.depth - 1):
            conv_out = self.left_convs[i](layer_input)
            convolution_outputs.append(conv_out)
            downsampled = self.downsample(conv_out)
            layer_input = downsampled

        # bottom
        conv_out = self.left_convs[-1](layer_input)
        layer_input = conv_out

        # right
        for i in range(0, self.depth - 1)[::-1]:
            upsampled = self.upsample(layer_input)
            concat = self.crop_and_concat(convolution_outputs[i], upsampled)
            conv_output = self.right_convs[i](concat)
            layer_input = conv_output

        output = self.final_conv(layer_input)
        # TASK 11.3: Add conditional
        if self.padding == "valid":
            # TASK 11.2: Crop the output of the U-Net to an appropriate tile size
            output_size = output.size()[2:]
            lowest_res = self.downsample_factor ** (self.depth - 1)
            target_output_size = output_size - output_size % np.array(
                (lowest_res,) * self.ndim
            )
            output = center_crop(output, target_output_size)
        return output


# %% tags=[]
unet_tests.TestUNet(UNet).run3d_tiled()
