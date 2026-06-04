# unet-exercise

In this exercise repo for DL@MBL you will learn about the U-Net and implement a configurable version yourself.

## Exercise Overview

In this notebook, we will implement a U-Net architecture. Through this exercise you should gain an understanding of the U-Net architecture in particular as well as learn how to approach the implementation of an architecture in general and familiarize yourself a bit more with the inner workings of pytorch.

The exercise is split into five parts:

In part 1 you will implement the building blocks of the U-Net. That includes the convolutions, downsampling, upsampling and skip connections. We will go in the order of how difficult they are to implement.

In part 2 you will combine the modules you've built in part 1 to implement the U-Net module.

Parts 3 and 4 are light on coding tasks but you will learn about two important concepts: receptive fields and translational equivariance.

Finally, in part 5 you will train your first U-Net of the course! This will just be a first flavor though since you will learn much more about that in the next exercise.


## Setup

Please run the setup script to create the environment for this exercise and download data.

```bash
source setup.sh
```

Now open the `exercise.ipynb` notebook in your preferred tool and make sure to select the `02-unet-exercise` kernel.

## TA Info
- For development purposes please install `pre-commit` to run black and ruff.
- Edits to the exercise should be made in `solution.py`. A github action will generate the exercise and solution notebooks when you push your changes.
- Add the tag "solution" to cells that should only be in `solution.ipynb` and the tag "task" to cells that should only be in `exercise.ipynb`.
