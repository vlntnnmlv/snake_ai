from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from pyglet.window import key
from app import *

import pandas as pd
import pyglet
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


window = Game("data/snake.jpeg", "data/food.png")
pyglet.app.run()
