# 2d_aspectratio_mediolateralaxis


#%%

import tifffile as tiff
import matplotlib.pyplot as plt
import os, shutil
import numpy as np

from cellpose import core, utils, io, models, metrics, plot
from glob import glob
#import pickle

use_GPU = core.use_gpu()
yn = ['NO', 'YES']
print(f'>>> GPU activated? {yn[use_GPU]}')


#%%
#nuclear segmentation in a folder of images
model_path = "H:/PROJECTS-03/Agnese/cellpose training/2d/models/nuclei_h2b"

