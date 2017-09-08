from libs import utils
from skimage import data
from skimage.transform import resize
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

image_files = [os.path.join('img_session1_assign', image_i)
               for image_i in os.listdir('img_session1_assign')
               if '.jpg' in image_i]

image_read = [plt.imread(image_i)
               for image_i in image_read]
image_read = [utils.imcrop_tosquare(image_i)
               for image_i in image_read]