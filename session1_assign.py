from skimage import data
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import os

def rename_images():
	path = './img_session1_assign'
	files = os.listdir(path)
	ctr = 1

	for file_i in files:
		f = '000%02d.jpg' %ctr
		os.rename(os.path.join(path, file_i), os.path.join(path, f + '.jpg'))
		ctr = ctr + 1

if __name__ == "__main__":
    rename_images()