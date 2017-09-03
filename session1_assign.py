from libs import utils
from skimage import data
from skimage.transform import resize
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import os

# Minimize TensorFlow console warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

""" Use this function to rename images in a directory if you want them
uniform from 000001.jpg to 000100.jpg
"""
"""def rename_images():
	path = './img_session1_assign'
	files = os.listdir(path)
	ctr = 1

	for file_i in files:
		f = '000%03d.jpg' %ctr
		os.rename(os.path.join(path, file_i), os.path.join(path, f + '.jpg'))
		ctr = ctr + 1

if __name__ == "__main__":
	rename_images()
"""

image_files = [os.path.join('img_session1_assign', image_files_i)
               for image_files_i in os.listdir('img_session1_assign')
               if '.jpg' in image_files_i]

assert(len(image_files) == 100)

# Read images contained in the dataset
images = [plt.imread(images_i)[..., :3] for images_i in image_files]

# Crop images to square 
images = [utils.imcrop_tosquare(images_i) for images_i in images]

# Resize crop images to 100px by 100px
images = [resize(images_i, (100, 100)) for images_i in images]

# Batch dimension (100, 100, 100, 3)
images = np.array(images).astype(np.float32)

# Returns error if images.shape is not (100, 100, 100, 3)
assert(images.shape == (100, 100, 100, 3))
# Plot figure
plt.figure(figsize=(10, 10))
# Save image montage to dataset.png
# plt.imshow(utils.montage(images, saveto='dataset.png'))
# Show figure
# plt.show()

sess = tf.Session()
"""You can use np.mean but it won't be recognized inside the tensorflow
session. We used tf.reduce_mean instead.
"""
mean_image_op = tf.reduce_mean(images, axis=0)
mean_image = sess.run(mean_image_op)
assert(mean_image.shape == (100, 100, 3))
plt.imshow(mean_image)
plt.show()
plt.imsave(arr=mean_image, fname='mean.png')
