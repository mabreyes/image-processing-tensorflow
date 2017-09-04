from libs import utils
from skimage import data
from skimage.transform import resize
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import os

# TODO: Study again these concepts
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
# plt.figure(figsize=(10, 10))
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
# plt.imshow(mean_image)
# plt.show()
plt.imsave(arr=mean_image, fname='mean.png')

# mean_image_4d = tf.reduce_mean(images, axis=0, keep_dims=True)
subtraction = images - tf.reduce_mean(images, axis=0, keep_dims=True)

"""In this part, I compared the difference between the standard deviation
of a batch that is not divided and divided to images.shape[0]. I did this 
because normally the formula of standard deviation is sqrt((x-mean)^2 / (n-1))
Where x is the image batch, mean is the mean of the image batch, and n is
the number of images contained inside the batch or images.shape[0]
"""
# Without / images.shape[0]
std_image_op = tf.sqrt(tf.reduce_mean(subtraction * subtraction, axis=0))
std_image = sess.run(std_image_op)
assert (std_image.shape == (100, 100) or std_image.shape == (100, 100, 3))
std_image_show = std_image / np.max(std_image)

# With / images.shape[0]
std_image_op_2 = tf.sqrt(tf.reduce_mean(subtraction * subtraction, axis=0) / images.shape[0])
std_image_2 = sess.run(std_image_op_2)
assert (std_image_2.shape == (100, 100) or std_image_2.shape == (100, 100, 3))
std_image_show_2 = std_image_2 / np.max(std_image_2)

# Show difference
fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True, sharex=True)
axs[0].imshow(std_image_show)
axs[0].set_title('Without / images.shape[0]')
axs[1].imshow(std_image_show_2)
axs[1].set_title('With / images.shape[0]')
# plt.show(fig.show())
# plt.imsave(arr=std_image_show, fname='std.png')

# TODO: Check if the equation for normalization is correct
# Normalization 
norm_images_op = (images - mean_image_op) / std_image_op
norm_images = sess.run(norm_images_op)
print(np.min(norm_images), np.max(norm_images))
print(images.dtype)
norm_images_show = (norm_images - np.min(norm_images)) / (np.max(norm_images) - np.min(norm_images))
plt.figure(figsize=(10, 10))
plt.imshow(utils.montage(norm_images_show, 'normalized.png'))
plt.show()
