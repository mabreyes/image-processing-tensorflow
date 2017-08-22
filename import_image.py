import os
import ssl
import urllib.request as urq
import matplotlib.pyplot as plt
import numpy as np

def download_image(folder_name):
    os.mkdir(folder_name)

    for img_i in range(1, 11):
        f = '000%02d.jpg' %img_i
        url = 'https://s3.amazonaws.com/cadl/celeb-align/' + f
        print (f, end='\n')
        # urq.urlretrieve(url, os.path.join(folder_name, f))

    image_dataset = [os.path.join(folder_name, file_i)
                     for file_i in os.listdir(folder_name)
                     if file_i.endswith('.jpg')
                     or file_i.endswith('.png')]

def plot_image(file_name):
    img = plt.read(file_name)
    plt.imshow(img)

# Function call:
# img_select = files[np.random.randint(0, len(files))]
# plot_image(img_select)

def imcrop_tosquare(image):
    # image.shape, prints (a, b, c) where a [0] is row, b [1] is column, and c[2] is color channel
    # If shape of row (image.shape[0]) is greater than the column (image.shape[1])
    if image.shape[0] > image.shape[1]:
        # Subtract the row and column if there's an extra
        extra = (image.shape[0] - image.shape[1])
        # If extra is divisible by two
        if extra % 2 == 0:
            # Example: Shape is (128, 96, 3), so (128 - 96) // 2 = 16 (starts from 16 to -16)
            # There are 96 rows in all exactly as the columns (so square)
            crop = image[extra // 2: -extra // 2, : ]
        else:
            crop = image[max(0, extra // 2 + 1): min(-1, -(extra // 2)), :]
    # If shape of column (image.shape[1]) is greater than the row (image.shape[0])
    elif image.shape[1] > image.shape[0]:
        extra = (image.shape[1] = image.shape[0])
        if extra % 2 == 0:
            crop = image[:, extra // 2, -extra // 2]
        else:
            crop = image[:, max(0, extra // 2 + 1): min(-1, -(extra // 2))]
    # If the image is already square
    else:
        crop = image
    return crop

def imcrop(image, amt):
    if amt <= 0 or amt >= 1:
        return image
    row_i = int(image.shape[0] * amt) // 2
    col_i = int(image.shape[1] * amt) // 2
    return img[row_i: -row_i, col_i: -col_i]
