import cv2
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='patch extraction')
parser.add_argument('--imageDir', required=True)
parser.add_argument('--imageName', required=True)
parser.add_argument('--imageNumber', type=int, required=True)
args = parser.parse_args()

# load the image
image_dir =  args.imageDir
image_name = args.imageName
image_number = args.imageNumber
filename = os.path.join(image_dir, image_name)
image = cv2.imread(filename)

# define some values
patch_center_1 = None
patch_center_2 = None
if image_number == 0:
	patch_center_1 = np.array([1000, 2000]) # 0
	patch_center_2 = np.array([2200, 3300])
elif image_number == 1:
	patch_center_1 = np.array([2500, 3300]) # 0
	patch_center_2 = np.array([3300, 3300])	

# patch_center = np.array([80, 160]) # 3
# baseline = "woman_baseline.png"
# filename = "woman_patch.png"

first_patch_name = "{}_patch1.png".format(image_name)
second_patch_name = "{}_patch2.png".format(image_name)
first_patch_name = os.path.join(image_dir, first_patch_name)
second_patch_name = os.path.join(image_dir, second_patch_name)

# calc patch position and extract the patch
patch_size = int(200)

first_patch_x = int(patch_center_1[0] - patch_size / 2.)
first_patch_y = int(patch_center_1[1] - patch_size / 2.)
second_patch_x = int(patch_center_2[0] - patch_size / 2.)
second_patch_y = int(patch_center_2[1] - patch_size / 2.)

first_patch_image = image[first_patch_x:first_patch_x+patch_size, first_patch_y:first_patch_y+patch_size]
second_patch_image = image[second_patch_x:second_patch_x+patch_size, second_patch_y:second_patch_y+patch_size]


# show image and patch
color1 = (0, 255, 255)
color2 = (255, 0, 255)
color3 = (0, 0, 255)
thickness = 5
image = cv2.rectangle(image, (first_patch_y-5, first_patch_x-5), (first_patch_y+patch_size+5, first_patch_x+patch_size+5), color1, thickness) 
image = cv2.rectangle(image, (second_patch_y-5, second_patch_x-5), (second_patch_y+patch_size+5, second_patch_x+patch_size+5), color2, thickness)

cv2.imshow('image', image)
cv2.imshow('first_patch_image', first_patch_image)
cv2.imshow('second_patch_image', second_patch_image)

image_name = "modified_{}".format(image_name)
cv2.imwrite(first_patch_name, first_patch_image)
cv2.imwrite(second_patch_name, second_patch_image)
cv2.imwrite(os.path.join(image_dir, image_name), image)
cv2.waitKey(0)