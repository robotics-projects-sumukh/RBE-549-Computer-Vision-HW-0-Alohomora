#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code

Colab file can be found at:
	https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

# Code starts here:

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from cv2 import filter2D
from sklearn.cluster import KMeans
import time

def check_folder_exists(folder_path):
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)
            
def plot_images(fig_size, filters, x_len, y_len, name):
	fig = plt.figure(figsize=fig_size)
	length = len(filters)
	for idx in np.arange(length):
		ax = fig.add_subplot(y_len, x_len, idx + 1, xticks=[], yticks=[])
		plt.imshow(filters[idx], cmap="gray")
	plt.axis("off")
	plt.savefig(name, bbox_inches="tight", pad_inches=0.3)
	plt.close()

def generate_derivative_of_gaussian_filters(orientations, scales):
    filters = []
    for scale in range(len(scales)):
        for orientation in range(orientations):
            sigma = 1.5 * (scale + 1)
            theta = orientation * 2 * np.pi / orientations
            size = scales[scale]
            kernel = np.zeros((size, size))
            for i in range(size):
                for j in range(size):
                    x = i - size // 2
                    y = j - size // 2
                    kernel[i, j] = np.exp(- (x ** 2 + y ** 2) / (2 * sigma ** 2)) * (x * np.cos(theta) + y * np.sin(theta))
            filters.append(kernel)
    return filters

def one_dimensional_gaussian(sigma, x, order):
    x = np.array(x)
    var = sigma**2

    # Gaussian Function
    g = (1/np.sqrt(2*np.pi*var))*(np.exp((-1*x*x)/(2*var)))

    if order == 0:
        return g
    elif order == 1:
        return -g*((x)/(var))
    else:
        return g*(((x*x) - var)/(var**2))

def Gaussian(kernel_size, scales):
    var = scales * scales
    shape = (kernel_size,kernel_size)
    n,m = [(i - 1)/2 for i in shape]
    x,y = np.ogrid[-m:m+1,-n:n+1]
    g = (1/np.sqrt(2*np.pi*var))*np.exp( -(x*x + y*y) / (2*var) )
    return g

def LoG(kernel_size, scales):
    var = scales * scales
    shape = (kernel_size,kernel_size)
    n,m = [(i - 1)/2 for i in shape]
    x,y = np.ogrid[-m:m+1,-n:n+1]
    g = (1/np.sqrt(2*np.pi*var))*np.exp( -(x*x + y*y) / (2*var) )
    h = g*((x*x + y*y) - var)/(var**2)
    return h

def generate_derivation_of_gaussian_filter(scale, x_derivative_order, y_derivative_order, pts, kernel_size):
    g_x = one_dimensional_gaussian(scale, pts[0,...], x_derivative_order)
    g_y = one_dimensional_gaussian(3*scale, pts[1,...], y_derivative_order)
    image = g_x*g_y
    image = np.reshape(image,(kernel_size,kernel_size))
    return image

def generate_leung_malik_filters(kernel_size = 49, orientations = 6, version = 'LML'):
    filters = []
    if version == 'LML':
        kernel_scales_derivation_of_gaussian  = np.array([np.sqrt(2), 2, 2*np.sqrt(2)])
    elif version == 'LMS':
        kernel_scales_derivation_of_gaussian  = np.array([1, np.sqrt(2), 2])

    kernel_range  = (kernel_size - 1)/2

    x = [np.arange(-kernel_range,kernel_range+1)]
    y = [np.arange(-kernel_range,kernel_range+1)]

    [x,y] = np.meshgrid(x,y)

    unrotated_points = [x.flatten(), y.flatten()]
    unrotated_points = np.array(unrotated_points)

    for scale in range(len(kernel_scales_derivation_of_gaussian)):
        for orient in range(orientations):
            theta = (np.pi * orient)/orientations
            c = np.cos(theta)
            s = np.sin(theta)

            rot_mat = [[c+0,-s+0],[s+0,c+0]]
            rot_mat = np.array(rot_mat)
            
            rotated_points = np.dot(rot_mat,unrotated_points)

            filter = generate_derivation_of_gaussian_filter(kernel_scales_derivation_of_gaussian[scale], 1, 0, rotated_points, kernel_size)
            filters.append(filter)
            filter = generate_derivation_of_gaussian_filter(kernel_scales_derivation_of_gaussian[scale], 2, 0, rotated_points, kernel_size)
            filters.append(filter)

    if version == 'LML':
        kernel_scales_LoG_Gaussian = np.array([np.sqrt(2), 2, 2*np.sqrt(2), 4])
    elif version == 'LMS':
        kernel_scales_LoG_Gaussian = np.array([1, np.sqrt(2), 2, 2*np.sqrt(2)])    

    for i in range(len(kernel_scales_LoG_Gaussian)):
        filter = LoG(kernel_size, kernel_scales_LoG_Gaussian[i])
        filters.append(filter)

    for i in range(len(kernel_scales_LoG_Gaussian)):
        filter = LoG(kernel_size, 3*kernel_scales_LoG_Gaussian[i])
        filters.append(filter)

    for i in range(len(kernel_scales_LoG_Gaussian)):
        filter   = Gaussian(kernel_size, kernel_scales_LoG_Gaussian[i])
        filters.append(filter)

    return np.array(filters)

def generate_gabor_filters(orientations, scales, spacing):
    filters = []
    for s in range(len(scales)):
        for orientation in range(orientations):
            sigma = 1.5 * (scales[s] + 1)
            theta = orientation * np.pi / orientations
            size = 49
            kernel = np.zeros((size, size))
            gaussian_kernel = Gaussian(size, sigma)
            # A gabor filter is a gaussian kernel function modulated by a sinusoidal plane wave.
            for i in range(size):
                for j in range(size):
                    x = i - size // 2
                    y = j - size // 2
                    kernel[i, j] = gaussian_kernel[i, j] * np.cos(2 * np.pi * spacing[s] * (x * np.cos(theta) + y * np.sin(theta)))
            filters.append(kernel)
    return filters

def generate_half_disc_masks(orientations, scales):
    masks = []
    for scale in scales:
        for orientation in range(orientations):
            # The half-disc masks are simply (pairs of) binary images of half-discs.
            kernel = np.zeros((scale, scale))
            theta = orientation * np.pi / orientations
            for i in range(scale):
                for j in range(scale):
                    x = i - scale // 2
                    y = j - scale // 2
                    if x**2 + y**2 <= (scale//2)**2 and x * np.cos(theta) + y * np.sin(theta) >= 0:
                        kernel[i, j] = 1
            masks.append(kernel)
            kernel_mirror = np.zeros((scale, scale))
            for i in range(scale):
                for j in range(scale):
                    x = i - scale // 2
                    y = j - scale // 2
                    if x**2 + y**2 <= (scale//2)**2 and x * np.cos(theta) + y * np.sin(theta) < 0:
                        kernel_mirror[i, j] = 1
            masks.append(kernel_mirror)
    return masks

def filter2d(image, num ,filter):
    filter_size = filter.shape[0]
    image_size_x = image.shape[0]
    image_size_y = image.shape[1]
    padding_size = filter_size // 2
    padded_image = np.pad(image, padding_size, mode='constant')
    filtered_image = np.zeros_like(image)
    for i in range(image_size_x):
        for j in range(image_size_y):
            roi = padded_image[i:i+filter_size, j:j+filter_size]
            filtered_image[i, j] = np.sum(roi * filter)
    return filtered_image

def filter_image(image, filter_bank):
	filter_image = np.zeros((image.shape[0], image.shape[1], 0))
	for i in range(len(filter_bank)):
		filter = np.array(filter_bank[i])
		filter_map = filter2D(image, -1, filter)
		filter_image = np.dstack((filter_image, filter_map))
	return filter_image

def get_filtered_images(img, dog_filter_bank, lms_filter_bank, gabor_filter_bank):
	dog_filtered_images = filter_image(img, dog_filter_bank)
	lms_filtered_images = filter_image(img, lms_filter_bank)
	gabor_filtered_images = filter_image(img, gabor_filter_bank)
	filtered_images = np.dstack((dog_filtered_images, lms_filtered_images, gabor_filtered_images))
	return filtered_images

def predict_labels(filtered_images, cluster_count, map_type="texton"):
	if map_type == "texton":
		image_height, image_width, image_depth = filtered_images.shape	
		filtered_images = filtered_images.reshape(image_height * image_width, image_depth)
	elif map_type == "brightness":
		image_height, image_width = filtered_images.shape
		filtered_images = filtered_images.reshape(image_height * image_width, 1)
	elif map_type == "color":
		image_height, image_width, _ = filtered_images.shape
		filtered_images = filtered_images.reshape(image_height * image_width, 3)          
	kmeans = KMeans(n_clusters=cluster_count, random_state=0)
	labels = kmeans.fit_predict(filtered_images)
	labels = np.reshape(labels, (image_height, image_width))
	return labels

def save_image(image, base_path, type_of_filter, name):
	check_folder_exists(base_path + type_of_filter)
	cv2.imwrite(
		base_path + type_of_filter + name,
		image,
		[int(cv2.IMWRITE_JPEG_QUALITY), 90],
	)
     
def get_chi_square_dist(image, bins, mask, inverse_mask):
	chi_square_dist = image * 0
	for i in range(bins):
		tmp = np.zeros_like(image)
		tmp[image == i] = 1
		g_i = cv2.filter2D(tmp, -1, mask)
		h_i = cv2.filter2D(tmp, -1, inverse_mask)

		temp_chi_distance = ((g_i - h_i) ** 2) / (g_i + h_i + 0.005)
		chi_square_dist = chi_square_dist + temp_chi_distance
	chi_square_dist = chi_square_dist / 2
	return chi_square_dist
     
def get_gradient_map(image, half_disc_masks, num_bins):
	gradient_map = np.array(image)
	for i in range(0, len(half_disc_masks), 2):
		chi_square_dist = get_chi_square_dist(
			image, num_bins, half_disc_masks[i], half_disc_masks[i + 1]
		)
		gradient_map = np.dstack((gradient_map, chi_square_dist))
	gradients = np.mean(gradient_map, axis=2)
	return gradients


def main():
     
	# Start time
	start_time = time.time()

	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""
	dog_filter_scales = [7, 11, 21, 31]
	dog_orientations = 24
	dog_filter_bank = generate_derivative_of_gaussian_filters(orientations=dog_orientations, scales=dog_filter_scales)
	print("Size of DoG filter bank: ", len(dog_filter_bank))
	check_folder_exists("Results/DoG/")
	plot_images((dog_orientations, len(dog_filter_scales)), dog_filter_bank, x_len=dog_orientations, y_len=len(dog_filter_scales), name="Results/DoG/DoG.png")


	"""
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
	"""
	lm_orientations = 6
	lml_filter_bank = generate_leung_malik_filters(kernel_size = 49, orientations = lm_orientations, version = 'LML')
	print("Size of LM filters filter bank - Large: ", len(lml_filter_bank))
	check_folder_exists("Results/LMF/")
	plot_images((12, 4), lml_filter_bank, x_len=12, y_len=4, name="Results/LMF/LML.png")
     
	lms_filter_bank = generate_leung_malik_filters(kernel_size = 49, orientations = lm_orientations, version = 'LMS')
	print("Size of LM filters filter bank - Small: ", len(lms_filter_bank))
	check_folder_exists("Results/LMF/")
	plot_images((12, 4), lms_filter_bank, x_len=12, y_len=4, name="Results/LMF/LMS.png")
     

	"""
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""
	gabor_orientations = 16
	gabor_spacing = [0.3, 0.2, 0.15, 0.08, 0.05, 0.03]
	gabor_scales = [1, 3, 7, 11, 17, 21]
	gabor_filter_bank  = generate_gabor_filters(orientations=gabor_orientations, scales=gabor_scales, spacing=gabor_spacing)
	print("Size of Gabor filter bank: ", len(gabor_filter_bank))
	check_folder_exists("Results/Gabor/")
	plot_images((gabor_orientations, len(gabor_scales)), gabor_filter_bank, x_len=gabor_orientations, y_len=len(gabor_scales), name="Results/Gabor/Gabor.png")


	"""
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""
	half_disc_scales = [9, 17, 33]	
	half_disc_masks = generate_half_disc_masks(orientations=8, scales=half_disc_scales)
	print("Size of Half-disk masks: ", len(half_disc_masks))
	check_folder_exists("Results/HalfDisk/")
	plot_images((8, 6), half_disc_masks, x_len=8, y_len=6, name="Results/HalfDisk/HDMasks.png")
    

	imagefiles_base_path = "../BSDS500/Images/"
	imagefiles = os.listdir(imagefiles_base_path)
	imagefiles.sort()
	imagefiles = imagefiles
     
	# Time taken to generate filters
	time_after_filters = time.time()
	print("Time taken to generate filters: ", time_after_filters - start_time)
    
	for imagefile in imagefiles:
		image_processing_start_time = time.time()  
		
		plt.axis("off")
		print("*"*20)
		print("Processing Image: ", imagefile)
		# Read image
		img_rgb = cv2.imread(imagefiles_base_path + imagefile)
		img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
		img_size = img.shape
		
		"""
		Generate Texton Map
		Filter image using oriented gaussian filter bank
		"""
		filtered_images = get_filtered_images(img, dog_filter_bank, lms_filter_bank, gabor_filter_bank)
		print("Number of filtered images: ", filtered_images.shape[2])

		"""
		Generate texture ID's using K-means clustering
		Display texton map and save image as TextonMap_ImageName.png,
		use command "cv2.imwrite('...)"
		"""
		texton_map = predict_labels(filtered_images, cluster_count=64, map_type="texton")
		texton_map = 3 * texton_map # To make the texton map visible
		cmap = plt.get_cmap('gist_rainbow')
		texton_map_ = cmap(texton_map) * 255
		save_image(texton_map_, "Results/", "TextonMap/", imagefile)
        			
		"""
		Generate Texton Gradient (Tg)
		Perform Chi-square calculation on Texton Map
		Display Tg and save image as Tg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
		texton_gradient = get_gradient_map(texton_map, half_disc_masks, num_bins=64)
		texton_gradient1 = texton_gradient
		texton_gradient1 = cv2.convertScaleAbs(texton_gradient1)
		texton_gradient1 = cv2.cvtColor(texton_gradient1, cv2.COLOR_GRAY2BGR)
		save_image(texton_gradient1, "Results/", "TextonGradient/", "TextonGradient_" + imagefile)

		print("Texton Gradient processing done, it's shape: ", texton_gradient.shape)
          
		"""
		Generate Brightness Map
		Perform brightness binning 
		"""
		image_for_brightness = np.array(img)
		brightness_map = predict_labels(image_for_brightness, cluster_count=16, map_type="brightness")
		brightness_map_ = 16 * brightness_map # To make the brightness map visible
		brightness_map_ = cmap(brightness_map_) * 255
		save_image(brightness_map_, "Results/", "BrightnessMap/", "BrightnessMap_" + imagefile)
       
		"""
		Generate Brightness Gradient (Bg)
		Perform Chi-square calculation on Brightness Map
		Display Bg and save image as Bg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
		brightness_gradient = get_gradient_map(brightness_map, half_disc_masks, num_bins=16)
		brightness_gradient1 = brightness_gradient
		brightness_gradient1 = cv2.convertScaleAbs(brightness_gradient1)	
		brightness_gradient1 = cv2.cvtColor(brightness_gradient1, cv2.COLOR_GRAY2BGR)
		save_image(brightness_gradient1, "Results/", "BrightnessGradient/", "BrightnessGradient_" + imagefile)
		print("Brightness Gradient processing done, it's shape: ", brightness_gradient.shape)

		"""
		Generate Color Map
		Perform color binning or clustering
		"""
		image_for_color = np.array(img_rgb)
		color_map = predict_labels(image_for_color, cluster_count=16, map_type="color")
		color_map_ = 16 * color_map
		color_map_ = cmap(color_map_) * 255
		save_image(color_map_, "Results/", "ColorMap/", "ColorMap_" + imagefile)

		"""
		Generate Color Gradient (Cg)
		Perform Chi-square calculation on Color Map
		Display Cg and save image as Cg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
		color_gradient = get_gradient_map(color_map, half_disc_masks, num_bins=16)
		color_gradient1 = color_gradient
		color_gradient1 = cv2.convertScaleAbs(color_gradient1)
		color_gradient1 = cv2.cvtColor(color_gradient1, cv2.COLOR_GRAY2BGR)
		save_image(color_gradient1, "Results/", "ColorGradient/", "ColorGradient_" + imagefile)
		print("Color Gradient processing done, it's shape: ", color_gradient.shape)

		"""
		Read Sobel Baseline
		use command "cv2.imread(...)"
		"""
		sobel_baseline = cv2.imread("../BSDS500/SobelBaseline/" + imagefile.split(".")[0] + ".png", 0)

		"""
		Read Canny Baseline
		use command "cv2.imread(...)"
		"""
		canny_baseline = cv2.imread("../BSDS500/CannyBaseline/" + imagefile.split(".")[0] + ".png", 0)
          
		print("Sobel Baseline Shape: ", sobel_baseline.shape)
		print("Canny Baseline Shape: ", canny_baseline.shape)

		"""
		Combine responses to get pb-lite output
		Display PbLite and save image as PbLite_ImageName.png
		use command "cv2.imwrite(...)"
		"""
		PbLite = (1/3) * (texton_gradient + brightness_gradient + color_gradient) * (0.5 * sobel_baseline + 0.5 * canny_baseline)
		PbLite = cv2.normalize(PbLite, None, 0, 255, cv2.NORM_MINMAX)
		save_image(PbLite, "Results/", "PbLite/", "PbLite_" + imagefile)	
		plt.axis("off")
		plt.imshow(PbLite, cmap="gray")
		check_folder_exists("Results/PbLite/")
		plt.savefig("Results/PbLite/PbLite_plt" + imagefile, bbox_inches="tight", pad_inches=0)	
		plt.close()
          
		# Time taken to process one image
		image_processing_end_time = time.time()
		print("Time taken to process image: ", image_processing_end_time - image_processing_start_time)
          
	# Total time taken
	end_time = time.time()
	print("*"*20)
	print("Total time taken to process all images: ", end_time - time_after_filters)
          
    
if __name__ == '__main__':
    main()
 


