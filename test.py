import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line


# convert coordinates frim txt to csv 
# we need to extract geometric primitives from those lines.
# the result mask for primitives in the folders should be of uint8 color type and have three channel [w, h, 3], dtype = uint8, with 0 to 255 unique value
# Tool mask is a binary mask with 0 and 255 values, dtype = uint8 in the folders 
# for primitives, the input to the model should be fload64 with values range between 0-1 (convert from 0, 255 to float range between 0 and 1)
# for tool mask, input to the model will be converted to float64 values between 0 -1 
import pandas as pd
import numpy as np
import cv2
import glob 
import os 

# text file 
'''
x ...... y
x1 ......y1 ..... edge line 1 
x2 ......y2 ..... edge line 2
x3 ......y3 ..... mid line 
x4 ......y4 ..... tip point 
'''
# convert text files generated from imageJ to csv

# print(pd.__version__)
def edgeline1(read_file):
	# Edge line 
	edge1_p1_x = int(read_file['X'][0])
	edge1_p1_y = int(read_file['Y'][0])
	edge1_p2_x = int(read_file['X'][1])
	edge1_p2_y = int(read_file['Y'][1])

	edgeline1_point1 = (edge1_p1_x, edge1_p1_y)
	edgeline1_point2 = (edge1_p2_x, edge1_p2_y)

	edgeline1 = (edgeline1_point1, edgeline1_point2)

	return edgeline1  

def edgeline2(read_file):
	# Edge line 
	edge2_p1_x = int(read_file['X'][2])
	edge2_p1_y = int(read_file['Y'][2])
	edge2_p2_x = int(read_file['X'][3])
	edge2_p2_y = int(read_file['Y'][3])

	edgeline2_point1 = (edge2_p1_x, edge2_p1_y )
	edgeline2_point2 = (edge2_p2_x, edge2_p2_y ) 
	edgeline2 = (edgeline2_point1, edgeline2_point2)

	return edgeline2  

def midline(read_file): 
	# Edge line 1 
	edge1_p1_x = int(read_file['X'][0])
	edge1_p1_y = int(read_file['Y'][0])
	edge1_p2_x = int(read_file['X'][1])
	edge1_p2_y = int(read_file['Y'][1])

	# Edge line 2 
	edge2_p1_x = int(read_file['X'][2])
	edge2_p1_y = int(read_file['Y'][2])
	edge2_p2_x = int(read_file['X'][3])
	edge2_p2_y = int(read_file['Y'][3])
	# Mid line 
	midline_p1_x = int((edge1_p1_x + edge2_p1_x) /2)
	midline_p1_y = int((edge1_p1_y + edge2_p1_y) /2)
	midline_p2_x = int((edge1_p2_x + edge2_p2_x) /2)
	midline_p2_y = int((edge1_p2_y + edge2_p2_y) /2)

	midline_point1 = (midline_p1_x, midline_p1_y)
	midline_point2 = (midline_p2_x, midline_p2_y)
	midline = (midline_point1, midline_point2 )
	return midline

def tipoint(read_file):
	tippoint_x = int(read_file['X'][4])
	tippoint_y = int(read_file['Y'][4])
	tippoint = (tippoint_x, tippoint_y)
	return tippoint


# def AnnotateLines(linepoints, path, filename,  w=1270, h=820, linethickness = 15):
# 	image = np.zeros((h,w), dtype=np.uint8)
# 	image = cv2.line(image, linepoints[0], linepoints[1], 255, linethickness)
# 	image = cv2.distanceTransform(image,cv2.DIST_L2,5)
# 	image = cv2.normalize(image, image, 0, 1.0, cv2.NORM_MINMAX)
# 	image = np.uint8(image*255)
# 	image_copy = np.zeros((h,w,3), dtype=np.uint8)
# 	image_copy[:,:,0] = image
# 	image_copy[:,:,1] = image
# 	image_copy[:,:,2] = image
# 	pathToSave = path + filename + '.png'
# 	cv2.imwrite(pathToSave, image)
# 	#cv2.imshow('ss',image)
# 	#cv2.waitKey(200)

# def AnnotateTipPoint(tippoint, path, filename,  w=1270, h=820 ):
# 	image = np.zeros((h,w), dtype=np.uint8)
# 	image = cv2.circle(image, tippoint, radius=0, color=255, thickness=100)
# 	image = cv2.distanceTransform(image,cv2.DIST_L2,5)
# 	image = cv2.normalize(image, image, 0, 1.0, cv2.NORM_MINMAX)
# 	image = np.uint8(image*255)
# 	image_copy = np.zeros((h,w,3), dtype=np.uint8)
# 	image_copy[:,:,0] = image
# 	image_copy[:,:,1] = image
# 	image_copy[:,:,2] = image
# 	pathToSave = path + filename + '.png'
# 	cv2.imwrite(pathToSave, image)
# 	#cv2.imshow('ss',image)
# 	#cv2.waitKey(200)

def extractFrame():

	vidcap = cv2.VideoCapture('/home/bao/Downloads/2021-09-29_130434_VID014.mp4')
	success,image = vidcap.read()
	count = 0
	while success:
	  cv2.imwrite("/home/bao/Downloads/data_05_Oct/frame%d.jpg" % count, image)     # save frame as JPEG file      
	  success,image = vidcap.read()
	  print('Read a new frame: ', success)
	  count += 1

def drawMidLine(linepoints, path, filename,  w=1280, h=720, linethickness = 1):

	mask = np.zeros((h,w),dtype=np.uint8)
	mask = cv2.line(mask, linepoints[0], linepoints[1], 255, linethickness)
	#-----------------------------Classic straight-line Hough transform----------------------------
	h, theta, d = hough_line(mask)        
	for _, angle, dist in zip(*hough_line_peaks(h, theta, d, min_distance=9, min_angle=10, threshold=0.5*h.max(), num_peaks=1)):
	    y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
	    y1 = (dist - mask.shape[1] * np.cos(angle)) / np.sin(angle)
	x0P = 0
	y0P = int(y0)
	x1P = mask.shape[1]
	y1P = int(y1)
	#-----------------------------Approximate Line-------------------------------------------------
	midLine=np.zeros_like(mask, dtype=np.uint8)

	cv2.line(midLine,(x0P,y0P),(x1P,y1P), 255, linethickness)
	pathToSave = path + filename + '_midLine.png'
	cv2.imwrite(pathToSave, midLine)

def drawEdgeLine1(linepoints, path, filename,  w=1280, h=720, linethickness = 1):

	mask = np.zeros((h,w),dtype=np.uint8)
	mask = cv2.line(mask, linepoints[0], linepoints[1], 255, linethickness)
	#-----------------------------Classic straight-line Hough transform----------------------------
	h, theta, d = hough_line(mask)        
	for _, angle, dist in zip(*hough_line_peaks(h, theta, d, min_distance=9, min_angle=10, threshold=0.5*h.max(), num_peaks=1)):
	    y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
	    y1 = (dist - mask.shape[1] * np.cos(angle)) / np.sin(angle)
	x0P = 0
	y0P = int(y0)
	x1P = mask.shape[1]
	y1P = int(y1)
	#-----------------------------Approximate Line-------------------------------------------------
	midLine=np.zeros_like(mask, dtype=np.uint8)

	cv2.line(midLine,(x0P,y0P),(x1P,y1P), 255, linethickness)
	pathToSave = path + filename + '_edgeLine_Line_1.png'
	cv2.imwrite(pathToSave, midLine)
	return midLine

def drawEdgeLine2(linepoints, path, filename,  w=1280, h=720, linethickness = 1):

	mask = np.zeros((h,w),dtype=np.uint8)
	mask = cv2.line(mask, linepoints[0], linepoints[1], 255, linethickness)
	#-----------------------------Classic straight-line Hough transform----------------------------
	h, theta, d = hough_line(mask)        
	for _, angle, dist in zip(*hough_line_peaks(h, theta, d, min_distance=9, min_angle=10, threshold=0.5*h.max(), num_peaks=1)):
	    y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
	    y1 = (dist - mask.shape[1] * np.cos(angle)) / np.sin(angle)
	x0P = 0
	y0P = int(y0)
	x1P = mask.shape[1]
	y1P = int(y1)
	#-----------------------------Approximate Line-------------------------------------------------
	midLine=np.zeros_like(mask, dtype=np.uint8)

	cv2.line(midLine,(x0P,y0P),(x1P,y1P), 255, linethickness)
	pathToSave = path + filename + '_edgeLine_Line_2.png'
	cv2.imwrite(pathToSave, midLine)
	return midLine

def drawTipPoint(tippoint, path, filename,  w=1280, h=720):

	mask = np.zeros((h,w),dtype=np.uint8)
	mask[tippoint] = 255
	pathToSave = path + filename + '_tipPoint_Approximated.png'
	cv2.imwrite(pathToSave, mask)

# path to xy coordinates csv file 
coordscsv =glob.glob('/home/bao/Downloads/trocar_estimation_adrien/liver_tools_trocars_compressed/undistorted/T4/coordinates/*.csv')


# Iterate all csv file 
for path in coordscsv:
	filename = os.path.basename(path)
	filename = os.path.splitext(filename)[0]
	read_file = pd.read_csv(path)
	width = 1280
	height = 720
	edgeline1_points = edgeline1(read_file)
	pathToEdgeLine1 = '/home/bao/Downloads/trocar_estimation_adrien/liver_tools_trocars_compressed/results/T4/'
	print(pathToEdgeLine1)
	ed1 = drawEdgeLine1(edgeline1_points, pathToEdgeLine1, filename ,width, height, 1 )

	edgeline2_points = edgeline2(read_file)
	pathToEdgeLine2 = pathToEdgeLine1
	ed2 = drawEdgeLine2(edgeline2_points, pathToEdgeLine2, filename ,width, height, 1 )

	edgeline2_midline = midline(read_file)
	pathToMidLine = pathToEdgeLine1
	drawMidLine(edgeline2_midline , pathToMidLine, filename, width, height, 1 )

	tippoint = tipoint(read_file)
	pathToTip = pathToEdgeLine1
	drawTipPoint(tippoint, pathToTip, filename,  width, height)

	ed = ed1+ed2
	pathToSave = pathToEdgeLine1 + filename + '_edgeLine.png'
	cv2.imwrite(pathToSave, ed)