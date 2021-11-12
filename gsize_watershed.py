"""
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_watershed/py_watershed.html
This code performs grain size distribution analysis and dumps results into a csv file.
It uses watershed segmentation for better segmentation.
Compare results to regular segmentation. 
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import measure, color, io

fig = plt.figure()

fig.add_subplot(4, 4, 1)
img1 = cv2.imread("input/12.jpg")
plt.imshow(img1)
plt.title("Original")

fig.add_subplot(4, 4, 2)
img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title("cvtColor: COLOR_BGR2GRAY")

pixels_to_um = 0.5 # 1 pixel = 500 nm (got this from the metadata of original image)

#Threshold image to binary using OTSU. ALl thresholded pixels will be set to 255
# ret1, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret1, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)


gray = cv2.medianBlur(img,5)
fig.add_subplot(4, 4, 3)
plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
plt.title("medianBlur")

edged = cv2.Canny(gray, 50, 150)
fig.add_subplot(4, 4, 4)
plt.imshow(edged, cmap='gray', vmin=0, vmax=255)
plt.title("Canny")

kernel_d = np.ones((10,10),np.uint8)
kernel_e = np.ones((10,10),np.uint8)
edged_d = cv2.dilate(edged, kernel_d, iterations=1)
fig.add_subplot(4, 4, 5)
plt.imshow(edged_d, cmap='gray', vmin=0, vmax=255)
plt.title("dilate")

edged_e = cv2.erode(edged_d, kernel_e, iterations=1)
fig.add_subplot(4, 4, 6)
plt.imshow(edged_e, cmap='gray', vmin=0, vmax=255)
plt.title("erode")


# Morphological operations to remove small noise - opening
#To remove holes we can use closing
kernel = np.ones((3,3),np.uint8)
# opening = cv2.morphologyEx(edged,cv2.MORPH_OPEN,kernel, iterations = 2)

from skimage.segmentation import clear_border
opening = clear_border(edged_e) #Remove edge touching grains
# #Check the total regions found before and after applying this. 
fig.add_subplot(4, 4, 7)
plt.imshow(opening, cmap='gray', vmin=0, vmax=255)
plt.title("opening")

#Now we know that the regions at the center of cells is for sure cells
#The region far away is background.
#We need to extract sure regions. For that we can use erode. 
#But we have cells touching, so erode alone will not work. 
#To separate touching objects, the best approach would be distance transform and then thresholding.

# let us start by identifying sure background area
# dilating pixes a few times increases cell boundary to background. 
# This way whatever is remaining for sure will be background. 
#The area in between sure background and foreground is our ambiguous area. 
#Watershed should find this area for us. 
edged_e=cv2.bitwise_not(edged_e)
sure_bg = cv2.dilate(edged_e,kernel,iterations=2)
fig.add_subplot(4, 4, 8)
plt.imshow(sure_bg, cmap='gray', vmin=0, vmax=255)
plt.title("sure_bg")
# plt.show()


# Finding sure foreground area using distance transform and thresholding
#intensities of the points inside the foreground regions are changed to 
#distance their respective distances from the closest 0 value (boundary).
#https://www.tutorialspoint.com/opencv/opencv_distance_transformation.htm
dist_transform = cv2.distanceTransform(edged_e,cv2.DIST_L2,3)
fig.add_subplot(4, 4, 9)
plt.imshow(dist_transform, cmap='gray', vmin=0, vmax=255)
plt.title("dist_transform")

#Let us threshold the dist transform by 20% its max value.
#print(dist_transform.max()) gives about 21.9
ret2, sure_fg = cv2.threshold(dist_transform,0.2*dist_transform.max(),255,0)
fig.add_subplot(4, 4, 10)
plt.imshow(sure_fg, cmap='gray', vmin=0, vmax=255)
plt.title("sure_fg")


#0.2* max value seems to separate the cells well.
#High value like 0.5 will not recognize some grain boundaries.

# Unknown ambiguous region is nothing but bkground - foreground
sure_fg = np.uint8(sure_fg)

unknown = cv2.subtract(sure_bg,sure_fg)
fig.add_subplot(4, 4, 11)
plt.imshow(unknown, cmap='gray', vmin=0, vmax=255)
plt.title("unknown")


#Now we create a marker and label the regions inside. 
# For sure regions, both foreground and background will be labeled with positive numbers.
# Unknown regions will be labeled 0. 
#For markers let us use ConnectedComponents. 
ret3, markers = cv2.connectedComponents(sure_fg)
fig.add_subplot(4, 4, 12)
plt.imshow(markers, cmap='gray', vmin=0, vmax=255)
plt.title("markers")
plt.show()

#One problem rightnow is that the entire background pixels is given value 0.
#This means watershed considers this region as unknown.
#So let us add 10 to all labels so that sure background is not 0, but 10
markers = markers+10

# Now, mark the region of unknown with zero
markers[unknown==255] = 0
#plt.imshow(markers, cmap='jet')   #Look at the 3 distinct regions.

#Now we are ready for watershed filling. 
markers = cv2.watershed(img1,markers)
#The boundary region will be marked -1
#https://docs.opencv.org/3.3.1/d7/d1b/group__imgproc__misc.html#ga3267243e4d3f95165d55a618c65ac6e1


#Let us color boundaries in yellow. OpenCv assigns boundaries to -1 after watershed.
img1[markers == -1] = [0,255,255]  

img2 = color.label2rgb(markers, bg_label=0)

cv2.imshow('Overlay on original image', img1)
cv2.imshow('Colored Grains', img2)
cv2.waitKey(0)

#Now, time to extract properties of detected cells
# regionprops function in skimage measure module calculates useful parameters for each object.
regions = measure.regionprops(markers, intensity_image=img)

#Can print various parameters for all objects
#for prop in regions:
#    print('Label: {} Area: {}'.format(prop.label, prop.area))

#Best way is to output all properties to a csv file
#Let us pick which ones we want to export. 

propList = ['Area',
            'equivalent_diameter', #Added... verify if it works
            'orientation', #Added, verify if it works. Angle btwn x-axis and major axis.
            'MajorAxisLength',
            'MinorAxisLength',
            'Perimeter',
            'MinIntensity',
            'MeanIntensity',
            'MaxIntensity']    
    

output_file = open('image_measurements.csv', 'w')
output_file.write('Grain #' + "," + "," + ",".join(propList) + '\n') #join strings in array by commas, 
#First cell to print grain number
#Second cell blank as we will not print Label column

grain_number = 1
for region_props in regions:
    output_file.write(str(grain_number) + ',')
    #output cluster properties to the excel file
#    output_file.write(str(region_props['Label']))
    for i,prop in enumerate(propList):
        if(prop == 'Area'): 
            to_print = region_props[prop]*pixels_to_um**2   #Convert pixel square to um square
        elif(prop == 'orientation'): 
            to_print = region_props[prop]*57.2958  #Convert to degrees from radians
        elif(prop.find('Intensity') < 0):          # Any prop without Intensity in its name
            to_print = region_props[prop]*pixels_to_um
        else: 
            to_print = region_props[prop]     #Reamining props, basically the ones with Intensity in its name
        output_file.write(',' + str(to_print))
    output_file.write('\n')
    grain_number += 1
    
output_file.close()   #Closes the file, otherwise it would be read only. 