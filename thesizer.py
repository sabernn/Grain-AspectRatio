# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import matplotlib.pyplot as plt
import argparse
import imutils
import cv2
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True,
	help="width of the left-most object in the image (in inches)")
# args = vars(ap.parse_args())
args = {'image': 'input/9.jpg','width': 1.0}
# print(args)



fig = plt.figure()
fig.add_subplot(2, 3, 1)
# load the image, convert it to grayscale, and blur it slightly
# fig.add_subplot(2, 3, 1)
image = cv2.imread(args["image"])
plt.imshow(image)
plt.title("Original")

fig.add_subplot(2, 3, 2)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray)
plt.title("cvtColor")

fig.add_subplot(2, 3, 3)
gray = cv2.medianBlur(gray,5)
plt.imshow(gray)
plt.title("medianBlur")
# gray = cv2.GaussianBlur(gray, (3, 3), 0)
# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
kernel_d = np.ones((8,8),np.uint8)
kernel_e = np.ones((8,8),np.uint8)

fig.add_subplot(2, 3, 4)
edged = cv2.Canny(gray, 50, 100)
plt.imshow(edged)
plt.title("Canny")

fig.add_subplot(2, 3, 5)
edged = cv2.dilate(edged, kernel_d, iterations=1)
plt.imshow(edged)
plt.title("dialate")

fig.add_subplot(2, 3, 6)
edged = cv2.erode(edged, kernel_e, iterations=1)
plt.imshow(edged)
plt.title("erode")

plt.show()

# fig1 = plt.figure()
# fig1.add_subplot(1, 2, 1)
# image = cv2.imread(args["image"])
# plt.imshow(image)
# plt.title("Original")

# fig1.add_subplot(1, 2, 2)
# image_otsu = cv2.threshold(image[:,:,0], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# plt.imshow(image_otsu)
# plt.title("OTSU")

# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_NONE)
cnts = imutils.grab_contours(cnts)
# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None

print(f"Number of contours: {len(cnts)}")

# loop over the contours individually
for c in cnts:
	# if the contour is not sufficiently large, ignore it
	if cv2.contourArea(c) < 100:
		continue
	# compute the rotated bounding box of the contour
	orig = image.copy()
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")
	# order the points in the contour such that they appear
	# in top-left, top-right, bottom-right, and bottom-left
	# order, then draw the outline of the rotated bounding
	# box
	box = perspective.order_points(box)
	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
	# loop over the original points and draw them
	for (x, y) in box:
		cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)


    # unpack the ordered bounding box, then compute the midpoint
	# between the top-left and top-right coordinates, followed by
	# the midpoint between bottom-left and bottom-right coordinates
	(tl, tr, br, bl) = box
	(tltrX, tltrY) = midpoint(tl, tr)
	(blbrX, blbrY) = midpoint(bl, br)
	# compute the midpoint between the top-left and top-right points,
	# followed by the midpoint between the top-righ and bottom-right
	(tlblX, tlblY) = midpoint(tl, bl)
	(trbrX, trbrY) = midpoint(tr, br)
	# draw the midpoints on the image
	cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
	# draw lines between the midpoints
	cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
		(255, 0, 255), 2)
	cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
		(255, 0, 255), 2)

	# compute the Euclidean distance between the midpoints
	dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
	dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
	# if the pixels per metric has not been initialized, then
	# compute it as the ratio of pixels to supplied metric
	# (in this case, inches)
	if pixelsPerMetric is None:
		pixelsPerMetric = dB / args["width"]


	# compute the size of the object
	dimA = dA / pixelsPerMetric
	dimB = dB / pixelsPerMetric
	# draw the object sizes on the image
	cv2.putText(orig, "{:.1f}in".format(dimA),
		(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)
	cv2.putText(orig, "{:.1f}in".format(dimB),
		(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)
	# show the output image
	cv2.imshow('ImageWindow',orig)
	cv2.waitKey(0)
