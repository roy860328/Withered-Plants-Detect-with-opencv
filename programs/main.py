# cd /Users/roy/Downloads/NCU/影像處理/
# source /Users/roy/Downloads/NCU/影像處理/imgp/bin/activate

import cv2
import numpy as np


def detect_circles(gray, img_circle):
	# detect circles in the image
	# circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 400)
	circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1, 300,
						param1=50,param2=30,minRadius=200,maxRadius=250)
	
	# ensure at least some circles were found
	if circles is not None:
		# convert the (x, y) coordinates and radius of the circles to integers
		circles = np.round(circles[0, :]).astype("int")

		# loop over the (x, y) coordinates and radius of the circles
		for (x, y, r) in circles:
			# draw the circle in the output image, then draw a rectangle
			# corresponding to the center of the circle
			cv2.circle(img_circle, (x, y), r, (0, 255, 0), 4)
			cv2.rectangle(img_circle, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
	return img_circle
	# show the circle image

# Gray Img Remove Background (Not Very Well，不知意義、效用)
# Watershed Algorithm
def segmentate_image(gray):
	ret, img_segmentate = cv2.threshold(gray,0,255, 12)
	# print(gray, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	return img_segmentate

# Video Background Remove
def subtract_Background():
	cap = cv2.VideoCapture('6.MP4')

	fgbg = cv2.createBackgroundSubtractorMOG2()

	while(1):
		ret, frame = cap.read()

		fgmask = fgbg.apply(frame)

		cv2.imshow('frame',fgmask)
		k = cv2.waitKey(30) & 0xff
		if k == 27:
			break

	cap.release()
	cv2.destroyAllWindows()

# Image Background Remove
def clip(img):
	#== Parameters =======================================================================
	BLUR = 21
	CANNY_THRESH_1 = 10
	CANNY_THRESH_2 = 200
	MASK_DILATE_ITER = 10
	MASK_ERODE_ITER = 10
	MASK_COLOR = (0.0,0.0,0.0) # In BGR format


	#== Processing =======================================================================

	# img = cv2.resize(img, (350, 450))
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	#-- Edge detection -------------------------------------------------------------------
	edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
	# 區塊放大
	edges = cv2.dilate(edges, None)
	# 區塊縮小
	edges = cv2.erode(edges, None)
	# cv2.imshow('result', edges)								   # Display
	# cv2.waitKey(0)

	#-- Find contours in edges, sort by area ---------------------------------------------
	contour_info = []
	im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)


	for c in contours:
		contour_info.append((
			c,
			cv2.isContourConvex(c),
			cv2.contourArea(c),
		))

	contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
	max_contour = contour_info[0]	# 只圈選最大輪廓

	#-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
	# Mask is black, polygon is white
	mask = np.zeros(edges.shape)
	# cv2.fillConvexPoly(mask, max_contour[0], (255))	# 為mask加上輪廓

	for i in range(1, len(contour_info)):
		# img = cv2.imread('test1.jpg')
		# img = cv2.resize(img, (550, 650))
		cv2.fillConvexPoly(mask, contour_info[i][0], (255))	# 為mask加上輪廓
		# cv2.imshow('result', mask)
		# cv2.waitKey(0)
	#-- Smooth mask, then blur it --------------------------------------------------------
	mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
	mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
	mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)		# 模糊化去噪
	mask_stack = np.dstack([mask]*3)	# Create 3-channel alpha mask
	
	#-- Blend masked img into MASK_COLOR background --------------------------------------
	mask_stack  = mask_stack.astype('float32') / 255.0		  # Use float matrices, 
	img	= img.astype('float32') / 255.0				 #  for easy blending

	masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR) # Blend
	masked = (masked * 255).astype('uint8')					 # Convert back to 8-bit 
	mask_stack = (mask_stack * 255).astype('uint8')
	img = (img * 255).astype('uint8')
	for i in range(1, len(contour_info)):
		cv2.drawContours(img, contour_info[i][0], -1, (0,255,0), 3)	# 為原始圖片加上輪廓

	cv2.imshow('result', np.hstack([img, masked, mask_stack]))							# Display
	cv2.waitKey(0)

	# cv2.imwrite('result/result1.jpg', np.hstack([img, masked, mask_stack]) )		   # Save

	return masked
def ori_clip():
	#== Parameters =======================================================================
	BLUR = 21
	CANNY_THRESH_1 = 10
	CANNY_THRESH_2 = 200
	MASK_DILATE_ITER = 10
	MASK_ERODE_ITER = 10
	MASK_COLOR = (0.0,0.0,0.0) # In BGR format


	#== Processing =======================================================================

	#-- Read image -----------------------------------------------------------------------
	img = cv2.imread('test1.jpg')
	img = cv2.resize(img, (550, 650))
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	#-- Edge detection -------------------------------------------------------------------
	edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
	edges = cv2.dilate(edges, None)
	edges = cv2.erode(edges, None)

	#-- Find contours in edges, sort by area ---------------------------------------------
	contour_info = []
	_, contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

	for c in contours:
		contour_info.append((
			c,
			cv2.isContourConvex(c),
			cv2.contourArea(c),
		))
	contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
	max_contour = contour_info[0]

	#-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
	# Mask is black, polygon is white
	mask = np.zeros(edges.shape)
	cv2.fillConvexPoly(mask, max_contour[0], (255))

	#-- Smooth mask, then blur it --------------------------------------------------------
	mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
	mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
	mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
	mask_stack = np.dstack([mask]*3)	# Create 3-channel alpha mask

	#-- Blend masked img into MASK_COLOR background --------------------------------------
	mask_stack  = mask_stack.astype('float32') / 255.0		  # Use float matrices, 
	img		 = img.astype('float32') / 255.0				 #  for easy blending

	masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR) # Blend
	masked = (masked * 255).astype('uint8')					 # Convert back to 8-bit 

	cv2.imshow('img', masked)								   # Display
	cv2.waitKey()

	#cv2.imwrite('C:/Temp/person-masked.jpg', masked)		   # Save

def detect_color(img):
	# img = cv2.imread('test1.jpg')   # test1.jpg  75.bmp
	# img = cv2.resize(img, (550, 650))

	# [G, B, R] https://www.rapidtables.com/convert/color/rgb-to-hex.html
	boundaries = [
		# ([15, 100, 20], [25, 255, 255]),
		([70, 50, 100], [110, 255, 180]),
	]
	# loop over the boundaries
	for (lower, upper) in boundaries:
		# create NumPy arrays from the boundaries
		lower = np.array(lower, dtype = "uint8")
		upper = np.array(upper, dtype = "uint8")
	 
		# find the colors within the specified boundaries and apply
		# the mask
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(hsv,lower, upper )	
		# mask = cv2.inRange(img, lower, upper) # 找區間，黑：F 白：T
		output = cv2.bitwise_and(img, img, mask = mask)		# and or not xor
	 
		# show the images
		mask = np.dstack([mask]*3)
		cv2.imshow("images", np.hstack([img, output]))
		cv2.waitKey(0)
	cv2.imwrite('result/detect_color/test2.jpg', np.hstack([img, output]) )		   # Save

def detect_Color_HSV(img, rectMergeSize, rectSize):
	lowerBound=np.array([15, 100, 20])
	upperBound=np.array([25, 255, 255])

	# lowerBound=np.array([33,80,40])
	# upperBound=np.array([102,255,255])

	kernelOpen=np.ones((5,5))
	kernelClose=np.ones((20,20))

	# img = cv2.imread('test1.jpg')
	# img = cv2.imread('asdf.jpeg')
	# img = cv2.imread('green.png')
	img = cv2.resize(img, (550, 650))

	#convert BGR to HSV
	# imgHSV= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	# create the Mask
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv,lowerBound, upperBound )	

	## 
	kernel = np.ones((5,5),np.uint8)
	mask = cv2.dilate(mask,kernel,iterations = 1)
	mask = cv2.erode(mask,kernel,iterations = 1)

	
	#morphology https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
	maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
	maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)

	maskFinal=maskClose
	_, conts, h =cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	
	# cv2.drawContours(img,conts,-1,(255,0,0),3)

	# for i in range(len(conts)):
	# 	x,y,w,h=cv2.boundingRect(conts[i])
	# 	print(x,y,w,h)
	# 	cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255), 2)
	# 	cv2.putText(img, str(i+1),(x,y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255))
	label = []
	for i in range(len(conts)):
		x,y,w,h=cv2.boundingRect(conts[i])
		label.append([x,y,w,h])
	print(label, len(label))

	for i in range(len(label)):
		for j in range(len(label)):
			if i == j or label[j] == [0,0,0,0]:
				continue
			if (abs(label[i][0]-label[j][0]) < rectMergeSize) or (abs(label[i][0]-(label[j][0]+label[j][2])) < rectMergeSize) or \
				(abs((label[i][0]+label[i][2])-label[j][0]) < rectMergeSize) or (abs((label[i][0]+label[i][2])-(label[j][0]+label[j][2])) < rectMergeSize):
				if (abs(label[i][1]-label[j][1]) < rectMergeSize) or (abs(label[i][1]-(label[j][1]+label[j][3])) < rectMergeSize) or \
					(abs((label[i][1]+label[i][3])-label[j][1]) < rectMergeSize) or (abs((label[i][1]+label[i][3])-(label[j][1]+label[j][3])) < rectMergeSize):
					x = int(label[i][0]);y = int(label[i][1]);
					print(label[i], label[j])
					if (label[i][0] > label[j][0]):
						x = int(label[j][0])
						if (label[i][0]+label[i][2] < label[j][0]+label[j][2]):
							label[i][2] = int((label[j][0]+label[j][2])-label[j][0])
						else:
							label[i][2] = int(label[i][0]+label[i][2]-label[j][0])
					else:
						if (label[i][0]+label[i][2] < label[j][0]+label[j][2]):
							label[i][2] = int((label[j][0]+label[j][2])-label[i][0])
						else:
							label[i][2] = int(label[i][0]+label[i][2]-label[i][0])
					label[i][0] = x

					if (label[i][1] > label[j][1]):
						y = int(label[j][1])
						if (label[i][1]+label[i][3] < label[j][1]+label[j][3]):
							label[i][3] = int((label[j][1]+label[j][3])-label[j][1])
						else:
							label[i][3] = int(label[i][1]+label[i][3]-label[j][1])
					else:
						if (label[i][1]+label[i][3] < label[j][1]+label[j][3]):
							label[i][3] = int((label[j][1]+label[j][3])-label[i][1])
						else:
							label[i][3] = int(label[i][1]+label[i][3]-label[i][1])
					label[i][1] = y

					print(label[i])	
					label[j] = [0,0,0,0]
	print(label, len(label))
	i = 0
	for x,y,w,h in label:
		if ( w*h > rectSize):
			cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255), 2)
			cv2.putText(img, str(i+1),(x,y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255))
			i += 1

	maskClose = np.dstack([maskClose]*3)
	mask = np.dstack([mask]*3)
	cv2.imshow("images", np.hstack([img, mask, maskClose]))
	cv2.waitKey(0)
	ori = cv2.imread('test4.jpg')
	ori = cv2.resize(ori, (550, 650))
	cv2.imwrite('result/result4.jpg', np.hstack([ori, img, maskClose]) )	

if __name__ == '__main__':
	img = cv2.imread('test4.jpg')
	img = cv2.resize(img, (550, 650))

	clip_img = clip(img)
	detect_Color_HSV(clip_img, 20 ,400)

	### image init
	ori = cv2.imread("test1.jpg", cv2.IMREAD_COLOR)  # cv2.IMREAD_COLOR = 1
	ori = cv2.resize(ori, (540, 660))
	img_circle = ori.copy()
	gray = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)

	### image process
	img_circle = detect_circles(gray, img_circle)
	img_segmentate = segmentate_image(gray)

	# ### show image
	# cv2.imshow("result", np.hstack([gray, img_segmentate]))
	# cv2.waitKey(0)
 # 	### show image
	# img_segmentate = cv2.cvtColor(img_segmentate, cv2.COLOR_GRAY2BGR)
	# cv2.imshow("result", np.hstack([img_circle, img_segmentate]))
	# cv2.waitKey(0)


	# img = cv2.imread("test1.jpg")
	# img = cv2.resize(img, (550, 650))
	# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	# mask = cv2.inRange(hsv,(15, 100, 20), (25, 255, 255) )
	# cv2.imshow("orange", mask);cv2.waitKey();cv2.destroyAllWindows()
	# ori = cv2.imread("test1.jpg", cv2.IMREAD_COLOR)  # cv2.IMREAD_COLOR = 1
	# print(ori)
	# ori = cv2.resize(ori, (540, 660)) >> 7
	# ori = ori << 7
	# print(ori)
	# cv2.imshow("result", np.hstack([ori]))
	# cv2.waitKey(0)
