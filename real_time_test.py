import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt
import math
import scipy.io
import os
# print(list(plt.colormaps))

def detect_bbox(img,bbox_image):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # # Debugging statements
        # if area > 1:
        #     print("Area of contour is: {}".format(area))
        
        areaMin = 15
        areaMax = 100
        if area > areaMin and area < areaMax:
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
            else:
                cx = 0
                cy = 0
            print(cx, cy)
            # cv2.drawContours(bbox_image, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # print(len(approx))
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(bbox_image, (x, y), (x + w, y + h), (0, 255, 0), 5)
            # cv2.circle(bbox_image, (cx, cy), 7, (0,255,0), -1)
def line_creation(source_image, overlay):
    lines = cv2.HoughLinesP(source_image, rho=6, theta=np.pi / 2, threshold=160, lines=np.array([]), minLineLength=40, maxLineGap=4)

    overlay_image = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
    houghline = overlay_image.copy()

    if lines is not None:
        for line in lines:
            
            start_point = (line[0][0], line[0][1]) # represents the top left corner of image
            end_point = (line[0][2], line[0][3]) # represents the bottom right corner of image
            color = (0, 255, 0) # Green color in BGR
            thickness = 2 # Line thickness
                
            cv2.line(houghline, start_point, end_point, color, thickness)
    
    return houghline

def ROI_creation(source_image, row_start, row_end, col_start, col_end):
    ROI_frame = source_image[row_start:row_end, col_start:col_end] #old one was [94:348, 166:275]
    ROI_image = np.zeros_like(source_image)
    x = row_start 
    y = col_start 
    for i in range(0, row_end-row_start):
        for j in range(0, col_end-col_start):
            if ROI_frame[i][j] != 0:
                ROI_image[x + i, y + j] = ROI_frame[i, j]
    return ROI_image


def line_creation2(source_image, overlay):
    
    lines = cv2.HoughLinesP(source_image, rho=6, theta=np.pi / 2, threshold=160, lines=np.array([]), minLineLength=40, maxLineGap=4)
    overlay_image = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
    
    houghline = overlay_image.copy()
    houghcircle = overlay_image.copy()
    
    length_line_list = []

    if lines is not None:
        for line in lines:

            x1 = line[0][0]
            y1 = line[0][1]
            x2 = line[0][2]
            y2 = line[0][3]

            start_point = (x1, y1)
            end_point = (x2, y2)
            
            lengthOfLine = math.sqrt(abs(x2-x1)^2 + abs(y2-y1)^2)
            length_line_list.append(lengthOfLine)

        index_number = length_line_list.index(max(length_line_list))

        x1 = lines[index_number][0][0]
        y1 = lines[index_number][0][1]
        x2 = lines[index_number][0][2]
        y2 = lines[index_number][0][3]

        start_point = (x1, y1)
        end_point = (x2, y2)

        color = (0, 255, 0) # Green color in BGR
        thickness = 2 # Line thickness of 9 px
        radius = 5 #circle radius

        cv2.line(houghline, start_point, end_point, color, thickness)
        cv2.circle(houghcircle, end_point, radius, color, thickness)
    
    return houghline


def needle_tip_estimation(source_image, overlay):
    
    lines = cv2.HoughLinesP(source_image, rho=6, theta=np.pi / 2, threshold=160, lines=np.array([]), minLineLength=40, maxLineGap=4)
    overlay_image = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
    
    houghcircle = overlay_image.copy()
    
    length_line_list = []

    if lines is not None:
        for line in lines:

            x1 = line[0][0]
            y1 = line[0][1]
            x2 = line[0][2]
            y2 = line[0][3]
            
            lengthOfLine = math.sqrt(abs(x2-x1)^2 + abs(y2-y1)^2)
            length_line_list.append(lengthOfLine)

        index_number = length_line_list.index(max(length_line_list))

        x1 = lines[index_number][0][0]
        y1 = lines[index_number][0][1]
        x2 = lines[index_number][0][2]
        y2 = lines[index_number][0][3]

        start_point = (x1, y1)

        color = (0, 255, 0) # Green color in BGR
        thickness = 2 # Line thickness of 9 px
        radius = 5 #circle radius

        cv2.circle(houghcircle, start_point, radius, color, thickness)
    
    return houghcircle

class NeedleVisualization:

    def __init__(self, frame):
        self.frame = frame
        self.frameWidth = 440
        self.frameHeight = 440
        
        #ROI parameters
        self.rstart = 140 #previously 94
        self.rend = 348
        self. cstart = 195 #previously 166
        self. cend = 235 #previously 275

        #Initial Preprocessing
        self.resized_frame = cv2.resize(self.frame, (self.frameWidth,self.frameHeight))
        self.resized_frame = cv2.cvtColor(self.resized_frame, cv2.COLOR_RGB2GRAY)


    def line_creation(self, source_image, overlay):
        lines = cv2.HoughLinesP(source_image, rho=6, theta=np.pi / 2, threshold=160, lines=np.array([]), minLineLength=40, maxLineGap=4)

        overlay_image = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
        houghline = overlay_image.copy()

        if lines is not None:
            for line in lines:
                
                start_point = (line[0][0], line[0][1]) # represents the top left corner of image
                end_point = (line[0][2], line[0][3]) # represents the bottom right corner of image
                color = (0, 255, 0) # Green color in BGR
                thickness = 2 # Line thickness
                    
                cv2.line(houghline, start_point, end_point, color, thickness)
        
        return houghline

    def line_creation2(self, source_image, overlay):
        
        lines = cv2.HoughLinesP(source_image, rho=6, theta=np.pi / 2, threshold=160, lines=np.array([]), minLineLength=40, maxLineGap=4)
        overlay_image = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
        
        houghline = overlay_image.copy()
        houghcircle = overlay_image.copy()
        
        length_line_list = []

        if lines is not None:
            for line in lines:

                x1 = line[0][0]
                y1 = line[0][1]
                x2 = line[0][2]
                y2 = line[0][3]

                start_point = (x1, y1)
                end_point = (x2, y2)
                
                lengthOfLine = math.sqrt(abs(x2-x1)^2 + abs(y2-y1)^2)
                length_line_list.append(lengthOfLine)

            index_number = length_line_list.index(max(length_line_list))

            x1 = lines[index_number][0][0]
            y1 = lines[index_number][0][1]
            x2 = lines[index_number][0][2]
            y2 = lines[index_number][0][3]

            start_point = (x1, y1)
            end_point = (x2, y2)
            print(end_point)

            color = (0, 255, 0) # Green color in BGR
            thickness = 2 # Line thickness of 9 px
            radius = 5 #circle radius
    
            cv2.line(houghline, start_point, end_point, color, thickness)
            cv2.circle(houghcircle, end_point, radius, color, thickness)
        
        return houghline

    def ROI_creation(source_image, row_start, row_end, col_start, col_end):
        ROI_frame = source_image[row_start:row_end, col_start:col_end] #old one was [94:348, 166:275]
        ROI_image = np.zeros_like(source_image)
        x = row_start 
        y = col_start 
        for i in range(0, row_end-row_start):
            for j in range(0, col_end-col_start):
                if ROI_frame[i][j] != 0:
                    ROI_image[x + i, y + j] = ROI_frame[i, j]
        return ROI_image

    
    def detect_needle_line(self):

        #Achieving desired region of interest within Raw Frame
        ##############################################################
        ROI_image = ROI_creation(self.resized_frame,self.rstart,self.rend,self.cstart,self.cend)
        ##############################################################

        #Applying Paper Algorithm Filters
        #############################################################
        # gabor_filter = cv2.getGaborKernel((6,6), sigma=0.5, theta=0, lambd=0.5, gamma=0.8, psi=0, ktype=cv2.CV_32F)
        gabor_filter = cv2.getGaborKernel((3,3), sigma=0.95, theta=0, lambd=5, gamma=0.8, psi=0, ktype=cv2.CV_32F)
        # gabor_filter = cv2.getGaborKernel((3,3), sigma=0.5, theta=0, lambd=30, gamma=0.8, psi=0, ktype=cv2.CV_32F)

        gabor_output = cv2.filter2D(ROI_image, -1, gabor_filter)

        #Binarized image is divided into grids for needle axis localization.
        # - Median filter
        median_filter = cv2.medianBlur(gabor_output, 7)
        # - automatic thresholding
        threshold = cv2.threshold(median_filter, 250, 255, cv2.THRESH_BINARY)[1]
        # - morphological operations
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        eroded = cv2.erode(threshold, element)
        dilated = cv2.dilate(eroded, element)
        #############################################################
            
        #Hough Line Transforms
        #############################################################
        houghline = line_creation(dilated, self.resized_frame)
        #############################################################
        return houghline
    




#videos to choose from
NeedleViz_path1 = '1/edited data/102622_Water.mp4'
NeedleViz_path2 = 'Data/edited data/102822_Water.mp4'
NeedleViz_oilAndLatex = 'Data/edited data/oil and latex/capture_5_2022-11-12T16-56-03.mp4'
NeedleViz_gelAndLatex = 'Data/edited data/ultrasound gel and latex/capture_4_2022-11-12T17-33-19.mp4'
NeedleViz_clarius1 = 'Data/edited data/clarius_FinalPrototype_needlejustWater.mp4'
NeedleViz_clarius2 = 'Data/edited data/clarius_FinalPrototype_needlejustWater2.mp4'
NeedleViz_clarius3 = 'Data/edited data/clarius_FinalPrototype_needleWithSolid.mp4'
NeedleViz_clarius4 = 'Data/edited data/clarius_FinalPrototype_needleWithSolid2.mp4'
NeedleViz_clarius5 = 'Data/edited data/clarius_FinalPrototype_needleWithSolid3.mp4'





#control playback speed
frame_rate = 30
# vc = cv2.VideoCapture(0) #opens camera
vc = cv2.VideoCapture(NeedleViz_clarius5)

frameWidth = 440
frameHeight = 440
vc.set(3, frameWidth)
vc.set(4, frameHeight)

size = (frameWidth, frameHeight)

#Preparing to create output videos
image_lst = []

if (vc.isOpened()== False): 
  print("Error opening video  file")

while(vc.isOpened()):
    rval, frame = vc.read()
    
    if rval == True:

        #Initial Frame preprocessing
        ##############################################################
        resized_frame = cv2.resize(frame, (frameWidth,frameHeight))
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2GRAY)
        ##############################################################


        #Achieving desired region of interest within Raw Frame
        ##############################################################
        rstart = 140 #previously 94
        rend = 348
        cstart = 195 #previously 166
        cend = 235 #previously 275

        ROI_image = ROI_creation(resized_frame,rstart,rend,cstart,cend)
        ############################################################## 
      
        #Applying Combination Filters
        #############################################################
        
        ### THRESHOLDING ###
        thresh = cv2.threshold(ROI_image, 90, 255, cv2.THRESH_BINARY)[1]

        ### BASIC MORPHOLOGICAL OPERATIONS ###
        # dilate = cv2.dilate(thresh, None, iterations=1)
        # erode = cv2.erode(dilate, None, iterations=1)
        # dilate_2 = cv2.dilate(erode, None, iterations=1)

        
        ### ADVANCED MORPHOLIGICAL OPERATIONS (skeletonization) ###
        skel_image = thresh.copy()

        # Step 1: Create an empty skeleton
        size = np.size(skel_image)
        skel = np.zeros(skel_image.shape, np.uint8)

        # Get a Cross Shaped Kernel
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

        #Step 2: Open the image
        open = cv2.morphologyEx(skel_image, cv2.MORPH_OPEN, element)
        #Step 3: Substract open from the original image
        temp = cv2.subtract(skel_image, open)
        #Step 4: Erode the original image and refine the skeleton
        eroded = cv2.erode(skel_image, element)
        skel = cv2.bitwise_or(skel_image,temp)
        skel_image = eroded.copy()
        #############################################################
        
        #Applying Edge and Bounding box detection
        #############################################################
        canny = cv2.Canny(skel_image, 73,200)
        bbox = resized_frame.copy()
        # detect_bbox(canny,bbox)
        #############################################################

        #Applying Paper Algorithm Filters
        #############################################################
        # gabor_filter = cv2.getGaborKernel((6,6), sigma=0.5, theta=0, lambd=0.5, gamma=0.8, psi=0, ktype=cv2.CV_32F)
        gabor_filter = cv2.getGaborKernel((3,3), sigma=0.95, theta=0, lambd=5, gamma=0.8, psi=0, ktype=cv2.CV_32F)
        # gabor_filter = cv2.getGaborKernel((3,3), sigma=0.5, theta=0, lambd=30, gamma=0.8, psi=0, ktype=cv2.CV_32F)

        gabor_output = cv2.filter2D(ROI_image, -1, gabor_filter)

        #Binarized image is divided into grids for needle axis localization.
        # - Median filter
        median_filter = cv2.medianBlur(gabor_output, 7)
        # - automatic thresholding
        threshold = cv2.threshold(median_filter, 250, 255, cv2.THRESH_BINARY)[1]
        # - morphological operations
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        eroded = cv2.erode(threshold, element)
        dilated = cv2.dilate(eroded, element)
        #############################################################
        
        #Hough Line Transforms
        #############################################################
        # houghline = line_creation(dilated, resized_frame)
        houghline = line_creation2(dilated, resized_frame)
        houghcircle = needle_tip_estimation(dilated, resized_frame)

        #############################################################
        
        #Overlaying segmentations onto B-mode image
        #############################################################################################
        # fgmaskV2_color = cv2.applyColorMap(bbox, cv2.COLORMAP_INFERNO)
        # resized_frame_revert = cv2.cvtColor(resized_frame, cv2.COLOR_GRAY2RGB)
        # overlay = cv2.addWeighted(resized_frame_revert, 0.5, fgmaskV2_color, 0.5, 1.0)
        # cv2.imshow("Bmode Overlay", overlay)
        ###########################################################################################

        # Debugging Statements
        # cv2.imshow('normal frame', resized_frame)
        # cv2.imshow('ROI frame', ROI_image)
        # cv2.imshow('thresholding', thresh)
        # cv2.imshow('Morphological Operations', skel_image)
        # cv2.imshow('Canny Edge Detection', canny)
        # cv2.imshow('Object Detection', bbox)
        # cv2.imshow('Paper Algorithm', dilated)
        # cv2.imshow('Hough Line Transform', houghline)
        # cv2.imshow('Needle tip Estimation', houghcircle)
        
        #Saving comparison frames as gif 
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_GRAY2BGR)
        line = cv2.cvtColor(houghline, cv2.COLOR_RGB2BGR)
        algorithm = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
        tip = cv2.cvtColor(houghcircle, cv2.COLOR_RGB2BGR)
        stack = np.hstack((resized_frame, line, tip))
        cv2.imshow("stacked", stack)
        image_lst.append(stack)

        # Press Q on keyboard to  exit
        if cv2.waitKey(frame_rate) & 0xFF == ord('q'): #original waitkey is 25
            break
    
    #Break out of loop if video is done
    else:
        break  

vc.release() #Release the video capture object

# Close window
cv2.destroyAllWindows()




#rewrite to work with .mat filesz





#Initial Frame preprocessing
##############################################################
resized_frame = cv2.resize(frame, (frameWidth,frameHeight))
resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2GRAY)
##############################################################


#Achieving desired region of interest within Raw Frame
##############################################################
rstart = 140 #previously 94
rend = 348
cstart = 195 #previously 166
cend = 235 #previously 275

ROI_image = ROI_creation(resized_frame,rstart,rend,cstart,cend)
############################################################## 

#Applying Combination Filters
#############################################################

### THRESHOLDING ###
thresh = cv2.threshold(ROI_image, 90, 255, cv2.THRESH_BINARY)[1]

### BASIC MORPHOLOGICAL OPERATIONS ###
# dilate = cv2.dilate(thresh, None, iterations=1)
# erode = cv2.erode(dilate, None, iterations=1)
# dilate_2 = cv2.dilate(erode, None, iterations=1)


### ADVANCED MORPHOLIGICAL OPERATIONS (skeletonization) ###
skel_image = thresh.copy()

# Step 1: Create an empty skeleton
size = np.size(skel_image)
skel = np.zeros(skel_image.shape, np.uint8)

# Get a Cross Shaped Kernel
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

#Step 2: Open the image
open = cv2.morphologyEx(skel_image, cv2.MORPH_OPEN, element)
#Step 3: Substract open from the original image
temp = cv2.subtract(skel_image, open)
#Step 4: Erode the original image and refine the skeleton
eroded = cv2.erode(skel_image, element)
skel = cv2.bitwise_or(skel_image,temp)
skel_image = eroded.copy()
#############################################################

#Applying Edge and Bounding box detection
#############################################################
canny = cv2.Canny(skel_image, 73,200)
bbox = resized_frame.copy()
# detect_bbox(canny,bbox)
#############################################################

#Applying Paper Algorithm Filters
#############################################################
# gabor_filter = cv2.getGaborKernel((6,6), sigma=0.5, theta=0, lambd=0.5, gamma=0.8, psi=0, ktype=cv2.CV_32F)
gabor_filter = cv2.getGaborKernel((3,3), sigma=0.95, theta=0, lambd=5, gamma=0.8, psi=0, ktype=cv2.CV_32F)
# gabor_filter = cv2.getGaborKernel((3,3), sigma=0.5, theta=0, lambd=30, gamma=0.8, psi=0, ktype=cv2.CV_32F)

gabor_output = cv2.filter2D(ROI_image, -1, gabor_filter)

#Binarized image is divided into grids for needle axis localization.
# - Median filter
median_filter = cv2.medianBlur(gabor_output, 7)
# - automatic thresholding
threshold = cv2.threshold(median_filter, 250, 255, cv2.THRESH_BINARY)[1]
# - morphological operations
element = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
eroded = cv2.erode(threshold, element)
dilated = cv2.dilate(eroded, element)
#############################################################

#Hough Line Transforms
#############################################################
# houghline = line_creation(dilated, resized_frame)
houghline = line_creation2(dilated, resized_frame)
houghcircle = needle_tip_estimation(dilated, resized_frame)

#############################################################

#Overlaying segmentations onto B-mode image
#############################################################################################
# fgmaskV2_color = cv2.applyColorMap(bbox, cv2.COLORMAP_INFERNO)
# resized_frame_revert = cv2.cvtColor(resized_frame, cv2.COLOR_GRAY2RGB)
# overlay = cv2.addWeighted(resized_frame_revert, 0.5, fgmaskV2_color, 0.5, 1.0)
# cv2.imshow("Bmode Overlay", overlay)
###########################################################################################

# Debugging Statements
# cv2.imshow('normal frame', resized_frame)
# cv2.imshow('ROI frame', ROI_image)
# cv2.imshow('thresholding', thresh)
# cv2.imshow('Morphological Operations', skel_image)
# cv2.imshow('Canny Edge Detection', canny)
# cv2.imshow('Object Detection', bbox)
# cv2.imshow('Paper Algorithm', dilated)
# cv2.imshow('Hough Line Transform', houghline)
# cv2.imshow('Needle tip Estimation', houghcircle)

#Saving comparison frames as gif 
resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_GRAY2BGR)
line = cv2.cvtColor(houghline, cv2.COLOR_RGB2BGR)
algorithm = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
tip = cv2.cvtColor(houghcircle, cv2.COLOR_RGB2BGR)
stack = np.hstack((resized_frame, line, tip))
cv2.imshow("stacked", stack)
image_lst.append(stack)


        

# Close window
cv2.destroyAllWindows()





curr_frame = 1
# print(type(scipy.io.loadmat(f'Pdata_acquisition1.mat')))

# if(os.path.exists("Pdata_acquisition11.mat")):
#      print("yay")
# else:
#     print("no!")
#use this to test real time file reading a deletion
while(True):
    #check whcih frame to collect
    mat_current = scipy.io.loadmat(f'Pdata_acquisition{curr_frame}.mat')

    
    print(f"current frame {curr_frame}")
    data_loop = mat_current["p_data"]
    #data normalization
    max_val = np.amax(data_loop)
    greyscaled_data = np.array(data_loop)/max_val


    #showing real data
    cv2.imshow("Loop display",greyscaled_data)
    #how often loop runs rn
    cv2.waitKey(25)
    #logic to move to next frame 
    if(os.path.exists(f"Pdata_acquisition{curr_frame+1}.mat")):
        print("yay")
        curr_frame = curr_frame + 1
    if(os.path.exists(f"Pdata_acquisition{curr_frame-100}.mat")):
        os.remove(f"Pdata_acquisition{curr_frame-100}.mat")
        print(f"removed file num:{curr_frame-100}")

    if curr_frame == 10:
        break
    



