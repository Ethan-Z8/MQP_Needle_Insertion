import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt
import math
import scipy.io
import os
# print(list(plt.colormaps))


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
    



