clear all
close all
clc





rosshutdown;

rosinit



PAIMG_sub = rossubscriber('PA_IMG', 'std_msgs/Float64MultiArray');
k = 1; t1=0;
tic;
while 1
PAIMG_msg = receive(PAIMG_sub);
PAIMG = PAIMG_msg.Data;

PAIMG0 = reshape(PAIMG,570,500);
imagesc(PAIMG0)
colormap gray
drawnow
t = toc;
fprintf(['Frame #',num2str(k),'  F=',num2str(1/(t-t1)),'Hz\n'])
k = k+1;
t1 = t;
end


rstart = 140;
rend =348;
cstart = 195;
cend = 235;

ROI_image = ROI_creation(PAIMG,rstart,rend,cstart,cend)

function  ROI_image = ROI_creation(image,row_start, row_end, col_start,col_end)
    ROI_frame = image(row_start:row_end, col_start:col_end)

    ROI_image = ROI_frame;
    return ROI_image;

end

