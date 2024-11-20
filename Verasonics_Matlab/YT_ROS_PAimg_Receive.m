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