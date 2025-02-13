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
% matrix_sum = sum(PAIMG0);

% imshow(PAIMG0)
% image(PAIMG0)
% imagesc(PAIMG0)
PAIMG1 = PAIMG0./max(PAIMG0(:));
PAIMG2 = db(PAIMG1);


imagesc(PAIMG2,[-50,0]);% dynamic range -50,0

colormap gray



% drawnow
t = toc;
fprintf(['Frame #',num2str(k),'  F=',num2str(1/(t-t1)),'Hz\n'])
k = k+1;
t1 = t;


end



    
    