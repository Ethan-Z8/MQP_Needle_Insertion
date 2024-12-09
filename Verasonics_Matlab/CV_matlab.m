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



function houghline = line_creation(source_image, overlay)
    % Perform edge detection (Canny) on the source image if not already done
    edges = edge(source_image, 'Canny');  % Apply Canny edge detector

    % Perform Hough transform to detect lines
    [H, T, R] = hough(edges);

    % Extract line segments from the Hough transform using the 'houghlines' function
    lines = houghlines(edges, T, R, 'FillGap', 4, 'MinLength', 40);
    
    % Convert the overlay image from grayscale to RGB if needed
    overlay_image = repmat(overlay, [1, 1, 3]);  % Convert grayscale to RGB
    houghline = overlay_image;  % Copy overlay image to draw lines

    % Draw the detected lines
    for k = 1:length(lines)
        start_point = lines(k).point1;  % Starting point of the line
        end_point = lines(k).point2;    % Ending point of the line
        color = [0, 255, 0];            % Green color in RGB
        thickness = 2;                  % Line thickness
        
        % Draw the line on the image
        houghline = insertShape(houghline, 'Line', [start_point, end_point], 'Color', color, 'LineWidth', thickness);
    end
end


function  ROI_image = ROI_creation(image,row_start, row_end, col_start,col_end)
    ROI_frame = image(row_start:row_end, col_start:col_end)

    ROI_image = ROI_frame;
end

function 

