clear all
close all
clc

rstart = 140;
rend =348;
cstart = 195;
cend = 235;


image_data = readtable('array_frame_1.csv');  % Use readmatrix in newer versions of MATLAB

% Display the image
class(image_data)

matrix = table2array(image_data);

% imagesc(matrix)
colormap gray



ROI_image = ROI_creation(matrix,rstart,rend,cstart,cend)
imagesc(ROI_image)

threshold = 90 / 255; % Normalize the threshold to [0, 1] range for MATLAB
binary_image = imbinarize(ROI_image, threshold);

% If you need the binary image as uint8 (0 or 255), scale it:
binary_image = uint8(binary_image) * 255;

skel_image = binary_image; % Create a copy of the binary image


size = numel(skel_image);
skel = zeros(size(skel_image), 'uint8');






% rosshutdown;

% rosinit


% PAIMG_sub = rossubscriber('PA_IMG', 'std_msgs/Float64MultiArray');
% k = 1; t1=0;
% tic;
% while 1
% PAIMG_msg = receive(PAIMG_sub);
% PAIMG = PAIMG_msg.Data;

% PAIMG0 = reshape(PAIMG,570,500);
% imagesc(PAIMG0)
% colormap gray
% drawnow
% t = toc;
% fprintf(['Frame #',num2str(k),'  F=',num2str(1/(t-t1)),'Hz\n'])
% k = k+1;
% t1 = t;
% end


% rstart = 140;
% rend =348;
% cstart = 195;
% cend = 235;

% ROI_image = ROI_creation(PAIMG,rstart,rend,cstart,cend)
function detect_bbox(img, bbox_image)
    % Find boundaries (contours) in a binary image
    boundaries = bwboundaries(img, 'noholes');
    
    % Loop through each contour
    for k = 1:length(boundaries)
        boundary = boundaries{k};  % Get the kth contour
        
        % Calculate the area of the contour
        area = polyarea(boundary(:,2), boundary(:,1));  % Note the order of coordinates in polyarea
        
        areaMin = 15;
        areaMax = 100;
        
        % Only process contours within the specified area range
        if area > areaMin && area < areaMax
            % Calculate the centroid (moment of the contour)
            M = regionprops(boundary(:,2), boundary(:,1), 'Centroid');
            cx = round(M.Centroid(1));
            cy = round(M.Centroid(2));
            disp([cx, cy])  % Display centroid coordinates
            
            % Draw bounding box
            % Get the bounding box for the contour (min bounding rectangle)
            x = min(boundary(:,2));
            y = min(boundary(:,1));
            w = max(boundary(:,2)) - x;
            h = max(boundary(:,1)) - y;
            
            % Draw rectangle around the contour
            bbox_image = insertShape(bbox_image, 'Rectangle', [x, y, w, h], 'Color', 'green', 'LineWidth', 5);
        end
    end
    imshow(bbox_image);  % Display the image with bounding boxes
end

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




% function  ROI_image = ROI_creation(image,row_start, row_end, col_start,col_end)
%     ROI_frame = image(row_start:row_end, col_start:col_end)

%     ROI_image = ROI_frame;
% end
function ROI_image = ROI_creation(source_image, row_start, row_end, col_start, col_end)
    % Extract the region of interest (ROI) from the source image
    ROI_frame = source_image(row_start:row_end-1, col_start:col_end-1);

    % Initialize an empty image of the same size as the source image
    ROI_image = zeros(size(source_image));
    % Offset variables
    x = row_start;
    y = col_start;
    % Iterate through the ROI and copy non-zero values to the new image
    for i = 1:(row_end - row_start)
        for j = 1:(col_end - col_start)
            if ROI_frame(i, j) ~= 0
                ROI_image(x + i - 1, y + j - 1) = ROI_frame(i, j);
            end
        end
    end
end





function houghline = line_creation2(source_image, overlay)
    % Perform edge detection (Canny) on the source image if not already done
    edges = edge(source_image, 'Canny');  % Apply Canny edge detector

    % Perform Hough transform to detect lines
    [H, T, R] = hough(edges);

    % Extract line segments from the Hough transform using 'houghlines'
    lines = houghlines(edges, T, R, 'FillGap', 4, 'MinLength', 40);
    
    % Convert the overlay image from grayscale to RGB if needed
    overlay_image = repmat(overlay, [1, 1, 3]);  % Convert grayscale to RGB
    houghline = overlay_image;  % Copy overlay image to draw lines
    houghcircle = overlay_image;  % Create a copy for drawing circle

    % List to store the lengths of the lines
    length_line_list = [];

    % Check if there are any lines detected
    if ~isempty(lines)
        for k = 1:length(lines)
            % Get start and end points of each line
            x1 = lines(k).point1(1);
            y1 = lines(k).point1(2);
            x2 = lines(k).point2(1);
            y2 = lines(k).point2(2);

            % Calculate the length of the line
            lengthOfLine = sqrt((x2 - x1)^2 + (y2 - y1)^2);
            length_line_list = [length_line_list, lengthOfLine];
        end
        
        % Find the index of the longest line
        [~, index_number] = max(length_line_list);

        % Get the coordinates of the longest line
        x1 = lines(index_number).point1(1);
        y1 = lines(index_number).point1(2);
        x2 = lines(index_number).point2(1);
        y2 = lines(index_number).point2(2);

        % Draw the longest line on the image
        color = [0, 255, 0];  % Green color in RGB
        thickness = 2;        % Line thickness

        % Draw the line on houghline image
        houghline = insertShape(houghline, 'Line', [x1, y1, x2, y2], 'Color', color, 'LineWidth', thickness);

        % Draw a circle at the end point of the longest line
        radius = 5;  % Circle radius
        houghcircle = insertShape(houghcircle, 'FilledCircle', [x2, y2, radius], 'Color', color, 'LineWidth', thickness);
    end
end
