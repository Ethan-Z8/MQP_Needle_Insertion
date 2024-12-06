% Read the image data from the CSV file
image_data = readtable('array_frame_1.csv');  % Use readmatrix in newer versions of MATLAB

% Display the image
imshow(image_data, []);  % The empty brackets ensure the    correct scaling for display
colormap gray;  % Use grayscale colormap for better display
axis off;  % Optionally, hide the axis
