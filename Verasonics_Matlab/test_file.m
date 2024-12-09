% Read the image data from the CSV file
image_data = readtable('array_frame_1.csv');  % Use readmatrix in newer versions of MATLAB

% Display the image
class(image_data)

matrix = table2array(image_data);

imagesc(matrix)
colormap gray