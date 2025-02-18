
data = load('C:\Users\ezhon\OneDrive\Desktop\Git_ultrasound\MQP_Needle_Insertion\Needeless_Test_data(p_data)\Pdata_acquisition1.mat');
info = whos('-file', 'C:\Users\ezhon\OneDrive\Desktop\Git_ultrasound\MQP_Needle_Insertion\Needeless_Test_data(p_data)\Pdata_acquisition1.mat');
disp(info);

rstart = 100;%was 300
rend =500;%was800
cstart = 100;
cend = 500;
ROI_image = ROI_creation(data,rstart,rend,cstart,cend);

imagesc(ROI_image)


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
