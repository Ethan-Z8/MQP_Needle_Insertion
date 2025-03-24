clear all
close all
clc

rosshutdown;

rosinit


%bluetooth
bluetoothObj = BluetoothClient();  

control_var = 0;
PAIMG_sub = rossubscriber('PA_IMG', 'std_msgs/Float64MultiArray')
%check the buffer size
k = 1; t1=0;
tic;
while 1

PAIMG_msg = receive(PAIMG_sub);
PAIMG = PAIMG_msg.Data;




% data = load('C:\Users\ezhon\OneDrive\Desktop\Git_ultrasound\MQP_Needle_Insertion\Needeless_Test_data(p_data)\Pdata_acquisition1.mat');
% info = whos('-file', 'C:\Users\ezhon\OneDrive\Desktop\Git_ultrasound\MQP_Needle_Insertion\Needeless_Test_data(p_data)\Pdata_acquisition1.mat');
% disp(info);



PAIMG0 = reshape(PAIMG,570,500);


PAIMG1 = PAIMG0./max(PAIMG0(:));
PAIMG2 = db(PAIMG1);


imagesc(PAIMG2,[-50,0]);% dynamic range -50,0


colormap gray
drawnow
t = toc;
fprintf(['Frame #',num2str(k),'  F=',num2str(1/(t-t1)),'Hz\n'])
k = k+1;
t1 = t;




inpict = PAIMG2;

%move the the max
%move the 30 positions
    %run algo spit out number


rstart = 100;%was 300
rend =470;%was800
cstart = 100;
cend = 400;

bluetoothObj.fullRight;
disp("move to the Right");

%FOR LOOP
for i = 1:30

%have to grab images from in here
%might desync image sent and image recieve

fprintf('value of i: %d\n', i); 


inpict = PAIMG2;
ROI_image = ROI_creation(inpict,rstart,rend,cstart,cend);

inpict = ROI_image;

imagesc(ROI_image);



inpict = inpict - mean(inpict,2);

% low pass elliptical filering of the input image (to remove further the
% salt pepper noise) - adjust filter cut off and order to your own
% preferences
spec_img = fftshift(fft2(inpict));

sze = size(spec_img);
cutoff1 = 0.5;
cutoff2 = 0.05;
n = 6;
f = EFilter(sze, cutoff1, cutoff2, n);

% apply filter
spec_img = spec_img.*f;

% generated backward the output image by inverse fft
outpict = real(ifft2(ifftshift(spec_img)));

%figure
%subplot(2,1,1),imshow(inpict)
%subplot(2,1,2),imshow(outpict)



%outpic is blurred 

%% attempt to extract the white segment
% Threshold the image to create a binary image
binaryImage = outpict > 0.4*max(outpict(:)); 

% Display the binary image
%figure;
%imshow(binaryImage);
%title('Binary Image (White Segments)');
%hold on;
%drawnow;

% find the boundary points
%gets non-zero points in a list y x which are vectors
[y,x] = ind2sub(size(binaryImage),find(binaryImage>0.5));
%calculate the boundery points from points found

[y_selec,x_selec] = myboundary(y,x);
if y_selec == -1 
    disp("not found!")
else

    %draw around found boundry points
    plot(x,y, '.', x_selec, y_selec, '.r')
    title('Boundary points Highlighted');
    
    %find the avg pix values for the threshold values
    pixelValues = inpict(sub2ind(size(binaryImage), y, x));
    avgPixelValue = mean(pixelValues);
    disp("avg pix val: " + avgPixelValue)
    
    
    %maybe find a way to keep track of time
    %if avgPixelValue < .3 && control_var < 5
    %    bluetoothObj.stepF;
    %    control_var = control_var + 1;
    %elseif avgPixelValue < .3 && control_var < 10
    %    bluetoothObj.stepB;
    %    control_var = control_var + 1;
    %elseif avgPixelValue < .3 && control_var > 10
    %    bluetoothObj.sweep
    %    control_var = 0;
    %end
    
    % last round !!!
    % let say we don't want to keep line objects with width > tol (in pixels)
    tol = 0.03*sze(2); % here the tol is 3% of the picture width
    [y_selec_unic,ia,ic] = unique(y_selec);
                       
    % "scroll" the image along the y direction and look for narrow  profiles 
    
    m = 0;  
    for k = 1:numel(y_selec_unic)
        ind = ic == k;
        x_selected = x_selec(ind);
        dx = max(x_selected) - min(x_selected);
        if dx < tol % we keep it
            m = m+1;
            yfinal(m) = y_selec_unic(k);
            xfinal(m) = mean(x_selected);
        end
    end
    
    % return
    %tmp = abs(diff(xfinal));
    %ind = find(tmp>0.1*max(tmp));
    %[v,ii] = max(diff(ind));
    %iii = ind(ii):ind(ii+1);
    %xxx = xfinal(iii);
    %yyy = yfinal(iii);
    
    %plot(xxx, yyy, 'dg')
    %hold off;
    
    
    %% FINAL PLOT !!!!!!!!!
    
    figure
    imshow(inpict)
    hold on 
    plot(xxx, yyy, '.r')
    hold off;
    drawnow;

    toc
end

bluetoothObj.stepB
disp("next step")
pause(1)

end

pause(100000)
disp("FIRST SET OVER")
end



















%1 or 2 not sure

% remove mean along dimension 2 

% [rows, cols] = size(img);
% rstart = rows * 0.1;
% rend =rows *0.8;%was800
% cstart = cols *0.1;
% cend = cols * 0.8;





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function varargout = EFilter(sze, cutoffM, cutoffm, n, varargin)

    %EFILTER constructs an elliptical lowpass filter Butterworth filter
    %   E = EFILTER(Size, cutoffM, cutoffm, n) designs an Nth order elliptical
    %   lowpass digital Butterworth filter where Size is a two element 
    %   [rows, cols] vector specifying the size of the filter to construct, 
    %   cutoffM and cutoffm, the cutoff freqency on the major and minor
    %   axes are 0 < cutoff <= 1.
    %
    %   If E = EFilter(Size, cutoffM, cutoffm, n, alpha), where alpha is an angle
    %   in radians, it will return and plot an elliptical filter rotated
    %   counter-clockwise through alpha.
    %
    %   If E = EFilter(Size, cutoffM, cutoffm, n, alpha, xoff, yoff), where xoff
    %   and yoff are offsets in the x and y direction, it will return and
    %   plot an eliptical filter which is offset by the specified amount.
    %   An offset of 0 corresponds to the center and an offset of 1
    %   corresponds to the edge of the filter. A positive offset shifts the
    %   filter in the positive direction.
    %
    %   Calling EFilter(...) without assigning the output variable
    %   plots the 3D surface described by the function.
    
    % Katie Streit   kstreit@rice.edu
    % ELEC 301
    % Rice University
    %
    % December 2001
    
    % Much of this code was based on Peter Kovesi's  (pk@cs.uwa.edu.au)
    % Matlab function for a lowpass Butterworth filter.
    
    if nargin == 4
    alpha = 0;
    offx = 0;
    offy = 0;
    elseif nargin == 5
    offx = 0;
    offy = 0;
    alpha = varargin{1};
    elseif nargin == 7
    alpha = varargin{1};
    offx = varargin{2};
    offy = varargin{3};
    else
        error('Invalid number of input arguments');
    end
    
        if nargout > 1
            error('Invalid number of output arguments');
        end
    
        if cutoffM < 0 | cutoffM > 1
        error('cutoffM frequency must be between 0 and 1');
        end
        
        if cutoffm < 0 | cutoffm > 1
            error('cutoffm frequency must be between 0 and 1');
        end  
          
        if rem(n,1) ~= 0 | n < 1
        error('n must be an integer >= 1');
        end
        
    
    %%extracts the sizes from sze 
    rows = sze(1);
    cols = sze(2);
    
    %x and y matrices normalized to +/-.5 and an offset of offx or offy
    x =  ((((ones(rows,1) * [1:cols])-offx*rows/2)  - (fix(cols/2)+1))/cols);
    y =  ((([1:rows]' * ones(1,cols))-offy*rows/2) - (fix(rows/2)+1))/rows;
    
    %applies a linear transformation to rotate through alpha. Note that it takes
    % uses negative alpha, which is caused by x and y being independent matrices.
    x2 = (x*cos(alpha) - y*sin(-alpha));
    y2 = (x*sin(-alpha) + y*cos(alpha));
    
    %constructs an elliptical cone (defined by a and b) of height r on each at 
    %each elliptical ring. (r is effectively the "radius")
    %r = sqrt(((x2/a).^2 + (y2/b).^2));
    
    %Designs the filter
    %f = 1./(1.0 + (r./cutoff).^(2*n));
    
    a = cutoffM/2;
    b = cutoffm/2;
    
    f = 1./(1+((x2/(a)).^2 + (y2/(b)).^2).^n);
    
    if nargout > 0
      varargout{1} = f;
    else
      %Plots a normalized (+/- 1), interpolated 3D image of the filter
      surf([-1:2/(cols-1):1],[-1:2/(rows-1):1], f);
      shading interp;
      title('Elliptical Butterworth filter');
      xlabel('x');
      ylabel('y');
      zlabel('intensity');
      grid on;
    end
    
    end
    
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
    
    