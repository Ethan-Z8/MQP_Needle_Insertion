tic

% filename = 'needle_tip_sample_2.jpg';
%load('Data_new\ethan_needle_250225 (1).mat');
load('C:\Users\ezhon\OneDrive\Desktop\Git_ultrasound\MQP_Needle_Insertion\ethan_data_250324\ethan_0324_9.mat');
%5 and 8 are messiest
%good for 9
%bug on image 10 that creates a very low yfinal point for some reason

% info = whos('-file', 'Data_new/ethan_needle_250225 (1).mat')


data = cell2mat(ImgData);

% size(data)

% data = squeeze(data);
data = reshape(data,[size(data,1),size(data,2),size(data,4)]);
data = mean(data,3);

% non_zero_arr = data(data ~= 0); % Filter out zeros
% min(non_zero_arr)
% max(data(:)) 6.7e+06
% min(data(:)) 0


inpict = data;
PAIMG = inpict;
PAIMG1 = data./max(data(:));
PAIMG2 = db(PAIMG1);

%FLOOR THE IMAGE HERE AND THEN DISPLAY IF VALUE LOWER THAN 
%ADD FLOOR\ TO ALL NUMBERS AND THEN DEVIDE BY 50
arr = PAIMG2;
arr(arr < -50) = -50;
arr = arr + 50;
arr = arr / 50;
imshow(arr);

PAIMG2 = arr;

% PAIMG2 = 10 * log10(PAIMG1 + eps);  % eps prevents log10(0)

min(PAIMG2(:));


% c = 1;  % Small constant to avoid log(0)
% log_image = log(inpict + c);
% log_image_normalized = mat2gray(log_image);  % Normalize to [0, 1]
% imshow(log_image_normalized);


% imagesc(PAIMG2);
% max(PAIMG1)
colormap gray








% remove mean along dimension 2 
% [rows, cols] = size(img);
% rstart = rows * 0.1;
% rend =rows *0.8;%was800
% cstart = cols *0.1;
% cend = cols * 0.8;

%ROWS 0-500
%COL 0-570
rstart = 150;%was 
rend = 470;%up to 570 
cstart = 225;%200
cend = 275;%300 good
ROI_image = ROI_creation(PAIMG2,rstart,rend,cstart,cend);
% size(ROI_image)

inpict = ROI_image;

%works here
% imagesc(inpict);

inpict = inpict - mean(inpict,2);
% inpict = mat2gray(inpict);


% imagesc(inpict)
% title("inpict")
% hold on
% imshow(inpict);
% max(inpict)



% low pass elliptical filering of the input image (to remove further the
% salt pepper noise) - adjust filter cut off and order to your own
% preferences

%rearange Fourier transform

% K = medfilt
spec_img = fftshift(fft2(inpict));

sze = size(spec_img);
cutoffM = 0.35;
cutoffm = 0.05;     
n = 6;
f = EFilter(sze, cutoffM, cutoffm, n);

% apply filter
spec_img = spec_img.*f;

% generated backward the output image by inverse fft
outpict = real(ifft2(ifftshift(spec_img)));

figure
subplot(2,1,1),imshow(inpict)
subplot(2,1,2),imshow(outpict)
drawnow;

% imagesc(outpict)


% %outpic is blurred 

%% attempt to extract the white segment
% Threshold the image to create a binary image
binaryImage = outpict > .5 * max(outpict(:)); 

% Display the binary image
figure;
imshow(binaryImage);
title('Binary Image (White Segments)');
hold on;
drawnow;

% pause(15);

% find the boundary points
%gets non-zero points in a list y x which are vectors
[y,x] = ind2sub(size(binaryImage),find(binaryImage > 0.5));
%calculate teh boundery points from points found
[y_selec,x_selec] = myboundary(y,x);
if y_selec == -1 
    disp("not found!")
else
    %draw around found boundry points
    plot(x,y, '.', x_selec, y_selec, '.r')
    title('Boundary points Highlighted');
    drawnow;
    
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
    tol_m = .001 * sze(2); % here the tol is x% of the picture width
    tol_m = 30;
    % tol_M =  0.1*sze(2);
    [y_selec_unic,ia,ic] = unique(y_selec);

    % plot(100,y_selec_unic,".r");    
    % "scroll" the image along the y direction and look for narrow  profiles 
    
    m = 0;  
    for k = 1:numel(y_selec_unic)
        ind = ic == k;%check indiciy of where that unique y value is in the ic(in the original y_selec matrix)
        x_selected = x_selec(ind);
        % disp(k);
        
        % disp("yvalue :" )
        % disp(y_selec(ind))
        % disp("xvalues :" )
        % disp(x_selec(ind))
        dx = max(x_selected) - min(x_selected);
        if dx < tol_m %|| dx > tol_M% we keep 
            m = m + 1;
            yfinal(m) = y_selec_unic(k);
            xfinal(m) = mean(x_selected);% or make this mean of max min or whatever we end up deciding
        end
    end


    y = 0;
    % length(yfinal)
    % numel(yfinal)
    % numel(yfinal) - 1
    new_y_final = yfinal;
    new_x_final = xfinal;
    % plot(xfinal,yfinal);




    for i = 1:(numel(yfinal) - 1)
        x  = i - y;
        current_y = yfinal(x);
        next_y = yfinal(i+1);

        % current_x = xfinal(x);
        % next_x = xfinal(i+1);
        % abs(current_x - next_x)
        if (next_y - current_y) > 15%check dist
            % numel(yfinal);
            new_y_final(i+1-y) = [];
            new_x_final(i+1-y) = [];
            y = y + 1 ;
        end
    end
    
    % p = polyfit(new_x_final, new_y_final, 0);
    % y_fit = polyval(p, x);
    % y_fit = polyval(p, x);
    plot(new_x_final,new_y_final);


    % for k -> num of elements in y_selec_uniq
        %get the y value y_val = y_sekec_unic(k)
        %get largest and smallest x vals corrasponding to that y val(thats why x selected does)
        %subtract abs val - dx


    % m = 0;  
    % % disp(numel(y_selec_unic))
    % for k = 1:numel(y_selec_unic)
    %     ind = ic == k;
    %     x_selected = x_selec(ind);
    %     dx = max(x_selected) - min(x_selected);
    %     if dx < tol_m %|| dx > tol_M% we keep 
    %         m = m + 1;
    %         yfinal(m) = y_selec_unic(k);
    %         xfinal(m) = mean(x_selected);
    %     end
    % end
    % plot(xfinal,yfinal)
    % title(':test')
    % disp("a")
    


    % return
    tmp = abs(diff(new_x_final));
    ind = find(tmp>0.2*max(tmp));%make temp higher.
    [v,ii] = max(diff(ind));
    iii = ind(ii):ind(ii+1);
    xxx = new_x_final(iii);
    yyy = new_y_final(iii);
    
    %plot(xxx, yyy, 'dg') %check
    
    hold off;
    
    
    %% FINAL PLOT !!!!!!!!!
    
    % figure
    % imshow(inpict)
    % hold on 
    % plot(xxx, yyy, '.r')
    % hold off;
    % drawnow;
    figure
    imshow(inpict)
    hold on 
    plot(new_x_final,  new_y_final, '.r')
    hold off;
    drawnow;

    toc
    
end










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

%if avgPixelValue < .3 && control_var < 5
%    bluetoothObj.stepF;
%    control_var = control_var + 1;

%mirrordirection = 0 %backwards



%if score == 0 && recover = 0 && mirror direction == 0
    %recover = 1
    %memory  = 0; 
    %bluetoothObj.stepB
    %bluetoothObj.stepB
    %bluetoothObj.stepB
    %bluetoothObj.stepB
%if score == 0 && recover = 0 && mirror direction == 1
    %recover = 1
    %memory = 0; 
    %bluetoothObj.stepF
    %bluetoothObj.stepF
    %bluetoothObj.stepF
    %bluetoothObj.stepF


%bluetoothObj.stepF
%memeory++
%if score != 0
    %recover = 0;
    %if memory <= 4 %NEEDLE IS MOVING "BACKWARDS for mirror"
    %mirrordirection = 0 
    %memory = 0;



