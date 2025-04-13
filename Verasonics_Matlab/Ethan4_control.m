clear all
close all
clc

rosshutdown;

rosinit
% inpict_res = cell(1:30);
% final_res = cell(1:30);
% results = struct;



for asdf = 1:1000
    inpict_res{asdf} =[];
    final_res{asdf} = [];
    pix_val{asdf} = [];
end

recvoer = 0;
direction = "right";
step = 0;

results = struct;

control_var = 0;
PAIMG_sub = rossubscriber('PA_IMG', 'std_msgs/Float64MultiArray');
k = 1; t1=0;

tic;


bluetoothObj = BluetoothClient();
disp("in while");
bluetoothObj.fullRight;
for i = 1:30
PAIMG_msg = receive(PAIMG_sub);
PAIMG = PAIMG_msg.Data;
PAIMG0 = reshape(PAIMG,570,500);
PAIMG1 = PAIMG0./max(PAIMG0(:));
PAIMG2 = db(PAIMG1);
arr = PAIMG2;
arr(arr < -50) = -50;
arr = arr + 50;
arr = arr / 50;
PAIMG2 = arr;
%IMPORTANT
% imagesc(PAIMG2);% dynamic range -50,0
colormap gray;
t = toc;
% fprintf(['Frame #',num2str(k),'  F=',num2str(1/(t-t1)),'Hz\n'])
k = k+1;
t1 = t;
colormap gray;
rstart = 150;%was 
rend = 470;%up to 570 
cstart = 225;%200
cend = 275;%300 good
disp("move to the Right");
fprintf('Loop number: %d\n', i); 
ROI_image = ROI_creation(PAIMG2,rstart,rend,cstart,cend);
inpict = ROI_image;
inpict = inpict - mean(inpict,2);
% low pass elliptical filering of the input image (to remove further the
% salt pepper noise) - adjust filter cut off and order to your own
% preferences
%rearange Fourier transform
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
%IMPORTANT
% figure
% subplot(2,1,1),imshow(inpict)
% subplot(2,1,2),imshow(outpict)
% title(['Partial + Image number: ', num2str(k)])
% drawnow;
% %outpic is blurred 

%% attempt to extract the white segment
% Threshold the image to create a binary image
binaryImage = outpict > .5 * max(outpict(:)); 

% Display the binary image
%IMPORTANT FIGURE
% figure;
% imshow(binaryImage);
% title('Binary Image (White Segments)');
% hold on;
% drawnow;

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
    %IMPORTANT
    % plot(x,y, '.', x_selec, y_selec, '.r')
    % title('Boundary points Highlighted');
    % drawnow;
    
    %find the avg pix values for the threshold values

    
    
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





    for a = 1:(numel(yfinal) - 1)
        x  = a - y;
        current_y = yfinal(x);
        next_y = yfinal(a+1);

        % current_x = xfinal(x);
        % next_x = xfinal(i+1);
        % abs(current_x - next_x)
        if (next_y - current_y) > 15%check dist
            % numel(yfinal);
            new_y_final(a+1-y) = [];
            new_x_final(a+1-y) = [];
            y = y + 1 ;
        end
    end
    
    % p = polyfit(new_x_final, new_y_final, 0);
    % y_fit = polyval(p, x);
    % y_fit = polyval(p, x);
    pixelValues = inpict(sub2ind(size(binaryImage), new_y_final, new_x_final));
    

    avgPixelValue = mean(pixelValues);
    accuracyScore = avgPixelValue * log(1 + numPixels);


    pix_val{i} = accuracyScore;
    disp("acc score: " + accuracyScore)



    %IMPORTANT ISH
    % plot(new_x_final,new_y_final);
    % hold off;

    
    
    %% FINAL PLOT !!!!!!!!!
    
    inpict_res{i} = inpict;
    val = {new_x_final,new_y_final};
    final_res{i} = val;
    % results(i).image = inpict; % Original image
    % results(i).coordinates = [new_x_final(:), new_y_final(:)]; % Nx2 array of coordinates
    % results(i).figure_handle = figure;


    inpict_f = inpict_res{i};
    val = final_res{i};
    new_x_final_f = val{1};
    new_y_final_f = val{2};
    pix_val_f = pix_val{i};
    %IMPORTANT
    imshow(inpict_f);
    figure
    imshow(inpict_f);
    hold on;
    plot(new_x_final_f, new_y_final_f, '.r');
    hold off;
    title(['Image number: ', num2str(i)]);
    disp(pix_val_f);
    drawnow;
    

    % toc

    % bluetoothObj.stepB
    % disp("STEP BACK")
end
if(accuracyScore > 0.4)
    disp("NEEDLE FOUND!")
    break
end
bluetoothObj.stepB
disp(["next step! Step done:",num2str(i)])
pause(1)
%this is the for loop
end 







while(1) 
    PAIMG_msg = receive(PAIMG_sub);
    PAIMG = PAIMG_msg.Data;
    PAIMG0 = reshape(PAIMG,570,500);
    PAIMG1 = PAIMG0./max(PAIMG0(:));
    PAIMG2 = db(PAIMG1);
    arr = PAIMG2;
    arr(arr < -50) = -50;
    arr = arr + 50;
    arr = arr / 50;
    PAIMG2 = arr;
    %IMPORTANT
    % imagesc(PAIMG2);% dynamic range -50,0
    colormap gray;
    t = toc;
    % fprintf(['Frame #',num2str(k),'  F=',num2str(1/(t-t1)),'Hz\n'])
    k = k+1;
    t1 = t;
    colormap gray;
    rstart = 150;%was 
    rend = 470;%up to 570 
    cstart = 225;%200
    cend = 275;%300 good
    disp("move to the Right");
    fprintf('Loop number: %d\n', i); 
    ROI_image = ROI_creation(PAIMG2,rstart,rend,cstart,cend);
    inpict = ROI_image;
    inpict = inpict - mean(inpict,2);
    % low pass elliptical filering of the input image (to remove further the
    % salt pepper noise) - adjust filter cut off and order to your own
    % preferences
    %rearange Fourier transform
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
    %IMPORTANT
    % figure
    % subplot(2,1,1),imshow(inpict)
    % subplot(2,1,2),imshow(outpict)
    % title(['Partial + Image number: ', num2str(k)])
    % drawnow;
    % %outpic is blurred 
    
    %% attempt to extract the white segment
    % Threshold the image to create a binary image
    binaryImage = outpict > .5 * max(outpict(:)); 
    
    % Display the binary image
    %IMPORTANT FIGURE
    % figure;
    % imshow(binaryImage);
    % title('Binary Image (White Segments)');
    % hold on;
    % drawnow;
    
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
        %IMPORTANT
        % plot(x,y, '.', x_selec, y_selec, '.r')
        % title('Boundary points Highlighted');
        % drawnow;
        
        %find the avg pix values for the threshold values
    
        
        
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
    
    
    
    
    
        for a = 1:(numel(yfinal) - 1)
            x  = a - y;
            current_y = yfinal(x);
            next_y = yfinal(a+1);
    
            % current_x = xfinal(x);
            % next_x = xfinal(i+1);
            % abs(current_x - next_x)
            if (next_y - current_y) > 15%check dist
                % numel(yfinal);
                new_y_final(a+1-y) = [];
                new_x_final(a+1-y) = [];
                y = y + 1 ;
            end
        end
        
        % p = polyfit(new_x_final, new_y_final, 0);
        % y_fit = polyval(p, x);
        % y_fit = polyval(p, x);
        pixelValues = inpict(sub2ind(size(binaryImage), new_y_final, new_x_final));
        
    
        avgPixelValue = mean(pixelValues);
        accuracyScore = avgPixelValue * log(1 + numPixels);
    
    
        pix_val{i} = accuracyScore;
        disp("acc score: " + accuracyScore)
    
    
    
        %IMPORTANT ISH
        % plot(new_x_final,new_y_final);
        % hold off;
    
        
        
        %% FINAL PLOT !!!!!!!!!
        
        inpict_res{i} = inpict;
        val = {new_x_final,new_y_final};
        final_res{i} = val;
        % results(i).image = inpict; % Original image
        % results(i).coordinates = [new_x_final(:), new_y_final(:)]; % Nx2 array of coordinates
        % results(i).figure_handle = figure;
    
    
        inpict_f = inpict_res{i};
        val = final_res{i};
        new_x_final_f = val{1};
        new_y_final_f = val{2};
        pix_val_f = pix_val{i};
        %IMPORTANT
        imshow(inpict_f);
        figure
        imshow(inpict_f);
        hold on;
        plot(new_x_final_f, new_y_final_f, '.r');
        hold off;
        title(['Image number: ', num2str(i)]);
        disp(pix_val_f);
        drawnow;
        
    
        % toc
    
        % bluetoothObj.stepB
        % disp("STEP BACK")
    end
    %from found state to lost and moving state
    if(accuracyScore < 0.5 && recover == 0)
        disp("NEEDLE Lost!")
        recover = 1;
        %step = 0;
        if(direction == "right")
            % disp("direction: ", direction)
            bluetoothObj.stepB
            bluetoothObj.stepB
            bluetoothObj.stepB
            bluetoothObj.stepB
        end
        if(direction == "left")
            bluetoothObj.stepF
            bluetoothObj.stepF
            bluetoothObj.stepF
            bluetoothObj.stepF
        end
        disp("direction: ", direction)
    end
    if(recover == 1 && direction == "right")
        if(accuracyScore > 0.5)
            disp("NEEDLE FOUND")
            if(step > 4)
                dirrection = "left";
            end
            recover = 0;
            step = 0;
        else
            step = step + 1;
            bluetoothObj.stepF
        end
    end
    if(recover == 1 && direction == "left")  
        if(accuracyScore > 0.5)
            disp("NEEDLE FOUND")
            if(step > 4)
                dirrection = "right";
            end
            step = 0;
            recover = 0;
        else
            step = step + 1;
            bluetoothObj.stepB
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














disp("Displaying")
%IMPORTANT
%can copy paste this to display after the run
% for i = 1:length(inpict_res)
%     inpict = inpict_res{i};
%     val = final_res{i};
%     new_x_final = val{1};
%     new_y_final = val{2};
%     pix_val_f = pix_val{i};
    
%     figure;
%     imshow(inpict);
%     hold on;
%     plot(new_x_final, new_y_final, '.r');
%     hold off;
%     title(['Image number: ', num2str(i)]);
%     disp(pix_val_f);
%     drawnow;
% end




% pause(100000)
% disp("FIRST SET OVER")

% end







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
  %IMPORTANT
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



