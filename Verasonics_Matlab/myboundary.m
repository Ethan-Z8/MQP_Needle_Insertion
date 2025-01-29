%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x_selec,y_selec] = myboundary(x,y)

    % do a "tight_boundary" operation once in X direction and once in Y
    % direction and keep only unique pairs
    [x_out,y_out] = tight_boundary(x,y);
    [y_out2,x_out2] = tight_boundary(y,x);

    % find unique x , y pairs 
    xx = [x_out(:);x_out2(:)];
    yy = [y_out(:);y_out2(:)];

    xy = unique([xx yy],'rows');
    x_selec = xy(:,1);
    y_selec = xy(:,2);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [x_out,y_out] = tight_boundary(x,y)

%https://fr.mathworks.com/matlabcentral/answers/299796-tight-boundary-around-a-set-of-points


    %split data into classes and find max/min for each class
    class_label = unique(x);
    upper_boundary = zeros(size(class_label));
    lower_boundary = zeros(size(class_label));
    for idx = 1:numel(class_label)
        class = y(x == class_label(idx));
        upper_boundary(idx) = max(class);
        lower_boundary(idx) = min(class);
    end

    left_boundary = y(x == class_label(1));
    right_boundary = y(x == class_label(end));

    % left_boundary
    x1 = class_label(1)*ones(size(left_boundary));
    y1 = left_boundary;

    % right_boundary
    x2 = class_label(end)*ones(size(right_boundary));
    y2 = right_boundary;

    % top boundary
    x3 = class_label;
    y3 = upper_boundary;

    % bottom boundary
    x4 = class_label;
    y4 = lower_boundary;

    x_out = [x1(:);x2(:);x3(:);x4(:);];
    y_out = [y1(:);y2(:);y3(:);y4(:);];

end
