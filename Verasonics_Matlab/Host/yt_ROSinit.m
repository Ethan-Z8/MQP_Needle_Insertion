global PA_IMG_pub PA_IMG_msg local_ip
rosshutdown;
URI = '';
ip = 'xxx.xxx.x.xxx'
ros_master_uri = URI; % need to check everytime

local_ip = ip;                    % need to check everytime

setenv('ROS_MASTER_URI', ros_master_uri);
setenv('ROS_IP', local_ip);
rosinit

PA_IMG_pub = rospublisher('PA_IMG', 'std_msgs/Float64MultiArray');
PA_IMG_msg = rosmessage(PA_IMG_pub);

