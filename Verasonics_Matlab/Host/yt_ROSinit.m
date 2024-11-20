global PA_IMG_pub PA_IMG_msg local_ip
rosshutdown;

ros_master_uri = 'http://192.168.0.197:11311'; % need to check everytime

local_ip = '192.168.0.157';                    % need to check everytime

setenv('ROS_MASTER_URI', ros_master_uri);
setenv('ROS_IP', local_ip);
rosinit

PA_IMG_pub = rospublisher('PA_IMG', 'std_msgs/Float64MultiArray');
PA_IMG_msg = rosmessage(PA_IMG_pub);

