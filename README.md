# MQP_Needle_Insertion

This code takes images from the verasonics machine and uses ROS to tranfer them over. The images are then processed and displayed and control an be applied depending on your own setttings and what it is configured for. The following instrucutions are how to run the code

Install matlab 
Dependencies include Matlab ROS toolbox

the host computer and Verasonics need to be on the same network for ROS tranfers to work

# Verasonics 
tranfer Host files to Verasonics computer 
  ytSetUpP4_1...: Run code fo verasonics
  yt_ROSinit: check the ROS master URI as well as the local_ip
  yt_ROSsync: defines buffer and type to send messages
# Home
Ethan3_test: can be run without any setup above and just process the images in Data_New
Ethan4_exp: Once ROS and ESP connected it can be called to run and process the images in real time as long as the Verasonics machine is sending images.
Move_test.m: Makes sure motor moves

BluetoothClient.m: creates connection to ESP
Connect computer to verasonics network 
Connect ESP with bluetooth
# ESP32
stepperCode.ino: This is used to run our motor which in our case was an Adafruit stepper
