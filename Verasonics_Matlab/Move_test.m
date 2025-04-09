


%testing the movment

% bluetoothlist

% bluetoothObj = BluetoothClient();  

bluetoothObj.fullRight
pause(5)
disp("ready");


for i = 1:30
fprintf('value of x: %d\n', i); 
bluetoothObj.stepB
disp("next step")
pause(1)
end
