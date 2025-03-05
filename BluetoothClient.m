% Bluetooth Client for Robotic Ultrasound MQP
% Connects to the device via bluetooth and allows commands to be sent via
% MatLab
%
% Author: Emerson Shatouhy


classdef BluetoothClient
    properties
        Device
    end
    
    methods
        function obj = BluetoothClient()
            obj.Device = bluetooth("Ultrasound Device MQP");
            fprintf('Connected to device!\n');
        end

        % Sweeps the stepper back and forth
        function sweep(obj)
            obj.Device.write("S")
        end

        % Steps the stepper forward
        function stepF(obj)
            obj.Device.write("F")
        end
        
        % Steps the stepper backwards
        function stepB(obj)
            obj.Device.write("B")
        end
        function fullRight(obj)
            obj.Device.write("R")
        end
        function fullLeft(obj)
            obj.Device.write("L")
        end
    end
end
