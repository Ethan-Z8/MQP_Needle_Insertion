#include <AccelStepper.h>
#include "BluetoothSerial.h"

// Check if Bluetooth is properly enabled
#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth is not enabled! Please run `make menuconfig` and enable it
#endif

BluetoothSerial SerialBT;

// Define pin connections
const int dirPin = 32;   // Direction pin
const int stepPin = 33;  // Step pin
const int dirPin2 = 25;  // Direction pin for second coil
const int stepPin2 = 26; // Step pin for second coil

// Define motor interface type
#define MOTOR_INTERFACE_TYPE 4 // 4 wire stepper with 2 control wires per coil

// Create instance
AccelStepper stepper(MOTOR_INTERFACE_TYPE, stepPin, dirPin, stepPin2, dirPin2);

// Control variables
bool sweepMode = false;
int sweepSteps = 30;  // Number of steps in each direction during sweep
int manualSteps = 1;  // Number of steps for manual control
char cmd;             // Command character from Bluetooth

void setup() {
    Serial.begin(9600);
    SerialBT.begin("Ultrasound Device MQP"); // Bluetooth device name
    
    // Set motor parameters
    stepper.setMaxSpeed(100);     // 100 steps per second
    stepper.setAcceleration(50);  // 50 steps/second²
    stepper.setSpeed(50);         // 50 steps per second
    
    Serial.println("Bluetooth Stepper Control Ready!");
}

void loop() {
    // Check for Bluetooth commands
    if (SerialBT.available()) {
        cmd = SerialBT.read();
        processCommand(cmd);
    }
    
    // Handle sweep mode if active
    if (sweepMode) {
        runSweep();
    }
    
    // Always run the stepper
    stepper.run();
}

void processCommand(char command) {
    switch (command) {
        case 'F': // Forward step
            stepper.move(manualSteps);
            SerialBT.println("Stepping forward");
            break;
            
        case 'B': // Backward step
            stepper.move(-manualSteps);
            SerialBT.println("Stepping backward");
            break;
            
        case 'S': // Toggle sweep mode
            sweepMode = !sweepMode;
            if (sweepMode) {
                SerialBT.println("Sweep mode activated");
            } else {
                SerialBT.println("Sweep mode deactivated");
                stepper.stop(); // Stop current movement
            }
            break;
        case 'R':
            stepper.moveTo(30);
            SerialBT.println("Full Right");
            break;
        case 'L':            
            stepper.moveTo(0);
            SerialBT.println("Full Left");
            break;

        case '+': // Increase manual step size
            manualSteps++;
            SerialBT.print("Manual step size: ");
            SerialBT.println(manualSteps);
            break;
            
        case '-': // Decrease manual step size
            if (manualSteps > 1) {
                manualSteps--;
                SerialBT.print("Manual step size: ");
                SerialBT.println(manualSteps);
            }
            break;
            
        case 'H': // Help menu
            printHelp();
            break;
    }
}

void runSweep() {
    static bool sweepDirection = true; // true = clockwise, false = counterclockwise
    // If we've reached the target position
    if (stepper.distanceToGo() == 0) {
        if (sweepDirection) {
            stepper.move(sweepSteps);
        } else {
            stepper.move(-sweepSteps);
        }
        sweepDirection = !sweepDirection;
        delay(1000); // Pause at each end
    }
}

void printHelp() {
    SerialBT.println("\n--- Stepper Control Commands ---");
    SerialBT.println("F: Step Forward");
    SerialBT.println("B: Step Backward");
    SerialBT.println("S: Toggle Sweep Mode");
    SerialBT.println("+: Increase manual step size");
    SerialBT.println("-: Decrease manual step size");
    SerialBT.println("H: Show this help menu");
    SerialBT.println("---------------------------\n");
}