#include "esp_sleep.h"  // Required for deep sleep functions

// 2022/11/24
// Written by Soham Naik - 11/08/2024
// Modified by Zaynab Mourtada - 12/29/2024
// Glove 1 - Single Light Source for OOK (On-Off Keying) Signal
// Single node: Index finger connected to D3 (PWM-capable pin)

int indexFingerLED = 3;  // PWM-capable pin

float shutterRate = 1000.0; // Default shutter rate in Hz
float shutterPeriod;        // Shutter period in microseconds
int pilotGap;               // Delay between each OOK signal
int pulseWidth;             // Width of each OOK bit LIGHT band
int gapWidth;               // Width of each OOK bit DARK band

unsigned long startTime;    // Start time to limit the signal duration

// Splits the shutter period into ON time and OFF time
void calculateTiming() {
  shutterPeriod = (1.0 / shutterRate) * 1e6; // Converting shutter rate to microseconds
  pulseWidth = shutterPeriod * 1.0;          // 50% ON (adjust multiplier if a true 50% duty cycle is needed)
  gapWidth = shutterPeriod * 1.0;            // 50% OFF (adjust multiplier if a true 50% duty cycle is needed)
  pilotGap = shutterPeriod * 7;
}

// Function to emit a gap between OOK signals (if desired)
void emitPilotGap() {
  digitalWrite(indexFingerLED, LOW);    // Turn LED OFF
  delayMicroseconds(pilotGap);
}

// Function to emit a symbol with a specified binary pattern
void emitSymbol(const char* binaryPattern) {
  for (int i = 0; binaryPattern[i] != '\0'; i++) {
    if (binaryPattern[i] == '1') {
      digitalWrite(indexFingerLED, HIGH);  // LED ON for "1" bit
      delayMicroseconds(pulseWidth);
    } else {
      digitalWrite(indexFingerLED, LOW);   // LED OFF for "0" bit
      delayMicroseconds(gapWidth);
    }
  }
}

// Function to generate the OOK signal using a predefined binary pattern
void generateOOKSignal() {
  // Optionally, uncomment the next line if you want to include a pilot gap:
  // emitPilotGap();
  // Choose the desired pattern:
  // emitSymbol("11110000");  // Example pattern for one user
  emitSymbol("10101010");    // Example pattern for another user
}

void setup() {
  pinMode(indexFingerLED, OUTPUT);
  calculateTiming();
  startTime = millis();  // Record the starting time
}

void loop() {
  // Run the OOK signal for 30 seconds (30,000 ms)
  if (millis() - startTime < 30000UL) {
    generateOOKSignal();
  } else {
    // Turn off the LED before shutdown
    digitalWrite(indexFingerLED, LOW);
    // Enter deep sleep mode to effectively "shutdown" the board
    esp_deep_sleep_start();
  }
}