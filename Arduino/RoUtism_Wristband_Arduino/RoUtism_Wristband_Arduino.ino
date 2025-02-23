#include "esp_sleep.h"  // Required for deep sleep functions 

const int redLED = D2;    // Red LED pin
const int greenLED = D4;  // Green LED pin

float shutterRate = 5000.0; // Set frequency to 5 kHz
float shutterPeriod;        // Shutter period in microseconds
int pulseWidth;             // Width of each ON pulse (50% duty cycle)
int gapWidth;               // Width of each OFF gap (50% duty cycle)

unsigned long loopStartTime; // To mark the beginning of the 45-second cycle

// Calculate timing parameters based on shutterRate (assumes 50% duty cycle)
void calculateTiming() {
  shutterPeriod = (1.0 / shutterRate) * 1e6; // period in microseconds
  pulseWidth = shutterPeriod / 2;            // 50% ON time
  gapWidth = shutterPeriod / 2;              // 50% OFF time
}

// Emit an OOK symbol using the specified binary pattern on a given LED pin
// '1' turns the LED ON for pulseWidth and '0' turns it OFF for gapWidth.
void emitSymbol(int ledPin, const char* binaryPattern) {
  for (int i = 0; binaryPattern[i] != '\0'; i++) {
    if (binaryPattern[i] == '1') {
      digitalWrite(ledPin, HIGH);
      delayMicroseconds(pulseWidth);
    } else {
      digitalWrite(ledPin, LOW);
      delayMicroseconds(gapWidth);
    }
  }
}

// Generate one OOK sequence on the given LED (here, greenLED)
void generateOOKSignal(int ledPin) {
  // Using a sample binary pattern "10101010" as an example.
  emitSymbol(ledPin, "10101010");
}

// Create a red preamble by toggling the red LED at 5 kHz for 1 second.
void redPreamble() {
  unsigned long start = micros();
  while ((micros() - start) < 1000000UL) { // run for 1 second
    digitalWrite(redLED, HIGH);
    delayMicroseconds(pulseWidth);
    digitalWrite(redLED, LOW);
    delayMicroseconds(gapWidth);
  }
}

void setup() {
  pinMode(redLED, OUTPUT);
  pinMode(greenLED, OUTPUT);
  calculateTiming();
  loopStartTime = millis();  // Record the starting time of the 45-second cycle
}

void loop() {
  // Continue the process until 45 seconds have passed
  while (millis() - loopStartTime < 45000UL) {
    // 1. Emit the red preamble.
    redPreamble();

    // 2. Emit 5 OOK sequences on the green LED.
    for (int i = 0; i < 5; i++) {
      generateOOKSignal(greenLED);
      // Optionally, add a short delay between sequences.
      delay(100);
    }
  }

  // Turn off both LEDs after 45 seconds.
  digitalWrite(redLED, LOW);
  digitalWrite(greenLED, LOW);
  
  // If you want to use deep sleep, you can uncomment the next line.
  esp_deep_sleep_start();
}
