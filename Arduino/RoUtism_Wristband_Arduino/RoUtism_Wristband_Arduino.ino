// define led according to pin diagram in article
const int redLed2 = D2; // there is no LED_BUILTIN available for the XIAO ESP32C3.
const int greenLed4 = D4;
bool ran = false; // used to pause the code

void setup() {
  // initialize digital pin led as an output
  pinMode(redLed2, OUTPUT);
  pinMode(greenLed4, OUTPUT);
}

void loop() {
  if(!ran){
    digitalWrite(redLed2, HIGH);   // turn the Red LED on 
    delay(5000);     // 5 seconds delay
    digitalWrite(redLed2, LOW);   // 0 turn off the Red Led 
    delay(1000);     // 1 second dely
    digitalWrite(greenLed4, HIGH);   // 1  turns on the Green LED
    delay(1000);                    
    digitalWrite(greenLed4, LOW);    // 0  turns off the Green LED
    delay(1000);
    digitalWrite(greenLed4, HIGH);   // 1  turns on the Green LED
    delay(1000);                   
    digitalWrite(greenLed4, LOW);  // 0  turns off the Green LED
    delay(1000);
    ran = true;
  }
  
}
