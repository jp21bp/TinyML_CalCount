/*
  File used for setup
*/

#include <Arduino_LSM9DS1.h>

namespace{

  float accel_sample_rate = 0.0f;

  void SetupLEDs(){
  // Init LEDs as an output
  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);

  digitalWrite(LEDR, HIGH);
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDB, HIGH);
  }




  
}

