/*
  CalCount Version 1
  This version will determine the MET of
    values collected during 1-sec window intervals
  Since the IMU has an Output Data Rate (ODR) of
    50Hz, then each window will contain 50 3-axis samples
  Additionally, since Tensorflow Lite Micro does not 
    support 1D Convolution layer, then the window sample
    will need to be reshaped from (50,3) to (50,1,3) in
    order to perform 2D convolution

  Future works:
    Instead of doing 1-sec window without overlap, do
      1-sec windows with 500ms sec overlap
    This is necessary bc movement is fluid and not rigid, which
      means that previous met predictions/ workout movements will
      influence the upcoming met prediction
    I.e., met movement is more like (12, 12.5, 13) and not
      (12, 13), for example

  
  Notes:
    When cross checking the accelerometer data, it orks
      better with "Serial.print" than with "MicroPrintf"
    The latter shows off values

*/

#include <TensorFlowLiteNew.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <Arduino_LSM9DS1.h>

#include "model_data.h"


namespace{//globals
// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 62 * 1024;
// Keep aligned to 16 bytes for CMSIS
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

const int time_steps = 50;
const int sensor_channels = 3;

float accel_buffer[time_steps*sensor_channels]; // used for gathering in (50,1,3) accel data
float* model_input_buffer = nullptr; //used to get address of model input
  // will be used to transfer accel_buffer into model_input

int counter = 0;  //Used for printing MET periodiacally
float sumMet = 0.0; //Used for summing METs
float avgMetSec = 0.0; //Used for averageing METs
float secPassed = 0.0;
float minPassed = 0.0;

float prevAvg = 0.0;
float currAvg = 0.0;

unsigned long startTime = 0;
unsigned long endTime = 0;
unsigned long timeBw = 0;

}


void setup() {
  //Init Serial
  tflite::InitializeTarget();
  MicroPrintf("Begin");

  if (!IMU.begin()) {
    MicroPrintf("Failed to initialized IMU!");
    while (true) {
      // NORETURN
    }
  }

  IMU.setContinuousMode();

  // // Init LEDs as an output
  // pinMode(LED_BUILTIN, OUTPUT);
  // pinMode(LEDR, OUTPUT);
  // pinMode(LEDG, OUTPUT);
  // pinMode(LEDB, OUTPUT);

  // digitalWrite(LEDR, HIGH);
  // digitalWrite(LEDG, HIGH);
  // digitalWrite(LEDB, HIGH);


  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  static tflite::MicroMutableOpResolver<15> micro_op_resolver;  // NOLINT
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddUnidirectionalSequenceLSTM();
  micro_op_resolver.AddWhile();
  micro_op_resolver.AddStridedSlice();
  micro_op_resolver.AddLess();
  micro_op_resolver.AddAdd();
  micro_op_resolver.AddGather();
  micro_op_resolver.AddSplit();
  micro_op_resolver.AddRelu();
  micro_op_resolver.AddLogistic();
  micro_op_resolver.AddMul();
  micro_op_resolver.AddConcatenation();
  micro_op_resolver.AddSlice();

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  interpreter->AllocateTensors();


  int size = 0;
  TfLiteTensor* model_input = interpreter->input(0);
  MicroPrintf("Checking Inputs");
  size = model_input->dims->size;
  for(int i = 0; i < size; i++){
    MicroPrintf("data[%d] = %d", i, model_input->dims->data[i]);
  }

  TfLiteTensor* model_output = interpreter->output(0);
  MicroPrintf("Checking Outputs");
  size = model_output->dims->size;
  for(int i = 0; i < size; i++){
    MicroPrintf("data[%d] = %d", i, model_output->dims->data[i]);
  }


  //Initializing accel_buffer
  for(int n = 0; n < time_steps*sensor_channels; n++){
    accel_buffer[n] = 0;
  }


  //Setting up model_input_buffer
  model_input_buffer = model_input->data.f;



  // startTime = millis();


}

void loop() {
  float x, y, z, met;

  // //block below works
  // IMU.readAcceleration(x, y, z); 
  // // MicroPrintf("Ax:%d Ay:%d Az:%d", x, y, z);
  // Serial.print("Ax:");
  // Serial.print(x);
  // Serial.print(' ');
  // Serial.print("Ay:");
  // Serial.print(y);
  // Serial.print(' ');
  // Serial.print("Az:");
  // Serial.println(z);


  // if (IMU.accelerationAvailable()) {//block works
  //   IMU.readAcceleration(x, y, z);

  //   Serial.print("Ax:");
  //   Serial.print(x);
  //   Serial.print(' ');
  //   Serial.print("Ay:");
  //   Serial.print(y);
  //   Serial.print(' ');
  //   Serial.print("Az:");
  //   Serial.println(z);
  // }


    // // "Serial.print" works better than "MicroPrintf"
    // if (IMU.accelerationAvailable()) {
    //   IMU.readAcceleration(x, y, z);
    //   // MicroPrintf("Ax:%f Ay:%f Az:%f", x, y, z);
    //   // MicroPrintf("Ax:%f", x);
    //   // MicroPrintf("Ay:%f", y);
    //   // MicroPrintf("Az:%f", z);
    //   // Serial.print("Ax:");
    //   // Serial.print(x);
    //   // Serial.print(' ');
    //   // Serial.print("Ay:");
    //   // Serial.print(y);
    //   // Serial.print(' ');
    //   // Serial.print("Az:");
    //   // Serial.println(z);
    // }


  TfLiteTensor* model_input = interpreter->input(0);
  if(IMU.accelerationAvailable()){
    IMU.readAcceleration(x, y, z);

    model_input->data.f[3 * counter] = x;
    model_input->data.f[3 * counter + 1] = y;
    model_input->data.f[3 * counter + 2] = z;
    
    // model_input_buffer[3 * counter] = x;
    // model_input_buffer[3 * counter + 1] = y;
    // model_input_buffer[3 * counter + 2] = z;

    // Serial.print("Ax:");
    // Serial.print(model_input_buffer[3 * counter]);
    // Serial.print(' ');
    // Serial.print("Ay:");
    // Serial.print(model_input_buffer[3 * counter + 1]);
    // Serial.print(' ');
    // Serial.print("Az:");
    // Serial.println(model_input_buffer[3 * counter + 2]);

    counter++;

    if(counter == 50){
      counter = 0; 

      // Serial.println(model_input_buffer);
      // MicroPrintf("%f", model_input_buffer);
      // for (int i = 0; i < time_steps*sensor_channels + 5; ++i){
      //   Serial.print(i);
      //   Serial.print("  ");
      //   Serial.println(model_input_buffer[i]);
      // }

      //Run model
      TfLiteStatus invoke_status = interpreter->Invoke();
      if (invoke_status != kTfLiteOk) {
        MicroPrintf("Invoke failed");
        return;
      }

      // Obtain a pointer to the output tensor
      TfLiteTensor* output = interpreter->output(0);
      
      met = output->data.f[0];

      Serial.print("MET: ");
      Serial.println(met);

    }


  }
  // counter++;
  // return;
}
