/*
  Cal Usage WK
  IMU model
  This one is done using the new "tflite-micro-arduino-examples-main" library
    This new library has LSTM and updates
  
*/

#include <TensorFlowLiteNew.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "preprocess.h"
#include "model_data.h"


namespace{//globals
// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 30 * 1024;
// Keep aligned to 16 bytes for CMSIS
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

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
  SetupLEDs();


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
  static tflite::MicroMutableOpResolver<1> micro_op_resolver;  // NOLINT
  micro_op_resolver.AddFullyConnected();
  

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  interpreter->AllocateTensors();


  TfLiteTensor* model_input = interpreter->input(0);
  // MicroPrintf("Checking Inputs");
  // MicroPrintf("%d", model_input->dims->size);
  // MicroPrintf("%d", model_input->dims->data[0]);
  // MicroPrintf("%d", model_input->dims->data[1]);
  // MicroPrintf("%d", model_input->type);
  // //types enum labels found under:
  //   // "portable_type_to_tflitetype"
  // if (model_input->type == kTfLiteInt8){
  //   MicroPrintf("Int8");
  // }
  // if (model_input->type == kTfLiteFloat32){
  //   MicroPrintf("float");
  // }
  // if (model_input->type == kTfLiteFloat64){
  //   MicroPrintf("float64");
  // }


  TfLiteTensor* model_output = interpreter->output(0);
  // MicroPrintf("Checking Outputs");
  // MicroPrintf("%d", model_output->dims->size);
  // MicroPrintf("%d", model_output->dims->data[0]);
  // MicroPrintf("%d", model_output->dims->data[1]);
  // MicroPrintf("%d", model_output->type);
  // if (model_output->type == kTfLiteInt8){
  //   MicroPrintf("Int8");
  // }
  // if (model_output->type == kTfLiteFloat32){
  //   MicroPrintf("float");
  // }
  // if (model_output->type == kTfLiteFloat64){
  //   MicroPrintf("float64");
  // }


  startTime = millis();
}

void loop() {
  float x,y,z;
  if(!IMU.readAcceleration(x, y, z)){
    MicroPrintf("Failed to read acceleration data");
  } 

  TfLiteTensor* model_input = interpreter->input(0);
  model_input->data.f[0] = x;
  model_input->data.f[1] = y;
  model_input->data.f[2] = z;

  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed");
    return;
  }

  TfLiteTensor* output = interpreter->output(0);
  float met = output->data.f[0];



  // MicroPrintf("Ax: %f Ay: %f Az: %f : %f", x, y, z, met);
  // delay(1000);


  counter += 1;
  sumMet += met;

  if(counter % 5000 == 0){
    endTime = millis();
    timeBw = endTime - startTime; //milliseconds passed
    secPassed = timeBw / 1000; // seconds passed
    minPassed = secPassed / 60; //mins passed
    avgMetSec = sumMet / secPassed;
    currAvg = avgMetSec;
    MicroPrintf("Summed: %f, secs passed: %f, avg: %f", sumMet, secPassed, avgMetSec);

    digitalWrite(LEDR, HIGH);
    digitalWrite(LEDG, HIGH);
    if(currAvg > prevAvg){
      digitalWrite(LEDG, LOW);
    }
    if(currAvg < prevAvg){
      digitalWrite(LEDR, LOW);
    }
    prevAvg = currAvg;
  }


}
