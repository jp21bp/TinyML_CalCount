### Version 1
# In this version, the following will be implemented:
    # The sensor data will be grouped into 1 sec time window
            # which implies that each "sample" will have a 
            # shape of (50,3), since 50hz ODR => 1 sec has 
            # 50 samples
    # Turn the (50,3) samples into (50,1,3)
        # Since TFLM only has 2d conv operation
    # A small version of DeepConvLSTM will be implemented
        # 3 Conv2d layers, 2 lstms, 3 fully connected 
##NOTE:
    # For some reason, some of the values in the
            # cleaned data have a "x,xxx" instead of 
            # a "x.xxx" format, which means something 
            # with the transformation/engineering is off
    # In future, add dropoff layers in model

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np

os.chdir("C:\\Users\\jwpar\\OneDrive\\Documents\\TML_CalCoProject\\CAPTURE-24\\Version1\\ModelTraining")

CLEAN_DIR = './miniClean'
SPLIT_PERC = 0.67
BATCH_SIZE = 16
SAVED_PATH = '../savedModel'
TIME_STEPS = 50     # Window is 1 sec, with 50 Hz, implies 50 samples in 1 sec
NUM_CHANNELS = 3
files = []

for (dirpath, dirnames, fnames) in os.walk(CLEAN_DIR):
    files.extend(fnames)
    break
# print(files)


train_split = int(len(files)*SPLIT_PERC)
trainset = files[0:train_split]
valset = files[train_split:len(files)]
# print(trainset)
# print(valset)

def singleSampleGen(files):
    for fname in files:
        df = pd.read_csv(CLEAN_DIR + '/' + fname, dtype=float)
        values = df.values
        maxSamples = values.shape[0] // TIME_STEPS
        for i in range(maxSamples - 1):
            orignalInputs = values[TIME_STEPS * i: TIME_STEPS * (i+1),:NUM_CHANNELS]
            inputs = orignalInputs.reshape([TIME_STEPS * 1 * NUM_CHANNELS]) 
                # Kept the "1" as a reminder that shape needs to be (TIME_STEPS, 1, NUM_CHANNELS)
            output = np.mean(values[TIME_STEPS * i: TIME_STEPS * (i+1), NUM_CHANNELS:])
            yield inputs, output

# trainGen = singleSampleGen(trainset)
# for i,(x,y) in enumerate(iter(trainGen)):
#   if(i == 10): break
#   print(x.shape, y.shape)
#   print(x,y) 


trainTfGen = tf.data.Dataset.from_generator(
    generator=lambda: singleSampleGen(trainset),
    output_signature=(
        tf.TensorSpec(shape=(TIME_STEPS * 1 * NUM_CHANNELS), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )
).batch(BATCH_SIZE, drop_remainder=True)

valTfGen = tf.data.Dataset.from_generator(
    generator=lambda: singleSampleGen(valset),
    output_signature=(
        tf.TensorSpec(shape=(TIME_STEPS * 1 * NUM_CHANNELS), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )
).batch(BATCH_SIZE, drop_remainder=True)

# sample = iter(trainTfGen)
# for i,(x,y) in enumerate(sample): ### How to go through entire tf.Dataset.generator
#     print(i)
#     print(x.shape, y.shape)

# list(trainTfGen.take(1))


# Model
CONV_FILTERS = 32
inputs = keras.Input(shape=(TIME_STEPS * 1 * NUM_CHANNELS), batch_size = BATCH_SIZE)
x = layers.Reshape((TIME_STEPS, 1, NUM_CHANNELS))(inputs)
x = layers.Conv2D(filters = CONV_FILTERS, kernel_size=[5,1], strides=[1,1], activation = 'relu')(x)
x = layers.Conv2D(filters = CONV_FILTERS, kernel_size=[5,1], strides=[1,1], activation = 'relu')(x)
x = layers.Conv2D(filters = CONV_FILTERS*2, kernel_size=[5,1], strides=[1,1], activation = 'relu')(x)
x = layers.Reshape((-1,CONV_FILTERS*2))(x)
x = layers.LSTM(units = 64, activation = 'relu', return_sequences=True)(x)
x = layers.LSTM(units = 64, activation = 'relu', return_sequences=False)(x)
x = layers.Dense(units = 32, activation='relu')(x)
x = layers.Dense(units = 8, activation='relu')(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs=inputs, outputs=outputs)


model.compile(loss='mse', optimizer='adam', metrics='mse')
model.summary()

history = model.fit(x=trainTfGen,
                    epochs = 1,
                    verbose = 1,
                    validation_data = valTfGen)


test = np.array([[0.8593265,-0.5249027,	0.29539275],
[0.827795,	-0.5170376,	0.43199173],
[0.8593265,	-0.34400487,	0.36770985],
[0.84356076,	-0.39119563,	0.3114632],
[0.77261496,-0.44625148,	0.2873575],
[0.8120293,	-0.42265612,	0.25521657],
[0.85144365,	-0.37546536,	0.36770985],
[0.91450655,	-0.37546536,	0.45609742],
[0.7568492,	-0.3833305,	0.39181554],
[0.81991214,	-0.47771198,	0.3998508],
[0.77261496,	-0.39119563,	0.2873575],
[0.7962635,	-0.42265612,	0.32753366],
[0.827795,	-0.40692586,	0.31949845],
[0.81991214,	-0.40692586,	0.2873575],
[0.8750922,	-0.37546536,	0.26325178],
[0.9066237,	-0.3833305,	0.1989699],
[0.93027234,	-0.43838635,	0.26325178],
[1.0248667,	-0.35186997,	0.34360415],
[1.1115782,	-0.31254435,	0.34360415],
[0.9539209,	-0.3282746,	0.26325178],
[0.74108344,	-0.37546536,	0.15879373],
[0.7804978,	-0.414791,	0.27128705],
[0.89085793,	-0.3911956,	0.34360415],
[1.0248667,	-0.36760023,	0.3516394],
[0.9618038,	-0.3911956,	0.22307563],
[0.8356779,	-0.36760023,	0.21504039],
[0.92238945,	-0.42265612,	0.23111084],
[0.88297504,	-0.414791,	0.19896992],
[0.90662366,	-0.42265612,	0.35967463],
[0.92238945,	-0.414791,	0.40788603],
[0.8593265,	-0.39906073,	0.32753366],
[0.8435607,	-0.43838635,	0.2873575],
[0.709552,	-0.36760023,	0.46413267],
[0.7332006,	-0.33613974,	0.48020312],
[0.646489, -0.3282746,	0.3757451],
[0.7016691,	-0.3833305,	0.39181554],
[0.7962636,	-0.40692586,	0.35967463],
[0.8987408,	-0.39906073,	0.3516394],
[0.9381552,	-0.46984684,	0.3275337],
[0.93027234,	-0.46984687,	0.27128705],
[0.890858,	-0.4934422,	0.3757451],
[0.8041464,	-0.40692586,	0.38378033],
[0.7253177,	-0.3282746,	0.27932227],
[0.74896634,	-0.47771198,	0.3114632],
[0.88297504,	-0.6192842,	0.20700514],
[1.0248668,	-0.4934422,	0.1989699],
[1.0879296,	-0.414791,	0.33556893],
[1.119461,	-0.5013073,	0.40788603],
[1.0721639,	-0.5013073,	0.3757451],
[0.81202924,	-0.5720935,	0.27932227]
])

test = test.reshape(1,50 * 1 * 3)
pred = model.predict(test) #true label is "3.0"
print('pre-pred')
print(pred)
print('post-pred')


# Following few lines are from colab 
    # https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/experimental_new_converter/Keras_LSTM_fusion_Codelab.ipynb
run_model = tf.function(lambda x: model(x))
# This is important, let's fix the input size.
BATCH_SIZE = 1
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([BATCH_SIZE, TIME_STEPS * 1 * NUM_CHANNELS], model.inputs[0].dtype))
    #Note: adding concrete func changes the layer names when trying to load 
            # and test using "tf.saved_model.load()"
        # This is especially important for input and output layers

model.save(SAVED_PATH,save_format = "tf", signatures=concrete_func)






