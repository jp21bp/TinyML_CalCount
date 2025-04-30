import os
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np

os.chdir("C:\\Users\\jwpar\\OneDrive\\Documents\\TML_CalCoProject\\CAPTURE-24\\ModelTraining")

CLEAN_DIR = './miniClean'
SPLIT_PERC = 0.67
BATCH_SIZE = 16
SAVED_PATH = 'savedModel'
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
    # As a regular python generator, it's fcn
        # is to create ONE sample from the target
        # dataset(s)
    # i.e, if two datasets (read one at a time)
        # each have 32 sample and this generator
        # returns (16,3), then ONE SAMPLE will
        # have shape of (16,3)
    # This is imporant when it comes to the
        # TF generator
    # In particular, a non batch tf generator
        # will work exactly the same as the 
        # python generator, but the returned
        # SINGLE sample will be tf.tensors(?)
    # BUT, when tryin to batch a tf generator,
        # it will take into account what makes
        # up a SINGLE SAMPLE
    # Ex: a batch size of 4, with the example above,
        # means that one batch will consist of 4
        # SINLGE SAMPLES, which means that ONE output/batch
        # of the batched tf generator will be
        # (batchSize, singleSampleSize) ==
        # (4, 16, 3)
    counter = 0
    for fname in files:
        counter +=1
        df = pd.read_csv(CLEAN_DIR + '/' + fname)
        values = df.values
        for i in range(values.shape[0]):
            inputs = values[i,:-1]
            outputs = values[i,-1]
            yield inputs, outputs

# trainGen = singleSampleGen(trainset)   
# valGen = singleSampleGen(valset)

# for (x,y) in trainGen: print(x.shape)


trainTfGen = tf.data.Dataset.from_generator(
    generator=lambda: singleSampleGen(trainset),
    output_types=(tf.float32, tf.float32),
    output_shapes=([3,],[])
)

batchedTrainTfGen = trainTfGen.batch(BATCH_SIZE)


valTfGen = tf.data.Dataset.from_generator(
    generator=lambda: singleSampleGen(valset),
    output_types=(tf.float32, tf.float32),
    output_shapes=([3,],[])
)

batchedValTfGen = valTfGen.batch(BATCH_SIZE)

# for i,(x,y) in enumerate(iter(trainGenData)): ### How to go through entire tf.Dataset.generator
#     print(i)
#     print(x.shape, y.shape)


model = tf.keras.Sequential([
    layers.Input(shape=(3,), name='input'),
    layers.Dense(25, activation='relu', name='lay1'),
    layers.Dense(15, activation='relu', name='lay2'),
    layers.Dense(5, activation='relu', name='lay3'),
    layers.Dense(1, name='last')
    ])
model.compile(loss='mse', optimizer='adam', metrics = ['mse'])

# model.summary()

history = model.fit(x=batchedTrainTfGen,
                    epochs = 20,
                    verbose = 1,
                    validation_data = batchedValTfGen)

test = np.array([-0.84360206, 0.2607037, -0.100506194])
test = test.reshape(1,3)
pred = model.predict(test)
print(pred)

savePath = os.path.join(os.getcwd(), SAVED_PATH)
tf.saved_model.save(model, savePath)