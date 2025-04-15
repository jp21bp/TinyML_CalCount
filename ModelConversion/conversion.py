import tensorflow as tf
import numpy as np
import os


# SEtting saved path
savedPath = os.path.join(os.getcwd(),"..\savedModel")
print(savedPath)
print(os.listdir(savedPath))


#Checking model works
model = tf.saved_model.load(savedPath)

print(type(model))

test = np.array([-0.84360206, 0.2607037, -0.100506194])
test = test.reshape(1,3)

pred = model.signatures['serving_default'](input=test)
print(pred['last'].numpy())


# COnversion: non-optimized
float_converter = tf.lite.TFLiteConverter.from_saved_model(savedPath)
float_tflite_model = float_converter.convert()
float_tflite_model_size = open('./non-optimized.tflite', "wb").write(float_tflite_model)
print("Float model is %d bytes" % float_tflite_model_size)


#converion: optimized
converter = tf.lite.TFLiteConverter.from_saved_model(savedPath)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
tflite_model_size = open('./optimized.tflite', "wb").write(tflite_model)
print("Quantized model is %d bytes" % tflite_model_size)


# FOr convesion from .tflite to .cc, check out:
    # TML, C3.5(KeywordSpotting).FavoriteKeywords