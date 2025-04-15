# TinyML_CalCount
This is a Tiny ML project exploring the Tiny MLOps pipeline through the development of a simple but functional regression model on an embedded device.
</br></br>
This project involves developing a regression model to estimate calorie usage during workouts, aiming to provide affordable tracking devices for low-resource communities. It delves into multiple facets of the Tiny MLOps pipelines, including: data engineering, model training, and model deployment onto an embedded device. By leveraging the CAPTURE-24 dataset, which offers diverse accelerometer data and activity annotations, the model will be trained to provide accurate calorie predictions. The device's impact will be assessed by comparing participation rates in activities with and without the device, and the primary evaluation metric will be Mean Squared Error (MSE). The goal is to enhance physical activity engagement, particularly among youth, by creating a low-cost alternative to existing solutions like Fitbit and Apple Watch, while ensuring data accuracy and adaptability to various workout types and environments.

# Directory Breakdown
The following directories contain the following:
<ul>
  <li>Arduino Model: files associates with the model's deployment onto an Arduino Nano 33 BLE Sense embedded device.</li>
  <li>Model Conversion: file used for converting the trained model onto ".tflite" format, as well as the ".tflite" model and the C-byte array model.</li>
  <li>Model Training: code for training the model, as well as reduced version of the extended datasets used to train the model. </li>
  <li>saved Model: contains the trained model's code using the "savedModel" format.</li>
</ul>

