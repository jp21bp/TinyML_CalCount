# TinyML_CalCount
## English
This is a Tiny ML project exploring the Tiny MLOps pipeline through the development of a simple but functional regression model on an embedded device.
</br></br>
This project involves developing a regression model to estimate calorie usage during workouts, aiming to provide affordable tracking devices for low-resource communities. It delves into multiple facets of the Tiny MLOps pipelines, including: data engineering, model training, and model deployment onto an embedded device. By leveraging the CAPTURE-24 dataset, which offers diverse accelerometer data and activity annotations, the model will be trained to provide accurate calorie predictions. The device's impact will be assessed by comparing participation rates in activities with and without the device, and the primary evaluation metric will be Mean Squared Error (MSE). The goal is to enhance physical activity engagement, particularly among youth, by creating a low-cost alternative to existing solutions like Fitbit and Apple Watch, while ensuring data accuracy and adaptability to various workout types and environments.

### Directory Breakdown
The following directories contain the following:
<ul>
  <li>Arduino Model: files associates with the model's deployment onto an Arduino Nano 33 BLE Sense embedded device.</li>
  <li>Model Conversion: file used for converting the trained model onto ".tflite" format, as well as the ".tflite" model and the C-byte array model.</li>
  <li>Model Training: code for training the model, as well as reduced version of the extended datasets used to train the model. </li>
  <li>saved Model: contains the trained model details using the "savedModel" format.</li>
</ul>

</br>

## Español
Este proyecto de Tiny ML explora la pipeline de Tiny MLOps a través de el desarrollo de un modelo de regresión, simple pero operativo, en un dispositivo embebido. 
</br></br>
Este proyecto consiste en desarrollar un modelo de regresión para estimar el uso de calorías durante los entrenamientos, con el objetivo de proveer dispositivos accesible para comunidades de bajos recursos. El proyecto se profundiza en múltiples facetas de la pipeline Tiny MLOps, incluyendo: ingeniería de datos, entrenamiento de modelos y despliegue de modelos en un dispositivo embebido. Aprovechando los datos de CAPTURE-24, que ofrece datos diversificados de acelerómetros y anotaciones de actividades, el modelo será entrenado para proporcionar predicciones precisas de calorías. El progreso del dispositivo se evaluará comparando las tasas de participación en actividades con y sin el dispositivo, y la métrica de evaluación principal será el Error Cuadrático Medio. El objetivo es aumentar el compromiso con la actividad física, especialmente entre los jóvenes, creando una alternativa de bajo costo a soluciones existentes como Fitbit y Apple Watch, al mismo tiempo que se garantiza la precisión de los datos y la adaptabilidad a diversos tipos de entrenamientos y entornos.

### Detalles de Carpetas
Las siguientes carpetas contienen lo siguiente: 
<ul>
  <li>Arduino Model: archivos pertinentes al despliego del modelo en el dispositivo embebido Arduino Nano 33 BLE Sense.</li>
  <li>Model Conversion: archivos utilizados para convertir el modelo entrenado en el formato de “.tflite”, y tambien los modelos de “.tflite” y C-byte array.</li>
  <li>Model Training: código para entrenar al modelo y los archivos de datos (reducidos) para entrenar el modelo.</li>
  <li>saved Model: contiene los detalles del modelo entrenado usando el formato “savedModel”.</li>
</ul>
