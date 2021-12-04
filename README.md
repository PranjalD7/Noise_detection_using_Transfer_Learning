# Noise_detection_using_Transfer_Learning
Noise detection using transfer learning. The model used for transfer  learning is YAMNet found on tensorflow(https://tfhub.dev/google/yamnet/1). The dataset used is the Microsoft Scalable Noisy Speech Dataset (MS-SNSD)( https://github.com/microsoft/MS-SNSD ) which consists of two classes clean and noise,This dataset contains a large collection of clean speech files and variety of environmental noise files in .wav format sampled at 16 kHz. We have also hosted the model on local host using flask.

Techs:Python 3.8.12,Tensorflow-2.7.0, tensorflow-hub 0.12.0, tensorflow-io 0.22.0, tensorflow-datasets 4.4.0,S oundFile 0.10.3.post1, Flas2.0.0, Flask-SQLAlchemy2.5.1, Jinja2 3.0.3

The file app.py contains the code for hosting the model on local host where the model has been deployed using flask. The files index.html contains the code for the home page of the 
web application where we can upload the sound files for classification, the file predict.html contains the code for the page which shows the prediction after classification.

The file noise_final_datapush contains the code for training the model using transfer learing on the Microsoft Scalable Noisy Speech Dataset, the weights are saved at the end
but the model can be retrained too.

set the paths before running the project
create an additional folder for storing the downloaded audios on host machine(app.py-  variable->path)
To run the web app-
1.run the app.py file
2.Enter URL http://localhost:5000/predict in your web browser
3. cilck on choose file to upload your .wav audio file and submit it.
4.Wait till the classification occurs and the result is displayed
