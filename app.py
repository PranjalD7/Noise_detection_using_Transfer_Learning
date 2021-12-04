#importing libraries 
from flask import Flask,render_template, jsonify, request
import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import io
import os
import soundfile as sf
import glob
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop,Adam
import tensorflow_io as tfio
import tensorflow_datasets as tfds
import re
import tensorflow as tf
import wave

#creating flask reference to this file
app = Flask(__name__)
my_classes = ['Noise', 'Clean']
map_class_to_id = {'Noise':0, 'Clean':1}


#creating page to upload file
@app.route('/')
def index():
    return render_template('index.html')
#creating page to output prediction 
@app.route('/predict', methods=['POST','GET'])
def predict():
   #requesting file from index.html and saving it on host machine
   file = request.files['file']
   path='./download_music_bot\temp.wav'
   file.save(path)
   

   #reloading saved model
   reloaded_model = tf.saved_model.load('./noise_bot_model')
   
   # Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio
   def load_wav_16k_mono():
    
    file_contents = tf.io.read_file('./download_music_bot\temp.wav')
    wav, sample_rate = tf.audio.decode_wav(
          file_contents,
          desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav
   serving_results = reloaded_model.signatures['serving_default'](load_wav_16k_mono())
   noise_or_clean = my_classes[tf.argmax(serving_results['classifier'])]
   return render_template('predict.html', prediction =noise_or_clean)
#hosting on localhost
if __name__ == '__main__':
    app.run(debug=True) 

