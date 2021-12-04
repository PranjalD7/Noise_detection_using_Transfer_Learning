#importing libraries
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import io
import os
import soundfile as sf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop,Adam
import tensorflow_io as tfio
import tensorflow_datasets as tfds
import glob # Package for Unix-style pathname pattern expansion

#creating classes
my_classes = ['Noise', 'Clean']
map_class_to_id = {'Noise':0, 'Clean':1}

#setting paths to dataset
directory='./MS-SNSD-master\MS-SNSD-master\noise_train'
dir2='./MS-SNSD-master\MS-SNSD-master\noise_test'
dir3='./train\clean_train'
dir4='./test\clean_test'

#storing all filenames fron different directories and joining .wav at the end
all_files =glob.glob(os.path.join(directory,"*.wav"))

all_files_clean=glob.glob(os.path.join(dir3,"*.wav"))

all_files_val=glob.glob(os.path.join(dir2,"*.wav"))

all_files_val2=glob.glob(os.path.join(dir4,"*.wav"))

#setting class variable array
target=np.zeros(len(all_files))

target_clean=np.ones(len(all_files_clean))

target_val=np.zeros(len(all_files_val))

target_val2=np.ones(len(all_files_val2))

#  Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. 
def load_wav_16k_mono(filename):
  
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
          file_contents,
          desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

#using function to pass only filename to  load_wav_16k_mono from the tensorflow dataset
def load_wav_for_map(filename, label):
  return load_wav_16k_mono(filename),label

#creating tensorflow dataset containing file names and target variable for noise files
main_ds = tf.data.Dataset.from_tensor_slices((all_files,target))

#loading files to load_wav_for_map to convert them to float tensor
main_ds = main_ds.map(load_wav_for_map)

#creating tensorflow dataset containing file names and target variable for clean files
main_ds2 = tf.data.Dataset.from_tensor_slices((all_files_clean,target_clean))

#loading files to load_wav_for_map to convert them to float tensor
main_ds2 = main_ds2.map(load_wav_for_map)

#joining the two datasets after conversion
final_ds=main_ds.concatenate(main_ds2)

#creating validation dataset and performing same procedure as above
val_ds=tf.data.Dataset.from_tensor_slices((all_files_val,target_val))
val_ds = val_ds.map(load_wav_for_map)

val_ds2=tf.data.Dataset.from_tensor_slices((all_files_val2,target_val2))
val_ds2= val_ds2.map(load_wav_for_map)

final_val_ds=val_ds.concatenate(val_ds2)

#importing yamnet
yamnet_model_handle ='https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

#running yamnet
def extract_embedding(wav_data, label):
  ''' run YAMNet to extract embedding from the wav data '''
  scores, embeddings, spectrogram = yamnet_model(wav_data)
  num_embeddings = tf.shape(embeddings)[0]
  return (embeddings,
            tf.repeat(label, num_embeddings)) 


final_ds = final_ds.map(extract_embedding).unbatch()
final_val_ds = final_val_ds.map(extract_embedding).unbatch()

final_ds = final_ds.cache().shuffle(200).batch(32).prefetch(tf.data.AUTOTUNE)
final_val_ds = final_val_ds.cache().shuffle(100).batch(32).prefetch(tf.data.AUTOTUNE)


#using the embeddings from yamnet and adding them to our sequential model with an input layer which accepts embeddings and 2 dense layers
my_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1024), dtype=tf.float32, name='input_embedding'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(len(my_classes))          
])

my_model.summary()
my_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 optimizer="adam",
                 metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                            patience=3,
                                            restore_best_weights=True)

history = my_model.fit(final_ds,
                       epochs=20,
                      validation_data=final_val_ds,
                      callbacks=callback)

#this code snippet is used for easy deployment of this model
class ReduceMeanLayer(tf.keras.layers.Layer):
  def __init__(self, axis=0, **kwargs):
    super(ReduceMeanLayer, self).__init__(**kwargs)
    self.axis = axis

  def call(self, input):
    return tf.math.reduce_mean(input, axis=self.axis)

saved_model_path = './noise_bot_model'

input_segment = tf.keras.layers.Input(shape=(), dtype=tf.float32, name='audio')
embedding_extraction_layer = hub.KerasLayer(yamnet_model_handle,
                                            trainable=False, name='yamnet')
_, embeddings_output, _ = embedding_extraction_layer(input_segment)
serving_outputs = my_model(embeddings_output)
serving_outputs = ReduceMeanLayer(axis=0, name='classifier')(serving_outputs)
serving_model = tf.keras.Model(input_segment, serving_outputs)
serving_model.save(saved_model_path, include_optimizer=False)

