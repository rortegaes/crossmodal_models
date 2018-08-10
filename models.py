from keras.layers import Embedding, Input, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, Concatenate, InputLayer, Reshape, BatchNormalization, Dropout
#from keras.models import Model, Sequential
from keras.metrics import categorical_accuracy
from keras.utils import to_categorical
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
import numpy as np
import json
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import glob
import os
from keras import optimizers
import sys
from scipy import misc
import shutil
import h5py
from PIL import Image
import time
import gc
import pickle

def generateTextualModel(num_class):
      model = Sequential()
      model.add(Embedding(len(word_index)+1, 300, embeddings_initializer="uniform", input_length=max_sequence_length, trainable=True))
      model.add(Conv1D(512, 5, activation="relu"))
      model.add(MaxPooling1D(5))
      model.add(Conv1D(512, 5, activation="relu"))
      model.add(MaxPooling1D(5))
      model.add(Conv1D(512, 5, activation="relu"))
      model.add(MaxPooling1D(35))
      model.add(Reshape((1,1,512)))
      model.add(Flatten())
      model.add(Dense(128, activation='relu'))
      model.add(Dense(num_class, activation='softmax'))
      return model
def generateVisualModel(num_class):
      model = Sequential()
      model.add(InputLayer(input_shape=(224,224,3)))
      model.add(Conv2D(64, (3,3), padding = "same", activation="relu"))
      model.add(BatchNormalization())
      model.add(Conv2D(64, (3,3), padding = "same", activation="relu"))
      model.add(BatchNormalization())
      model.add(MaxPooling2D(2))
      model.add(Conv2D(128, (3,3), padding = "same", activation="relu"))
      model.add(BatchNormalization())
      model.add(Conv2D(128, (3,3), padding = "same", activation="relu"))
      model.add(BatchNormalization())
      model.add(MaxPooling2D(2))
      model.add(Conv2D(256, (3,3), padding = "same", activation="relu"))
      model.add(BatchNormalization())
      model.add(Conv2D(256, (3,3), padding = "same", activation="relu"))
      model.add(BatchNormalization())
      model.add(MaxPooling2D(2))
      model.add(Conv2D(512, (3,3), padding = "same", activation="relu"))
      model.add(BatchNormalization())
      model.add(Conv2D(512, (3,3), padding = "same", activation="relu")) 
      model.add(BatchNormalization())
      model.add(MaxPooling2D((28,28),2))
      model.add(Flatten())
      model.add(Dense(128, activation='relu'))
      model.add(Dense(num_class, activation='softmax'))
      return model
def generateCrossModel(num_class):
      modelCaptions = Sequential()
      modelCaptions.add(Embedding(len(word_index) + 1,embedding_dimensions, weights = [embedding_matrix], input_length = max_sequence_length,trainable = False))
      modelCaptions.add(Conv1D(512, 5, activation="relu"))
      modelCaptions.add(MaxPooling1D(5))
      modelCaptions.add(Conv1D(512, 5, activation="relu"))
      modelCaptions.add(MaxPooling1D(5))
      modelCaptions.add(Conv1D(512, 5, activation="relu"))
      modelCaptions.add(MaxPooling1D(35))
      modelCaptions.add(Reshape((1,1,512)))
  
      modelImages = Sequential()
      modelImages.add(InputLayer(input_shape=(224,224,3)))
      modelImages.add(Conv2D(64, (3,3), padding = "same", activation="relu"))
      modelImages.add(Conv2D(64, (3,3), padding = "same", activation="relu"))
      modelImages.add(MaxPooling2D(2))
      modelImages.add(Conv2D(128, (3,3), padding = "same", activation="relu"))
      modelImages.add(Conv2D(128, (3,3), padding = "same", activation="relu"))
      modelImages.add(MaxPooling2D(2))
      modelImages.add(Conv2D(256, (3,3), padding = "same", activation="relu"))
      modelImages.add(Conv2D(256, (3,3), padding = "same", activation="relu"))
      modelImages.add(MaxPooling2D(2))
      modelImages.add(Conv2D(512, (3,3), padding = "same", activation="relu"))
      modelImages.add(Conv2D(512, (3,3), padding = "same", activation="relu"))      
      modelImages.add(MaxPooling2D((28,28),2))          
  
      mergedOut = Concatenate()([modelCaptions.output,modelImages.output])
      mergedOut = Flatten()(mergedOut)    
      mergedOut = Dense(128, activation='relu')(mergedOut)  
      mergedOut = Dense(2, activation='softmax')(mergedOut)
  
  
      newModel = Model([modelCaptions.input,modelImages.input], mergedOut)
      return newModel
