#!/usr/bin/env python
# coding: utf-8
# python train_DNN.py <pairing_method>
# python train_DNN.py 1

import os
import sys
import shutil
import datetime

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

os.environ['CUDA_VISIBLE_DEVICES']='1'

def load_samples(path):

    npy_dict = np.load(path, allow_pickle=True).item()

    X = npy_dict['data']
    Y = np.eye(2)[npy_dict['label']]
    
    return X, Y

def get_sample_size(y):
    ns = (y.argmax(axis=1)==1).sum()
    nb = (y.argmax(axis=1)==0).sum()
    print(ns, nb)
    return ns, nb

# Training sample

CWoLa_index = int(sys.argv[1])

data_path = f'./Sample/DNN/mixing_sample_{CWoLa_index:02}.npy'
X, y = load_samples(data_path)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=17)

train_size = get_sample_size(y_train)
test_size = get_sample_size(y_test)

train_epochs = 500
patience = 10
min_delta = 0.
learning_rate = 0.000228
batch_size = 512
save_model_name = f'./DNN_models/DNN_last_model_CWoLa_{CWoLa_index:02}/'

# 建立 DNN
n_layers = 5
num_hidden = 256

model = Sequential()
for i in range(n_layers):
    model.add(Dense(num_hidden, activation='relu'))
model.add(Dense(units=2, activation='softmax'))

#  print(model.summary())

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=min_delta, verbose=1, patience=patience)
check_point    = tf.keras.callbacks.ModelCheckpoint(save_model_name, monitor='val_loss', verbose=1, save_best_only=True)

history = model.fit(x=X_train, y=y_train, validation_split=0.2, epochs=train_epochs, batch_size=batch_size, callbacks=[early_stopping, check_point])

# Training results
best_model_name = f'./DNN_models/DNN_best_model_CWoLa_{CWoLa_index:02}/'
if not os.path.isdir(best_model_name):
    shutil.copytree(save_model_name, best_model_name, dirs_exist_ok=True)
    print('Save to best model')
best_model = tf.keras.models.load_model(best_model_name)
best_results = best_model.evaluate(x=X_test, y=y_test)
print(f'Testing Loss = {best_results[0]:.3}, Testing Accuracy = {best_results[1]:.3}')

loaded_model = tf.keras.models.load_model(save_model_name)
results = loaded_model.evaluate(x=X_test, y=y_test)
print(f'Testing Loss = {results[0]:.3}, Testing Accuracy = {results[1]:.3}')

if results[1] > best_results[1]:
    shutil.copytree(save_model_name, best_model_name, dirs_exist_ok=True)
    print('Save to best model')