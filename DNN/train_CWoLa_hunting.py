#!/usr/bin/env python
# coding: utf-8
# python train_CWoLa_hunting.py <train_file> <true_file> <model_name> <sample_type>
# python train_CWoLa_hunting.py ../Sample/DNN/CWoLa_hunting_10times_signal-500GeV.npy ../Sample/DNN/min_dR_500GeV_test.npy 10times_sig_no_higgs_pt_m_500GeV "500 GeV: 10 times signal, no Higgs pT, m"

import os
from random import sample
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

def load_samples(path, delete_col=None):
    npy_dict = np.load(path, allow_pickle=True).item()
    
    X = npy_dict['data']
    if delete_col != None:
        X = np.delete(X, delete_col, axis=1)
    Y = np.eye(2)[npy_dict['label']]
    return X, Y

def get_sample_size(y):
    ns = (y.argmax(axis=1)==1).sum()
    nb = (y.argmax(axis=1)==0).sum()
    print(ns, nb)
    return ns, nb

def main():

    # Training sample
    data_path = sys.argv[1]
    true_label_path = sys.argv[2]
    model_name = sys.argv[3]
    sample_type = sys.argv[4]
    
    d_col = (0,3,4,7)
    X, y = load_samples(data_path, d_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=17)

    train_size = get_sample_size(y_train)
    test_size = get_sample_size(y_test)

    train_epochs = 500
    patience = 10
    min_delta = 0.
    learning_rate = 0.0005
    batch_size = 256
    save_model_name = f'./DNN_models/DNN_last_model_CWoLa_hunting_{model_name}/'

    # 建立 DNN
    n_layers = 4
    num_hidden = 64

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
    best_model_name = f'./DNN_models/DNN_best_model_CWoLa_hunting_{model_name}/'
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


    # Compute AUC
    labels = y_test
    predictions = loaded_model.predict(X_test)

    y_test = np.argmax(labels, axis=1)
    y_prob = np.array(predictions)

    i = 1
    AUC = roc_auc_score(y_test==i,  y_prob[:,i])


    # Testing results on true label sample
    X_test, y_test = load_samples(true_label_path, delete_col=(0,3,4,7,10,11,12,13,15))
    true_label_results = loaded_model.evaluate(x=X_test, y=y_test)

    if true_label_results[1] < 0.5:
        y_test = y_test[:,[1,0]]
        true_label_results = loaded_model.evaluate(x=X_test, y=y_test)
    print(f'True label: Testing Loss = {true_label_results[0]:.3}, Testing Accuracy = {true_label_results[1]:.3}')

    # Compute AUC
    labels = y_test
    predictions = loaded_model.predict(X_test)

    y_test = np.argmax(labels, axis=1)
    y_prob = np.array(predictions)

    i = 1
    true_label_AUC = roc_auc_score(y_test==i,  y_prob[:,i])

    # Write results
    now = datetime.datetime.now()
    file_name = 'CWoLa_Hunting_training_results.csv'
    data_dict = {
                'Train signal size': [train_size[0]],
                'Train background size': [train_size[1]],
                'ACC': [results[1]],
                'AUC': [AUC],
                'ACC-true': [true_label_results[1]],
                'AUC-true': [true_label_AUC],
                'Sample Type': [sample_type],
                'Model Name': [model_name],
                'time': [now],
                }
    
    df = pd.DataFrame(data_dict)
    if os.path.isfile(file_name):
        training_results_df = pd.read_csv(file_name)
        pd.concat([training_results_df, df], ignore_index=True).to_csv(file_name, index=False)
    else:
        df.to_csv(file_name, index=False)

if __name__ == '__main__':
    main()