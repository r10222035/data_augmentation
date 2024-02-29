#!/usr/bin/env python
# coding: utf-8
# python train_CNN.py <train_file> <model_name> "<sample_type>"
# python train_CNN.py ../Sample/HVmodel/data/mix_sample_1.0.npy SB_1.0 "Sensitivity: 1.0"

import os
import sys
import shutil
import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def load_samples(path, change_order=False):

    root, _ = os.path.splitext(path)

    if change_order:
        X = np.load(f'{root}-data.npy').transpose(0, 2, 3, 1)
    else:
        X = np.load(f'{root}-data.npy')
    # Y = np.eye(2)[np.load(f'{root}-label.npy')]
    Y = np.load(f'{root}-label.npy')

    return X, Y


def get_sample_size(y):
    if len(y.shape) == 1:
        ns = (y==1).sum()
        nb = (y==0).sum()
    else:
        ns = (y.argmax(axis=1)==1).sum()
        nb = (y.argmax(axis=1)==0).sum()
    print(ns, nb)
    return ns, nb


# class CNN(tf.keras.Model):
#     def __init__(self, name='CNN', dim_image=(75, 75, 2), n_class=2):
#         super(CNN, self).__init__(name=name)

#         self.ptqk = tf.keras.Sequential([
#             tf.keras.layers.BatchNormalization(),
#             tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu'),
#             tf.keras.layers.MaxPool2D((2,2)),
#             tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu'),
#             tf.keras.layers.MaxPool2D((2,2)),
#             tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
#             tf.keras.layers.MaxPool2D((2,2)),
#             tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
#             tf.keras.layers.Flatten(),
#             tf.keras.layers.Dense(64, activation='relu'),
#             tf.keras.layers.Dense(64, activation='relu'),
#             tf.keras.layers.Dense(64, activation='relu'),
#         ])

#         """Output Layer"""
#         self._output = tf.keras.layers.Dense(n_class, activation='softmax')

#     @tf.function
#     def call(self, inputs, training=False):
#         latent_ptqk = self.ptqk(inputs)

#         return self._output(latent_ptqk)


class CNN(tf.keras.Model):
    def __init__(self, name='CNN'):
        super(CNN, self).__init__(name=name)

        self.sub_network = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (5,5), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(64, (5,5), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ])

    @tf.function
    def call(self, inputs, training=False):
        # split two channels
        channel1, channel2 = tf.split(inputs, num_or_size_splits=2, axis=-1)

        # pass through the same CNN
        output_channel1 = self.sub_network(channel1)
        output_channel2 = self.sub_network(channel2)

        # multiply the output
        output = tf.multiply(output_channel1, output_channel2)

        return output


def get_tpr_from_fpr(passing_rate, fpr, tpr):
    n_th = (fpr < passing_rate).sum()
    return tpr[n_th]


def get_sensitivity_scale_factor(model_name, background_efficiencies):
    true_label_path = '../Sample/HVmodel/data/new/mix_sample_testing.npy'
    X_test, y_test = load_samples(true_label_path)

    loaded_model = tf.keras.models.load_model(model_name)
    # true_label_results = loaded_model.evaluate(x=X_test, y=y_test)

    # if true_label_results[1] < 0.5:
    #     y_test = y_test[:,[1,0]]

    # Compute False positive rate, True positive rate
    predictions = loaded_model.predict(X_test)

    # labels = np.argmax(y_test, axis=1)
    labels = y_test
    y_prob = np.array(predictions)

    # fpr, tpr, _ = roc_curve(labels==1, y_prob[:,1])
    fpr, tpr, _ = roc_curve(labels==1, y_prob)

    signal_efficiencies = []
    for bkg_eff in background_efficiencies:
        signal_efficiencies.append(get_tpr_from_fpr(bkg_eff, fpr, tpr))

    return np.array(signal_efficiencies) / np.array(background_efficiencies)**0.5


def main():
    data_path = sys.argv[1]
    model_name = sys.argv[2]
    sample_type = sys.argv[3]

    print(f'Read data from {data_path}')

    X, y = load_samples(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=17)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, shuffle=False)

    train_size = get_sample_size(y_train)
    test_size = get_sample_size(y_test)

    # Training parameters
    batch_size = 512
    train_epochs = 500
    patience = 30
    min_delta = 0.
    learning_rate = 1e-4
    save_model_name = f'./CNN_models/CNN_last_model_CWoLa_hunting_{model_name}/'

    # Create the model
    model = CNN()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  # loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=min_delta, verbose=1, patience=patience)
    check_point = tf.keras.callbacks.ModelCheckpoint(save_model_name, monitor='val_loss', verbose=1, save_best_only=True)

    history = model.fit(x=X_train, y=y_train, validation_split=0.2, epochs=train_epochs, batch_size=batch_size, callbacks=[early_stopping, check_point])
    # history = model.fit(x=X_train, y=y_train, validation_split=0.2, epochs=train_epochs, batch_size=batch_size, shuffle=False, callbacks=[early_stopping, check_point])

    # Training results
    best_model_name = f'./CNN_models/CNN_best_model_CWoLa_hunting_{model_name}/'
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
    # labels = y_test
    predictions = loaded_model.predict(X_test)

    # y_test = np.argmax(labels, axis=1)
    y_prob = np.array(predictions)

    i = 1
    # AUC = roc_auc_score(y_test==i,  y_prob[:,i])
    AUC = roc_auc_score(y_test==i,  y_prob)

    # Testing results on true label sample
    true_label_path = '../Sample/HVmodel/data/new/mix_sample_testing.npy'
    X_test, y_test = load_samples(true_label_path)
    true_label_results = loaded_model.evaluate(x=X_test, y=y_test)

    # if true_label_results[1] < 0.5:
    #     y_test = y_test[:,[1,0]]
    #     true_label_results = loaded_model.evaluate(x=X_test, y=y_test)
    print(f'True label: Testing Loss = {true_label_results[0]:.3}, Testing Accuracy = {true_label_results[1]:.3}')

    # Compute AUC
    # labels = y_test
    predictions = loaded_model.predict(X_test)

    # y_test = np.argmax(labels, axis=1)
    y_prob = np.array(predictions)

    i = 1
    # true_label_AUC = roc_auc_score(y_test==i,  y_prob[:,i])
    true_label_AUC = roc_auc_score(y_test==i,  y_prob)

    background_efficiencies = [0.1, 0.01, 0.001]
    scale_factors = get_sensitivity_scale_factor(save_model_name, background_efficiencies)

    # Write results
    now = datetime.datetime.now()
    file_name = 'CWoLa_Hunting_Hidden_Valley_training_results.csv'
    data_dict = {
                'Train signal size': [train_size[0]],
                'Train background size': [train_size[1]],
                'Loss': [results[0]],
                'ACC': [results[1]],
                'AUC': [AUC],
                'Loss-true': [true_label_results[0]],
                'ACC-true': [true_label_results[1]],
                'AUC-true': [true_label_AUC],
                'Sample Type': [sample_type],
                'Model Name': [model_name],
                'TPR/FPR^0.5: FPR=0.1': [scale_factors[0]],
                'TPR/FPR^0.5: FPR=0.01': [scale_factors[1]],
                'TPR/FPR^0.5: FPR=0.001': [scale_factors[2]],
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
