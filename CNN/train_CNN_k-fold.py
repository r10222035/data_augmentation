#!/usr/bin/env python
# coding: utf-8
# python train_CNN.py <config_path.json>
# python train_CNN.py config_files/config_01.json

import os
import sys
import json
import shutil
import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import utils_CNN as utils

from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.model_selection import KFold

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_sample_size(y):
    if len(y.shape) == 1:
        ns = (y == 1).sum()
        nb = (y == 0).sum()
    else:
        ns = (y.argmax(axis=1) == 1).sum()
        nb = (y.argmax(axis=1) == 0).sum()
    print(ns, nb)
    return ns, nb


class CNN(tf.keras.Model):
    def __init__(self, name='CNN'):
        super(CNN, self).__init__(name=name)

        self.bn1 = tf.keras.layers.BatchNormalization()

        self.bn2 = tf.keras.layers.BatchNormalization()

        self.sub_network = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (5, 5), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Conv2D(64, (5, 5), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
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
        channel1 = self.bn1(channel1)
        channel2 = self.bn2(channel2)
        output_channel1 = self.sub_network(channel1)
        output_channel2 = self.sub_network(channel2)

        # multiply the output
        output = tf.keras.layers.Multiply()([output_channel1, output_channel2])

        return output


def get_highest_accuracy(y_true, y_pred):
    _, _, thresholds = roc_curve(y_true, y_pred)
    # compute highest accuracy
    thresholds = np.array(thresholds)
    if len(thresholds) > 1000:
        thresholds = np.percentile(thresholds, np.linspace(0, 100, 1001))
    accuracy_scores = []
    for threshold in thresholds:
        accuracy_scores.append(accuracy_score(y_true, y_pred > threshold))

    accuracies = np.array(accuracy_scores)
    return accuracies.max()


def get_tpr_from_fpr(passing_rate, fpr, tpr):
    n_th = (fpr < passing_rate).sum()
    return tpr[n_th]


def get_sensitivity_scale_factor(y_test, y_prob, background_efficiencies=[0.1, 0.01, 0.001]):
    fpr, tpr, _ = roc_curve(y_test == 1, y_prob)

    signal_efficiencies = []
    for bkg_eff in background_efficiencies:
        signal_efficiencies.append(get_tpr_from_fpr(bkg_eff, fpr, tpr))

    return np.array(signal_efficiencies) / np.array(background_efficiencies)**0.5


def k_fold_model_evaluate(model_path, k, X, y):
        Losses, ACCs = [], []
        for i in range(1, k+1):
            model = tf.keras.models.load_model(f'{model_path}_fold_{i}')
            results = model.evaluate(x=X, y=y, batch_size=512)
            Losses.append(results[0])
            ACCs.append(results[1])
        return np.array(Losses).mean(), np.array(ACCs).mean()

def k_fold_model_predict(model_path, k, X):
    y_preds = []
    for i in range(1, k+1):
        model = tf.keras.models.load_model(f'{model_path}_fold_{i}')
        y_pred = model.predict(X, batch_size=512)
        y_preds.append(y_pred)
    return np.array(y_preds).mean(axis=0)

def main():
    config_path = sys.argv[1]

    # Read config file
    with open(config_path, 'r') as f:
        config = json.load(f)

    train_npy_paths = config['train_npy_paths']
    # val_npy_paths = config['val_npy_paths']
    seed = config['seed']
    sensitivity = config['sensitivity']
    luminosity = config['luminosity']

    true_label_path = config['true_label_path']
    model_name = config['model_name']
    sample_type = config['sample_type']

    # Training parameters
    # 讀取參數設定
    with open('params.json', 'r') as f:
        params = json.load(f)

    # 從參數設定中獲取變數
    BATCH_SIZE = params['BATCH_SIZE']
    EPOCHS = params['EPOCHS']
    patience = params['patience']
    min_delta = params['min_delta']
    learning_rate = params['learning_rate']

    save_model_path = f'CNN_models/last_model_CWoLa_hunting_{model_name}'

    # Sampling dataset
    # r_train, r_val = 0.8, 0.2
    n_SR_S, n_SR_B, n_SB_S, n_SB_B = utils.compute_nevent_in_SR_SB(sensitivity=sensitivity, L=luminosity)

    nevents = np.array([n_SR_S, n_SB_S, n_SR_B, n_SB_B]).astype(int)
    X, y = utils.create_mix_sample_from_npy(train_npy_paths, nevents, seed=seed)

    # Define k-fold number
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        print(f'Fold {fold+1}')
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        train_size = get_sample_size(y_train)
        val_size = get_sample_size(y_val)

        with tf.device('CPU'):
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            train_dataset = train_dataset.shuffle(buffer_size=len(y_train)).batch(BATCH_SIZE)
            del X_train, y_train
            valid_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
            valid_dataset = valid_dataset.batch(BATCH_SIZE)
            del X_val, y_val

        save_model_name = f'{save_model_path}_fold_{fold+1}/'

        # Create the model  
        model = CNN()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                      metrics=['accuracy'])

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=min_delta, verbose=1, patience=patience)
        check_point = tf.keras.callbacks.ModelCheckpoint(save_model_name, monitor='val_loss', verbose=1, save_best_only=True)

        history = model.fit(train_dataset, validation_data=valid_dataset, epochs=EPOCHS, callbacks=[early_stopping, check_point])


    # Training results
    best_model_path = f'./CNN_models/best_model_CWoLa_hunting_{model_name}'
    if not os.path.isdir(f'{best_model_path}_fold_{k}'):
        for i in range(1, k+1):
            shutil.copytree(f'{save_model_path}_fold_{i}', f'{best_model_path}_fold_{i}', dirs_exist_ok=True)
        print('Save to best model')
    best_results = k_fold_model_evaluate(best_model_path, k, X, y)
    print(f'Testing Loss = {best_results[0]:.3}, Testing Accuracy = {best_results[1]:.3}')

    results = k_fold_model_evaluate(save_model_path, k, X, y)
    print(f'Testing Loss = {results[0]:.3}, Testing Accuracy = {results[1]:.3}')

    if results[0] < best_results[0]:
        for i in range(1, k+1):
            shutil.copytree(f'{save_model_path}_fold_{i}', f'{best_model_path}_fold_{i}', dirs_exist_ok=True)
        print('Save to best model')

    # Compute ACC & AUC
    y_pred = k_fold_model_predict(save_model_path, k, X)
    ACC = get_highest_accuracy(y, y_pred)
    AUC = roc_auc_score(y, y_pred)

    # Testing results on true label sample
    X_test, y_test = utils.load_samples(true_label_path)
    true_label_results = k_fold_model_evaluate(save_model_path, k, X_test, y_test)
    print(f'True label: Testing Loss = {true_label_results[0]:.3}, Testing Accuracy = {true_label_results[1]:.3}')

    # Compute ACC & AUC
    y_pred = k_fold_model_predict(save_model_path, k, X_test)
    true_label_ACC = get_highest_accuracy(y_test, y_pred)
    true_label_AUC = roc_auc_score(y_test, y_pred)

    background_efficiencies = np.array([0.1, 0.01, 0.001])
    scale_factors = get_sensitivity_scale_factor(y_test, y_pred, background_efficiencies)

    # Background subtraction
    X_test_B = X_test[y_test == 0]
    y_test_B = y_test[y_test == 0]
    origin_npy_paths = ['../Sample/HVmodel/data/origin/25x25']
    X_train_SR, y_train_SR, _, _ = utils.get_SR_SB_sample_from_npy(origin_npy_paths, nevents, seed=seed)

    y_prob_test = k_fold_model_predict(save_model_path, k, X_test_B)
    y_prob_train = k_fold_model_predict(save_model_path, k, X_train_SR)

    fpr, thresholds = utils.get_fpr_thresholds(y_test_B, y_prob_test)

    train_SR_S_pass, train_SR_B_pass, test_SR_B_pass = [], [], []
    for eff_test in background_efficiencies:
        threshold = utils.get_threshold_from_fpr(fpr, thresholds, eff_test)

        y_prob_train_S = y_prob_train[y_train_SR == 1]
        y_prob_train_B = y_prob_train[y_train_SR == 0]

        sig_pass = (y_prob_train_S > threshold).sum() / len(y_prob_train_S) * n_SR_S if n_SR_S > 0 else 0
        bkg_pass = (y_prob_train_B > threshold).sum() / len(y_prob_train_B) * n_SR_B if n_SR_B > 0 else 0
        train_SR_S_pass.append(sig_pass)
        train_SR_B_pass.append(bkg_pass)

        test_bkg_pass = (y_prob_test > threshold).sum() / len(y_prob_test) * n_SR_B
        test_SR_B_pass.append(test_bkg_pass)

    # Write results
    now = datetime.datetime.now()
    file_name = 'CWoLa_Hunting_Hidden_Valley_training_results-4.csv'
    data_dict = {
                'Train signal size': [train_size[0]],
                'Train background size': [train_size[1]],
                'Validation signal size': [val_size[0]],
                'Validation background size': [val_size[1]],
                'Loss': [results[0]],
                'ACC': [ACC],
                'AUC': [AUC],
                'Loss-true': [true_label_results[0]],
                'ACC-true': [true_label_ACC],
                'AUC-true': [true_label_AUC],
                'Sample Type': [sample_type],
                'Model Name': [model_name],
                'TPR/FPR^0.5: FPR=0.1': [scale_factors[0]],
                'TPR/FPR^0.5: FPR=0.01': [scale_factors[1]],
                'TPR/FPR^0.5: FPR=0.001': [scale_factors[2]],
                'Train SR signal pass: Test FPR=0.1': [train_SR_S_pass[0]],
                'Train SR signal pass: Test FPR=0.01': [train_SR_S_pass[1]],
                'Train SR signal pass: Test FPR=0.001': [train_SR_S_pass[2]],
                'Train SR background pass: Test FPR=0.1': [train_SR_B_pass[0]],
                'Train SR background pass: Test FPR=0.01': [train_SR_B_pass[1]],
                'Train SR background pass: Test FPR=0.001': [train_SR_B_pass[2]],
                'Test SR background pass: Test FPR=0.1': [test_SR_B_pass[0]],
                'Test SR background pass: Test FPR=0.01': [test_SR_B_pass[1]],
                'Test SR background pass: Test FPR=0.001': [test_SR_B_pass[2]],
                'Training epochs': [len(history.history['loss']) + 1],
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
