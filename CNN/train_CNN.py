#!/usr/bin/env python
# coding: utf-8
# python train_CNN.py <config_path.json>
# python train_CNN.py config_files/config_01.json

import os
import re
import sys
import json
import shutil
import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_info(path):
    # path: run path
    name = os.path.split(path)[1]

    with open(os.path.join(path, f'{name}_tag_1_banner.txt')) as f:
        for line in f.readlines():

            #  Integrated weight (pb)  :       0.020257
            match = re.match('#  Integrated weight \(pb\)  : +(\d+\.\d+)', line)
            if match:
                # unit: fb
                cross_section = float(match.group(1)) * 1000
            #  Number of Events        :       100000
            match = re.match('#  Number of Events        :       (\d+)', line)
            if match:
                # unit: fb
                nevent = int(match.group(1))

    return cross_section, nevent


def compute_nevent_in_SR_SB(sensitivity=1.0):
    results_s = np.load('../Sample/HVmodel/data/selection_results_SB_4400_5800_s.npy', allow_pickle=True).item()
    results_b = np.load('../Sample/HVmodel/data/selection_results_SB_4400_5800_b.npy', allow_pickle=True).item()

    # Total cross section and number of events
    xection, _ = get_info('../Sample/ppjj/Events/run_03')

    # cross section in signal region and sideband region
    cross_section_SR = results_b['cutflow_number']['Signal region'] / results_b['cutflow_number']['Total'] * xection
    cross_section_SB = results_b['cutflow_number']['Sideband region'] / results_b['cutflow_number']['Total'] * xection
    print(f'Background cross section, SR: {cross_section_SR:.2f} fb, SB: {cross_section_SB:.2f} fb')

    # number of background events in signal region and sideband region
    L = 139 * 1
    n_SR_B = cross_section_SR * L
    n_SB_B = cross_section_SB * L

    print(f'Background sample size: SR: {n_SR_B:.1f}, SB: {n_SB_B:.1f}')

    n_SR_S = sensitivity * np.sqrt(n_SR_B)
    n_SB_S = n_SR_S * results_s['cutflow_number']['Sideband region'] / results_s['cutflow_number']['Signal region']
    print(f'Signal sample size: SR: {n_SR_S:.1f}, SB: {n_SB_S:.1f}')

    return n_SR_S, n_SR_B, n_SB_S, n_SB_B


def create_mix_sample_from(npy_dirs: list, nevents: tuple, seed=0):
    # npy_dirs: list of npy directories
    # nevents: tuple of (n_sig_SR, n_sig_SB, n_bkg_SR, n_bkg_SB)
    data = None
    label = None

    data_sig_SR = np.load(os.path.join(npy_dirs[0], 'sig_in_SR-data.npy'))
    data_sig_SB = np.load(os.path.join(npy_dirs[0], 'sig_in_SB-data.npy'))
    data_bkg_SR = np.load(os.path.join(npy_dirs[0], 'bkg_in_SR-data.npy'))
    data_bkg_SB = np.load(os.path.join(npy_dirs[0], 'bkg_in_SB-data.npy'))

    n_sig_SR, n_sig_SB, n_bkg_SR, n_bkg_SB = nevents

    np.random.seed(seed)
    idx_sig_SR = np.random.choice(data_sig_SR.shape[0], n_sig_SR, replace=False)
    idx_sig_SB = np.random.choice(data_sig_SB.shape[0], n_sig_SB, replace=False)
    idx_bkg_SR = np.random.choice(data_bkg_SR.shape[0], n_bkg_SR, replace=False)
    idx_bkg_SB = np.random.choice(data_bkg_SB.shape[0], n_bkg_SB, replace=False)

    print(f'Preparing dataset from {npy_dirs}')
    for npy_dir in npy_dirs:

        data_sig_SR = np.load(os.path.join(npy_dir, 'sig_in_SR-data.npy'))
        data_sig_SB = np.load(os.path.join(npy_dir, 'sig_in_SB-data.npy'))
        data_bkg_SR = np.load(os.path.join(npy_dir, 'bkg_in_SR-data.npy'))
        data_bkg_SB = np.load(os.path.join(npy_dir, 'bkg_in_SB-data.npy'))

        new_data = np.concatenate([
            data_sig_SR[idx_sig_SR],
            data_bkg_SR[idx_bkg_SR],
            data_sig_SB[idx_sig_SB],
            data_bkg_SB[idx_bkg_SB]
        ], axis=0)

        if data is None:
            data = new_data
        else:
            data = np.concatenate([data, new_data], axis=0)

        new_label = np.zeros(sum(nevents))
        new_label[:n_sig_SR + n_bkg_SR] = 1

        if label is None:
            label = new_label
        else:
            label = np.concatenate([label, new_label])

    return data, label


def load_samples(path):
    root, _ = os.path.splitext(path)
    X = np.load(f'{root}-data.npy')
    Y = np.load(f'{root}-label.npy')
    return X, Y


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
    def __init__(self, parameters, name='CNN'):
        super(CNN, self).__init__(name=name)

        self.bn1 = tf.keras.layers.BatchNormalization()

        self.bn2 = tf.keras.layers.BatchNormalization()

        # create sub-network
        # self.sub_network = tf.keras.Sequential()

        # kernel = (parameters['CNN_kernel_size'], parameters['CNN_kernel_size'])

        # for i in range(parameters['n_CNN_layers_tot']):
        #     if i < parameters['n_CNN_layers_1']:
        #         n_filters = parameters['n_CNN_filters']
        #     else:
        #         n_filters = parameters['n_CNN_filters'] * 2

        #     if i == 0:
        #         self.sub_network.add(tf.keras.layers.Conv2D(n_filters, kernel, padding='same', activation='relu'))
        #     else:
        #         self.sub_network.add(tf.keras.layers.MaxPool2D((2, 2)))
        #         self.sub_network.add(tf.keras.layers.Conv2D(n_filters, kernel, padding='same', activation='relu'))

        # self.sub_network.add(tf.keras.layers.Flatten())

        # for _ in range(parameters['n_dense_layers']):
        #     self.sub_network.add(tf.keras.layers.Dense(parameters['dense_hidden_dim'], activation='relu'))

        # self.sub_network.add(tf.keras.layers.Dense(1, activation='sigmoid'))

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


def get_sensitivity_scale_factor(model_name, background_efficiencies, true_label_path):

    X_test, y_test = load_samples(true_label_path)

    loaded_model = tf.keras.models.load_model(model_name)

    # Compute False positive rate, True positive rate
    predictions = loaded_model.predict(X_test)

    labels = y_test
    y_prob = np.array(predictions)

    fpr, tpr, _ = roc_curve(labels == 1, y_prob)

    signal_efficiencies = []
    for bkg_eff in background_efficiencies:
        signal_efficiencies.append(get_tpr_from_fpr(bkg_eff, fpr, tpr))

    return np.array(signal_efficiencies) / np.array(background_efficiencies)**0.5


def main():
    config_path = sys.argv[1]

    # Read config file
    with open(config_path, 'r') as f:
        config = json.load(f)

    train_npy_paths = config['train_npy_paths']
    val_npy_paths = config['val_npy_paths']
    seed = config['seed']
    sensitivity = config['sensitivity']

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

    save_model_name = f'./CNN_models/last_model_CWoLa_hunting_{model_name}/'

    # Sampling dataset
    r_train, r_val = 0.8, 0.2
    n_SR_S, n_SR_B, n_SB_S, n_SB_B = compute_nevent_in_SR_SB(sensitivity=sensitivity)

    train_nevents = int(n_SR_S * r_train), int(n_SB_S * r_train), int(n_SR_B * r_train), int(n_SB_B * r_train)
    X_train, y_train = create_mix_sample_from(train_npy_paths, train_nevents, seed=seed)

    val_nevents = int(n_SR_S * r_val), int(n_SB_S * r_val), int(n_SR_B * r_val), int(n_SB_B * r_val)
    X_val, y_val = create_mix_sample_from(val_npy_paths, val_nevents, seed=seed)

    train_size = get_sample_size(y_train)
    val_size = get_sample_size(y_val)

    with tf.device('CPU'):
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=len(y_train)).batch(BATCH_SIZE)
        del X_train, y_train

        valid_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        valid_dataset = valid_dataset.batch(BATCH_SIZE)
        del X_val, y_val

    # Create the model
    n_CNN_layers_tot = params['n_CNN_layers_tot']
    n_CNN_layers_1 = params['n_CNN_layers_1']
    n_CNN_filters = params['n_CNN_filters']
    CNN_kernel_size = params['CNN_kernel_size']
    n_dense_layers = params['n_dense_layers']
    dense_hidden_dim = params['dense_hidden_dim']
    model_parameters = {
        'n_CNN_layers_tot': n_CNN_layers_tot,
        'n_CNN_layers_1': n_CNN_layers_1,
        'n_CNN_filters': n_CNN_filters,
        'CNN_kernel_size': CNN_kernel_size,
        'n_dense_layers': n_dense_layers,
        'dense_hidden_dim': dense_hidden_dim,
    }
    model = CNN(model_parameters)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=min_delta, verbose=1, patience=patience)
    check_point = tf.keras.callbacks.ModelCheckpoint(save_model_name, monitor='val_loss', verbose=1, save_best_only=True)

    history = model.fit(train_dataset, validation_data=valid_dataset, epochs=EPOCHS, callbacks=[early_stopping, check_point])

    # Training results
    best_model_name = f'./CNN_models/best_model_CWoLa_hunting_{model_name}/'
    if not os.path.isdir(best_model_name):
        shutil.copytree(save_model_name, best_model_name, dirs_exist_ok=True)
        print('Save to best model')
    best_model = tf.keras.models.load_model(best_model_name)
    best_results = best_model.evaluate(valid_dataset)
    print(f'Testing Loss = {best_results[0]:.3}, Testing Accuracy = {best_results[1]:.3}')

    loaded_model = tf.keras.models.load_model(save_model_name)
    results = loaded_model.evaluate(valid_dataset)
    print(f'Testing Loss = {results[0]:.3}, Testing Accuracy = {results[1]:.3}')

    if results[0] < best_results[0]:
        shutil.copytree(save_model_name, best_model_name, dirs_exist_ok=True)
        print('Save to best model')

    # Compute ACC & AUC
    y_pred = loaded_model.predict(X_val)
    ACC = get_highest_accuracy(y_val, y_pred)
    AUC = roc_auc_score(y_val, y_pred)

    # Testing results on true label sample
    X_test, y_test = load_samples(true_label_path)
    true_label_results = loaded_model.evaluate(x=X_test, y=y_test)
    print(f'True label: Testing Loss = {true_label_results[0]:.3}, Testing Accuracy = {true_label_results[1]:.3}')

    # Compute ACC & AUC
    y_pred = loaded_model.predict(X_test)
    true_label_ACC = get_highest_accuracy(y_test, y_pred)
    true_label_AUC = roc_auc_score(y_test, y_pred)

    background_efficiencies = [0.1, 0.01, 0.001]
    scale_factors = get_sensitivity_scale_factor(save_model_name, background_efficiencies, true_label_path)

    # Write results
    now = datetime.datetime.now()
    file_name = 'CWoLa_Hunting_Hidden_Valley_training_results-3.csv'
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
