#!/usr/bin/env python
# coding: utf-8
# python train_best_hp_DNN.py <pairing_method>
# python test_DNN.py 0

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

def main():

    CWoLa_index = int(sys.argv[1])

    f1 = round(CWoLa_index * 0.1,2)
    f2 = round(1 - f1, 2)
    
    test_path = f'./Sample/DNN/SPANet_high_level_test.npy'
    X_test, y_test = load_samples(test_path)

    # Training results
    best_model_name = f'./DNN_models/DNN_best_model_CWoLa_{CWoLa_index:02}/'
    best_model = tf.keras.models.load_model(best_model_name)
    best_results = best_model.evaluate(x=X_test, y=y_test)

    if best_results[1] < 0.5:
        y_test = y_test[:,[1,0]]
        best_results = best_model.evaluate(x=X_test, y=y_test)
    print(f'Testing Loss = {best_results[0]:.3}, Testing Accuracy = {best_results[1]:.3}')

    # # Compute AUC
    labels = y_test
    predictions = best_model.predict(X_test)

    y_test = np.argmax(labels, axis=1)
    y_prob = np.array(predictions)

    fig, ax = plt.subplots(1, 1, figsize=(6,5))

    i=1
    AUC = roc_auc_score(y_test==i,  y_prob[:,i])
    # label = 1
    # AUC = roc_auc_score(y_test,  y_prob[:,label])

    # if AUC < 0.5:
    #     label = 0
    #     AUC = roc_auc_score(y_test,  y_prob[:,label])
    # fpr, tpr, thresholds = roc_curve(y_test, y_prob[:,label])

    # accuracy_scores = []
    # # 最多用 1000 個
    # if len(thresholds) > 1000:
    #     thresholds = np.percentile(thresholds, np.linspace(0,100,1001))

    # for threshold in tqdm(thresholds):
    #     accuracy_scores.append(accuracy_score(y_test,  y_prob[:,label]>threshold))

    # accuracies = np.array(accuracy_scores)
    # ACC = accuracies.max() 

    # Plot ROC
    # labels = y_test
    # predictions = loaded_model.predict(X_test)

    # y_test = np.argmax(labels, axis=1)
    # y_prob = np.array(predictions)

    # fig, ax = plt.subplots(1, 1, figsize=(6,5))

    # i=1
    # AUC = roc_auc_score(y_test==i,  y_prob[:,i])
    # fpr, tpr, thresholds = roc_curve(y_test==i, y_prob[:,i])
    # ax.plot(fpr, tpr, label = f'AUC = {AUC:.3f}')

    # ax.set_title(f'ROC curve of DNN')
    # ax.set_xlabel('False Positive Rate')
    # ax.set_ylabel('True Positive Rate')
    # ax.legend()

    # plt.savefig(f'figures/ROC_DNN_{pairing_method}_high_level.png', facecolor='White', dpi=300, bbox_inches='tight')
   
    # Write results
    now = datetime.datetime.now()
    file_name = 'DNN_CWoLa_training_results.csv'
    data_dict = {
                'Fraction 1': [f1],
                'Fraction 2': [f2],
                'ACC': [best_results[1]],
                'AUC': [AUC],
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