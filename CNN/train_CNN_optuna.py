import os
import shutil
import optuna
import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.backend import clear_session

from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


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
        self.sub_network = tf.keras.Sequential()
        
        kernel = (parameters['CNN_kernel_size'], parameters['CNN_kernel_size'])

        for i in range(parameters['n_CNN_layers_tot']):
            if i < parameters['n_CNN_layers_1']:
                n_filters = parameters['n_CNN_filters']
            else:
                n_filters = parameters['n_CNN_filters'] * 2
            
            if i == 0:
                self.sub_network.add(tf.keras.layers.Conv2D(n_filters, kernel, padding='same', activation='relu'))
            else:
                self.sub_network.add(tf.keras.layers.MaxPool2D((2, 2)))
                self.sub_network.add(tf.keras.layers.Conv2D(n_filters, kernel, padding='same', activation='relu'))

        self.sub_network.add(tf.keras.layers.Flatten())

        for _ in range(parameters['n_dense_layers']):
            self.sub_network.add(tf.keras.layers.Dense(parameters['dense_hidden_dim'], activation='relu'))

        self.sub_network.add(tf.keras.layers.Dense(1, activation='sigmoid'))

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


def objective(trial):
    try:
        # Clear clutter from previous tf.keras session graphs.
        clear_session()

        train_path = '../Sample/HVmodel/data/split_val/75x75/mix_sample_5.0_75x75.npy'
        val_path = '../Sample/HVmodel/data/split_val/75x75/mix_sample_5.0_val_75x75.npy'
        true_label_path = '../Sample/HVmodel/data/split_val/75x75/mix_sample_test_75x75.npy'
        model_name = 'SB_5.0_75x75'
        sample_type = 'Sensitivity: 5.0, Resolution: 75x75'

        # Training parameters
        BATCH_SIZE = trial.suggest_categorical('BATCH_SIZE', [256, 512, 1024, 2048])
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
        EPOCHS = 500
        patience = 10
        min_delta = 0.
        save_model_name = f'./CNN_models/last_model_CWoLa_hunting_{model_name}_optuna/'

        print(f'Read data from {train_path}')

        X_train, y_train = load_samples(train_path)
        X_val, y_val = load_samples(val_path)

        train_size = get_sample_size(y_train)
        val_size = get_sample_size(y_val)
        
        with tf.device('CPU'):
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            train_dataset = train_dataset.shuffle(buffer_size=len(y_train)).batch(BATCH_SIZE)
            del X_train, y_train

            valid_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
            valid_dataset = valid_dataset.batch(BATCH_SIZE)

        # Create the model
        n_CNN_layers_tot = trial.suggest_int('n_CNN_layers_tot', 2, 4)
        n_CNN_layers_1 = trial.suggest_int('n_CNN_layers_1', 1, n_CNN_layers_tot)
        n_CNN_filters = trial.suggest_categorical('n_CNN_filters', [16, 32, 64, 128, 256])
        CNN_kernel_size = trial.suggest_int('CNN_kernel_size', 2, 6)

        n_dense_layers = trial.suggest_int('n_dense_layers', 1, 5)
        dense_hidden_dim = trial.suggest_categorical('dense_hidden_dim', [16, 32, 64, 128, 256])
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
        best_model_name = f'./CNN_models/best_model_CWoLa_hunting_{model_name}_optuna/'
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
        file_name = 'CWoLa_Hunting_Hidden_Valley_training_results-Optuna.csv'
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

        return true_label_results[0]

    except Exception as e:
        print(f"An error occurred: {e}")
        trial.report(float('inf'), step=0)
        raise optuna.TrialPruned()




if __name__ == "__main__":
    study = optuna.create_study(
        direction='minimize',
        storage='sqlite:///db.sqlite3',  # Specify the storage URL here.
        study_name='Original SB 5.0 75x75',
        load_if_exists=True
    )
    study.optimize(objective, n_trials=100)

    print(f'Number of finished trials: {len(study.trials)}')

    print('Best trial:')
    trial = study.best_trial

    print(f'  Value: {trial.value}')

    print('  Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')