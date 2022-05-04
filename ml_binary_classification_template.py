#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standard template to train and evaluate simple machine learning models.

This examples presents a simple workflow to fit a binary classifier trained to
identify rain in audio samples.

"""

import numpy as np
import pandas as pd
import os
from maad import sound
from librosa import feature
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

#%% Set variables
path_annotations = 'audio_labels.csv'  # manual annotations in csv table
path_audio = 'audio/'  # directory where the audio data is located
target_fs = 10000  # set target sampling rate for audio

#%% Load annotations
df = pd.read_csv(path_annotations)  

#%% Compute features
df_features = pd.DataFrame()

for idx_row, row in df.iterrows():
    full_path_audio = os.path.join(path_audio, row.sample_idx)
    s, fs = sound.load(full_path_audio)
    # resample
    s_resamp = sound.resample(s, fs, target_fs, res_type='kaiser_fast')
    # transform
    mfcc = feature.mfcc(y=s_resamp, sr=target_fs, n_mfcc=20, n_fft=1024, 
                        win_length=1024, hop_length=512, htk=True)
    mfcc = np.median(mfcc, axis=1)
    # format dataframe
    idx_names = ['mfcc_' + str(idx).zfill(2) for idx in range(1,mfcc.size+1)]
    row = row.append(pd.Series(mfcc, index=idx_names))
    row.name = idx_row
    df_features = df_features.append(row)

#%% Split development and test data
X = df_features.loc[:,df_features.columns.str.startswith('mfcc')]
y = (df_features.label=='LLUVIA').astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    shuffle=True,
                                                    random_state=42)


#%% Tune model hyperparameters
clf = RandomForestClassifier(n_jobs=-1, class_weight='balanced_subsample')


# Set tuning strategy
param_grid = {'n_estimators' : [1, 5, 10, 100, 300, 500],
              'max_features' : [2, 6, 10, 14, 18]}

skf = StratifiedKFold(n_splits=10)
clf_gs = GridSearchCV(clf, param_grid, scoring=['f1'], 
                           refit='f1', cv=skf, return_train_score=True,
                           n_jobs=-1, verbose=2).fit(X_train, y_train)


#%% Evaluation: compute metrics, error analysis
print('Mean cross-validated score of the best_estimator:', clf_gs.best_score_)
print('Parameter setting that gave the best results on hold out data', clf_gs.best_params_)

# Plots to explore results of cross-validation
params = ['param_max_features', 'param_n_estimators']
metrics = ['mean_test_f1', 'mean_fit_time']
fig, ax = plt.subplots(2,2, figsize=[10,10])

ax[0,0].plot(clf_gs.cv_results_[params[0]].tolist(), clf_gs.cv_results_[metrics[0]], 'o')
ax[0,0].set_xlabel(params[0]); ax[0,0].set_ylabel(metrics[0]);

ax[0,1].plot(clf_gs.cv_results_[params[1]].tolist(), clf_gs.cv_results_[metrics[0]], 'o')
ax[0,1].set_xlabel(params[1]); ax[0,1].set_ylabel(metrics[0]);

ax[1,0].plot(clf_gs.cv_results_[params[0]].tolist(), clf_gs.cv_results_[metrics[1]], 'o')
ax[1,0].set_xlabel(params[0]); ax[1,0].set_ylabel(metrics[1]);

ax[1,1].plot(clf_gs.cv_results_[params[1]].tolist(), clf_gs.cv_results_[metrics[1]], 'o')
ax[1,1].set_xlabel(params[1]); ax[1,1].set_ylabel(metrics[1]);


#%% Final evaluation on test data
y_pred = clf_gs.predict(X_test)
score = f1_score(y_test, y_pred)
print('Final test metrics:', score)