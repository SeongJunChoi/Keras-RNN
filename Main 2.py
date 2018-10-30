from Dataset_prepare import *  # Dataset 준비 함수들
from Making_model import *  # RNN 모델 함수들
from fclone_Dataset_prepare import *  # Dataset 준비 함수들

import scipy.io as sio
import random
import numpy as np
import string
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from tqdm import trange
# keras
from sklearn.linear_model import LinearRegression
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
import keras.backend as K
from keras.callbacks import EarlyStopping
# tunning
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

# import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# model save
from keras.models import load_model

# mdn layer
import mdn

main_location = 'D:\\CSJ\\Tensorflow\\Keras-RNN'

# =============================================================================
# Data set 준비 관련 parameter
# =============================================================================
# data 길이
seq_length = 30

# 사용할 emg channel 선택
emg_channel = [1, 2, 3, 4]

# chan_idx = 0,1,2,3
chan_idx = np.array(emg_channel) - 1

# test data 표정짓고 난뒤에 몇초 쓸껀지
cut_rate = 4

emg_pair = '1'

# marker 위치
# 1: Head rotation X, 2: Head rotation Y,3: Head rotation Z,
# 4: Brow Left UP, 5: Brow Left Down, 6: Brow Right UP, 7: Brow Right Down,
# 8: Brow Centering, 9: Left Brow Outer Down, 10: Right Brow Outer Down
# 11: Eye close Left 12: Eye close Right
# 13: Mouse Open, 14: Mouse Left Smile, 15: Mouse Right Smile,
# 16: Mouse Left Spread, 17: Mouse Right Spread 18: Mouse Left Frawn, 19: Mouse Right Frawn
# 20: Mouse Left Centering, 21: Mouse Right Centering,
# 22: Cheek Left UP, 23: Cheek Right UP
# 24: Left Eye rotation X, 25: Left Eye rotation Y, 26: Left Eye rotation Z
# 27: Right Eye rotation X, 28: Right Eye rotation Y, 29: Right Eye rotation Z

# 29개 marker 중에서 사용한 marker 고르기
mark_position = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
# mark_position = [4]

mark_idx = np.array(mark_position) - 1

# training 할 표정 선택
# 무표정: 1, 눈썹 모으기: 2, 눈썹 올리기: 3, 입모양 'ㅏ': 4, 입모양 'ㅔ': 5
# 입모양 'ㅣ': 6, 입모양 'ㅗ': 7, 입모양 'ㅜ': 8, 입 벌린후 왼쪽 광대 및 입꼬리 올리기: 9,
# 입 벌린후 오른쪽 광대 및 입꼬리 올리기: 10, 왼쪽 입꼬리 및 광대 올리기: 11,
# 양쪽 입꼬리 및 광대 올리기: 12, 오른쪽 입꼬리 및 광대 올리기: 13,
# 왼쪽 입꼬리 땡기기: 14, 오른쪽 입꼬리 땡기기: 15

# 15개 표정중에서 사용할 표정 고르기
# emo = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
emo = [1, 4, 5, 6, 7, 8, 9, 10, 12]

# 20개 trial 중에서 사용할 trial 고르기
N_trial = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

test_sample_rate = 30
# =============================================================================
# 모델 관련 parameter
# =============================================================================
epoch = 100
batch_size = 128
stack = 3
drop_out = 0.3
hidden_size = 512
N_MIXES = 2
OUTPUT_DIMS = int(len(mark_position))
temperature = 0.1

# para, newpath = param_string(test_sample_rate, emo, seq_length, epoch, stack, hidden_size)
para, newpath = MDN_param_string(test_sample_rate, emo, seq_length, epoch, stack, hidden_size,N_MIXES,temperature)

print('=' * 100);
print('Model의 parameter는 다음과 같습니다.');
print();
print(para)


# =============================================================================
# training data set 만들기
# =============================================================================
x_train, y_train, x_test, y_test, train_trial, test_trial = making_test_and_train_dataset(emo, mark_position, mark_idx,
                                                                                          emg_channel, chan_idx,
                                                                                          N_trial, seq_length)
# 코드 확인용으로 데이터 자르기
# x_train = x_train[:50,:,:]
# y_train = y_train[:50,:]
# x_test = x_test[:50,:,:]
# y_test = y_test[:50,:]
# val_x_train = val_x_train[:50,:,:]
# val_y_train = val_y_train[:50,:]

# =============================================================================
# 원하는 model 만들기
# =============================================================================
# model = stateful_stacked_RNN(seq_length, stack, hidden_size,drop_out)
# model = lstm(seq_length, stack, hidden_size,drop_out)
# model = original_lstm(seq_length, hidden_size,drop_out)
# model = fclone_lstm(seq_length, stack, hidden_size,drop_out)
#model = fclone_RMDN(seq_length, stack, hidden_size, drop_out, N_MIXES, OUTPUT_DIMS, mark_position)
model = fclone_bidirectional_RMDN(seq_length, stack, hidden_size, drop_out, N_MIXES, OUTPUT_DIMS, mark_position)


# =============================================================================
# model 학습 시키기
# =============================================================================
class CustomHistory(keras.callbacks.Callback):
    def init(self):
        self.train_loss = []
        self.val_loss = []

    def on_epoch_end(self, batch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        
custom_hist = CustomHistory()
custom_hist.init()

# 학습 조기종료 조건
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

# tensorboard 경로 설정
# tensorboard --logdir=D:/CSJ/Tensorflow/Keras-RNN/logs
tb_hist = keras.callbacks.TensorBoard(log_dir=newpath, histogram_freq=0, write_graph=True, write_images=True)



# <stateful-LSTM>
# for i in trange(epoch):
#    model.fit(x_train, y_train, epochs=1, batch_size=1, validation_data=(val_x_train, val_y_train), verbose=1,
#              shuffle=False, callbacks=[custom_hist, tb_hist, early_stop])
#    model.reset_states()
# y_pred = model.predict(x_test,batch_size=1,verbose=0)

# <일반 LSTM>
#model.fit(x_train, y_train, epochs=epoch, batch_size=30, validation_data=(x_test, y_test), verbose=1, shuffle=False,
#          callbacks=[custom_hist, tb_hist])
#y_pred = model.predict(x_test)

# <RMDN>
model.fit(x_train, y_train, epochs=epoch, batch_size=30, validation_data=(x_test, y_test), verbose=1, shuffle=False,
          callbacks=[custom_hist, tb_hist])

tmp_y_pred = model.predict(x_test)
y_pred = np.apply_along_axis(mdn.sample_from_output, 1, tmp_y_pred, OUTPUT_DIMS, N_MIXES, temp=temperature)
y_pred = np.squeeze(y_pred)

# =============================================================================
# 데이터 저장하기
# =============================================================================
saving_data(custom_hist, y_test, y_pred, model, para, newpath)

# model 저장하기
model_name = newpath + '\\model.h5'
model.save(model_name)












