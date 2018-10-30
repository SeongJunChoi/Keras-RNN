from Dataset_prepare import * # Dataset 준비 함수들
from Making_model import* # RNN 모델 함수들
from fclone_Dataset_prepare import * # Dataset 준비 함수들


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

main_location = 'D:\\CSJ\\데이터 선별\\marker_regression_코드(호승형)\\DB\\DB_processed2\\DB_raw2_marker_wsize_1_winc_1_emg_wsize_408_winc_17_delay_0\\regression\\DB_smooth'



# =============================================================================
# Data load 관련 parameter
# =============================================================================
marker_num = '21'
cordinate_num = '1'
emg_pair = '1'

# 불러올 파일 이름 
dep_ind = 'dep'; kfold = '1'; ext = '.mat'

# =============================================================================
# Data set 준비 관련 parameter
# =============================================================================
# data 길이
seq_length = 30
# 사용할 emg channel 선택
emg_channel = [1,2,3,4]
#chan_idx = 0,1,2,3
chan_idx = np.array(emg_channel) - 1
# test data 표정짓고 난뒤에 몇초 쓸껀지
cut_rate = 4
# training 할 표정 선택
# 화남: 1, 어금니깨물기:2, 비웃음(왼):3, 비웃음(오):4, 눈세게감기:5, 두려움:6, 행복:7
# 키스:8, 무표정:9, 슬픔:10, 놀람:11


emo = 7,11
test_sample_rate = 120
# =============================================================================
# 모델 관련 parameter
# =============================================================================
epoch = 1
batch_size=20
stack =1
drop_out = 0.2
hidden_size = 32

para,newpath = param_string(test_sample_rate,emo,seq_length,epoch,stack,hidden_size)
print('=' * 100); print('Model의 parameter는 다음과 같습니다.'); print(); print(para)
# =============================================================================
# main 
# =============================================================================
# 파일 위치 불러오기
files_location, files_name = find_file_address(main_location, marker_num, cordinate_num, emg_pair)

# training data set 만들기
x_train, y_train, val_x_train, val_y_train = making_train_validation_dataset(dep_ind,kfold,ext,files_name,files_location,emo,seq_length,emg_channel,chan_idx)

# test data set 만들기
x_test, y_test = making_test_dataset(dep_ind,kfold,ext,files_name,files_location,emo,seq_length,emg_channel,chan_idx,test_sample_rate,cut_rate)

# 코드 확인용으로 데이터 자르기
#x_train = x_train[:50,:,:]
#y_train = y_train[:50,:]
#x_test = x_test[:50,:,:]
#y_test = y_test[:50,:]
#val_x_train = val_x_train[:50,:,:]
#val_y_train = val_y_train[:50,:]


# 원하는 model 만들기
#model = stateful_stacked_RNN(seq_length, stack, hidden_size,drop_out)
model = lstm(seq_length, stack, hidden_size,drop_out)
# model = original_lstm(seq_length, hidden_size,drop_out)

# =============================================================================
# 
# =============================================================================
class CustomHistory(keras.callbacks.Callback):
    def init(self):
        self.train_loss=[]
        self.val_loss = []
        
    def on_epoch_end(self,batch,logs={}):
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        


#학습 조기종료 조건
early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1)

# tensorboard 경로 설정
#tensorboard --logdir=D:/CSJ/Tensorflow/Keras-RNN/logs
tb_hist = keras.callbacks.TensorBoard(log_dir=newpath,histogram_freq=0, write_graph=True,write_images=True)


# model 학습 시키기
custom_hist = CustomHistory()
custom_hist.init()

# =============================================================================
# stateful-LSTM 일경우
# =============================================================================
#for i in trange(epoch):
#    model.fit(x_train, y_train, epochs=1, batch_size=1, validation_data=(val_x_train, val_y_train), verbose=1,
#              shuffle=False, callbacks=[custom_hist, tb_hist, early_stop])
#    model.reset_states()
#
#y_pred = model.predict(x_test,batch_size=1,verbose=0)

# =============================================================================
# 일반 LSTM 일 경우
# =============================================================================
model.fit(x_train,y_train, epochs=epoch, batch_size=30, validation_data=(val_x_train,val_y_train), verbose=1, shuffle=False, callbacks=[custom_hist, tb_hist, early_stop])
y_pred = model.predict(x_test)

# 데이터 저장하기
saving_data(custom_hist,y_test,y_pred,model,para,newpath)

# model 저장하기
model_name = newpath + '\\model.h5'
model.save(modelname)

# 학습과정 plot
#plt.plot(custom_hist.train_loss)
#plt.plot(custom_hist.val_loss)
#plt.ylim(0.0,0.1)
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train','val'],loc='upper left')
#plt.show()

# model 평가하기
#trainScore = model.evaluate(x_train, y_train,verbose=0,batch_size=batch_size)
#model.reset_states()
#print('Train Score: ',trainScore)
#valScore = model.evaluate(val_x_train, val_y_train, verbose=0, batch_size=batch_size)
#model.reset_states()
#print('Validation Score: ', valScore)
#testScore = model.evaluate(x_test, y_test, verbose=0,batch_size=batch_size)
#model.reset_states()
#print('Test Score: ', testScore)




















