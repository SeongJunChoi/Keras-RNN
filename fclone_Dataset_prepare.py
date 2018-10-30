# RNN에 들어갈 data 가공하는 함수들 ##
# =============================================================================
# 사용할 lib 
# =============================================================================
import os
from os import listdir
#   glob lib 소환 (여러 파일을 한 번에 불러오기 위해)
import glob
#   csv lib 소환 (csv 파일로 불러오기 위해)
import csv
#   numpy lib 소환 (행렬 계산을 위해)
import numpy as np
#   pandas lib 소환 (csv 파일을 불러오기 위해)
import pandas as pd
#   random lib 소환 (data의 순서를 섞기 위해)
import random
#   tqdm에 있는 trange lib 소환 (progress bar를 출력하기 위해)
from tqdm import trange
# mat file 읽어오는 lib
import scipy.io as sio
# 조합 만드는 함수
import itertools as it
# random 함수
import random
import keras
from keras.utils.vis_utils import plot_model

# =============================================================================
# 윈도우 크기 만큼 잘라서 데이터셋을 만드는 함수  
# =============================================================================
def seq2dataset(data,seq_length):
    
    """

        Input
            data                    :   입력 데이터
            seq_length              :   window 크기, sequence length

        Output
            dataset                 :   잘린 크기


    """
    dataset=[]
    for i in range(len(data)-seq_length+1):
        subset = data[i:(i+seq_length)]
        dataset.append(subset)
    return np.array(dataset)
    

# =============================================================================
# training data set & validation set 만드는 함수
# =============================================================================
def making_test_and_train_dataset(emo,mark_position,mark_idx,emg_channel,chan_idx,N_trial,seq_length):
    
        
    """

        Input
            all_adjusted_input      :   원하는 길이로 편집된 input data set이 저장된 dictionary
            all_adjusted_target     :   원하는 길이로 편집된 target data set이 저장된 dictionary
            train_files             :   정해진 비율에 따라 train data로 지정된 files의 이름이 저장된 list
            test_files              :   정해진 비율에 따라 test data로 지정된 files의 이름이 저장된 list

        Output
            x_train                 :   EMG dat
            y_train                 :   Train target data의 값이 저장된 array

    """
    
   
    # 파일 불러오기
    data_emg = sio.loadmat("EMG.mat")
    data_marker = sio.loadmat("Mark.mat")
#    # 29개 marker 중에서 사용한 marker 고르기
#    mark_position = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
#    mark_idx = np.array(mark_position)-1
#    emg_channel = [1,2,3,4]
#
#    # 20개 trial 중에서 사용할 trial 고르기
#    N_trial = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    all_trial = list(it.combinations(N_trial,15))
    t = random.randrange(1,len(all_trial))
    train_trial = list(all_trial[t])
    test_trial = []
    # train / test trial 나누기
    for i in N_trial:
        if i not in train_trial:
            test_trial.append(i)   
    print('=' * 100); print(); print('Train trial은 다음과 같습니다.');print(train_trial);
    print('=' * 100); print(); print('Test trial은 다음과 같습니다.');print(test_trial);
    print('=' * 100); print(); print('선택한 표정은 다음과 같습니다.');print(emo);


    # mat 파일 읽기
    emg = data_emg['emg_se']
    marker = data_marker['mark_se']
    
        
    # Train에서 선택한 trial 과 표정에서 emg data set 만들기
    tmp_train_emg=[]
#    print('=' * 100); print(); print('EMG의 Train Data set을 만드는 중입니다.');
    for i in range(len(train_trial)):
        for j in range(len(emo)):
            trial_idx = train_trial[i]-1
            emo_idx = emo[j]-1
            tmp = emg[trial_idx,emo_idx]
            tmp_train_emg.append(tmp)
    tmp_train_emg = np.array(tmp_train_emg)
    
    # Test에서 선택한 trial과 표정에서 emg data set 만들기
    tmp_test_emg=[]
#    print('=' * 100); print(); print('EMG의 Test Data set을 만드는 중입니다.');
    for i in range(len(test_trial)):
        for j in range(len(emo)):
            trial_idx = train_trial[i]-1
            emo_idx = emo[j]-1
            tmp = emg[trial_idx,emo_idx]
            tmp_test_emg.append(tmp)
    tmp_test_emg = np.array(tmp_test_emg)
    
    # Train에서 선택한 trial 과 표정에서 marker data set 만들기
    tmp_train_mark=[]
#    print('=' * 100); print(); print('Marker의 Train Data set을 만드는 중입니다.');
    for i in range(len(train_trial)):
        for j in range(len(emo)):
            trial_idx = train_trial[i]-1
            emo_idx = emo[j]-1
            tmp = marker[trial_idx,emo_idx]
            #선택한 marker 만 선택
            tmp = tmp[:,mark_idx]
            tmp_train_mark.append(tmp)
    tmp_train_mark = np.array(tmp_train_mark)
    
    # Test에서 선택한 trial과 표정에서 emarker data set 만들기
    tmp_test_mark=[]
#    print('=' * 100); print(); print('Marker의 Test Data set을 만드는 중입니다.');
    for i in range(len(test_trial)):
        for j in range(len(emo)):
            trial_idx = train_trial[i]-1
            emo_idx = emo[j]-1
            tmp = marker[trial_idx,emo_idx]
            #선택한 marker 만 선택
            tmp = tmp[:,mark_idx]
            tmp_test_mark.append(tmp)
    tmp_test_mark = np.array(tmp_test_mark)
      
    
     # 표정에 맞게 가져온 데이터를 sequence 길이에 맞게 자르기 (train data set)
    x_train = []
    y_train = []
    print(); print('=' * 100) ; print(); print('sequence size: %d으로 train data를 만드는 중입니다.' %seq_length)
    print(); print('선택한 EMG channel은 %s 입니다.' %(emg_channel,))
    for i in range(len(tmp_train_emg)):
        # 선택한 emg 채널만 사용하도록 선택
        temp1 = tmp_train_emg[i,:,:]
        temp1 = temp1[:,chan_idx]
        # markert 사용하도록 선택
        temp2 = tmp_train_mark[i,:,:]
        
        # sequence window에 맞게 자르기
        temp_x_train = seq2dataset(temp1,seq_length) 
        temp_y_train = temp2[seq_length-1:,:] 
        
        # 자른 데이터 붙이기
        x_train.extend(temp_x_train)
        y_train.extend(temp_y_train)
        
    # 표정에 맞게 가져온 데이터를 sequence 길이에 맞게 자르기 (test data set)
    x_test = []
    y_test = []
    print(); print('=' * 100) ; print(); print('sequence size: %d으로 test data를 만드는 중입니다.' %seq_length)
    print(); print('선택한 EMG channel은 %s 입니다.' %(emg_channel,))
    for i in range(len(tmp_test_emg)):
        # 선택한 emg 채널만 사용하도록 선택
        temp1 = tmp_test_emg[i,:,:]
        temp1 = temp1[:,chan_idx]
        # markert 사용하도록 선택
        temp2 = tmp_test_mark[i,:,:]
        
        # sequence window에 맞게 자르기
        temp_x_test = seq2dataset(temp1,seq_length) 
        temp_y_test = temp2[seq_length-1:,:] 
        
        # 자른 데이터 붙이기
        x_test.extend(temp_x_test)
        y_test.extend(temp_y_test)
            
      
    print(); print('권장하는 batch size는 %d 입니다.' %len(temp_x_train))   
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    
    return x_train, y_train, x_test, y_test, train_trial, test_trial

# =============================================================================
# 
# =============================================================================
def making_test_and_train_dataset_minmaxscaling(emo,mark_position,mark_idx,emg_channel,chan_idx,N_trial,seq_length):
    
        
    """

        Input
            all_adjusted_input      :   원하는 길이로 편집된 input data set이 저장된 dictionary
            all_adjusted_target     :   원하는 길이로 편집된 target data set이 저장된 dictionary
            train_files             :   정해진 비율에 따라 train data로 지정된 files의 이름이 저장된 list
            test_files              :   정해진 비율에 따라 test data로 지정된 files의 이름이 저장된 list

        Output
            x_train                 :   EMG dat
            y_train                 :   Train target data의 값이 저장된 array

    """
    
   
    # 파일 불러오기
    data_emg = sio.loadmat("EMG.mat")
    data_marker = sio.loadmat("Mark.mat")
#    # 29개 marker 중에서 사용한 marker 고르기
#    mark_position = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
#    mark_idx = np.array(mark_position)-1
#    emg_channel = [1,2,3,4]
#
#    # 20개 trial 중에서 사용할 trial 고르기
#    N_trial = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    all_trial = list(it.combinations(N_trial,15))
    t = random.randrange(1,len(all_trial))
    train_trial = list(all_trial[t])
    test_trial = []
    # train / test trial 나누기
    for i in N_trial:
        if i not in train_trial:
            test_trial.append(i)   
    print('=' * 100); print(); print('Train trial은 다음과 같습니다.');print(train_trial);
    print('=' * 100); print(); print('Test trial은 다음과 같습니다.');print(test_trial);
    print('=' * 100); print(); print('선택한 표정은 다음과 같습니다.');print(emo);


    # mat 파일 읽기
    emg = data_emg['emg_se']
    marker = data_marker['mark_se']
    
        
    # Train에서 선택한 trial 과 표정에서 emg data set 만들기
    tmp_train_emg=[]
#    print('=' * 100); print(); print('EMG의 Train Data set을 만드는 중입니다.');
    for i in range(len(train_trial)):
        for j in range(len(emo)):
            trial_idx = train_trial[i]-1
            emo_idx = emo[j]-1
            tmp = emg[trial_idx,emo_idx]
            tmp_train_emg.append(tmp)
    tmp_train_emg = np.array(tmp_train_emg)
    
    # Test에서 선택한 trial과 표정에서 emg data set 만들기
    tmp_test_emg=[]
#    print('=' * 100); print(); print('EMG의 Test Data set을 만드는 중입니다.');
    for i in range(len(test_trial)):
        for j in range(len(emo)):
            trial_idx = train_trial[i]-1
            emo_idx = emo[j]-1
            tmp = emg[trial_idx,emo_idx]
            tmp_test_emg.append(tmp)
    tmp_test_emg = np.array(tmp_test_emg)
    
    # Train에서 선택한 trial 과 표정에서 marker data set 만들기
    tmp_train_mark=[]
#    print('=' * 100); print(); print('Marker의 Train Data set을 만드는 중입니다.');
    for i in range(len(train_trial)):
        for j in range(len(emo)):
            trial_idx = train_trial[i]-1
            emo_idx = emo[j]-1
            tmp = marker[trial_idx,emo_idx]
            #선택한 marker 만 선택
            tmp = tmp[:,mark_idx]
            tmp_train_mark.append(tmp)
    tmp_train_mark = np.array(tmp_train_mark)
    
    # Test에서 선택한 trial과 표정에서 emarker data set 만들기
    tmp_test_mark=[]
#    print('=' * 100); print(); print('Marker의 Test Data set을 만드는 중입니다.');
    for i in range(len(test_trial)):
        for j in range(len(emo)):
            trial_idx = train_trial[i]-1
            emo_idx = emo[j]-1
            tmp = marker[trial_idx,emo_idx]
            #선택한 marker 만 선택
            tmp = tmp[:,mark_idx]
            tmp_test_mark.append(tmp)
    tmp_test_mark = np.array(tmp_test_mark)
      
    
     # 표정에 맞게 가져온 데이터를 sequence 길이에 맞게 자르기 (train data set)
    x_train = []
    y_train = []
    print(); print('=' * 100) ; print(); print('sequence size: %d으로 train data를 만드는 중입니다.' %seq_length)
    print(); print('선택한 EMG channel은 %s 입니다.' %(emg_channel,))
    for i in range(len(tmp_train_emg)):
        # 선택한 emg 채널만 사용하도록 선택
        temp1 = tmp_train_emg[i,:,:]
        temp1 = temp1[:,chan_idx]
        # markert 사용하도록 선택
        temp2 = tmp_train_mark[i,:,:]
        
        # sequence window에 맞게 자르기
        temp_x_train = seq2dataset(temp1,seq_length) 
        temp_y_train = temp2[seq_length-1:,:] 
        
        # 자른 데이터 붙이기
        x_train.extend(temp_x_train)
        y_train.extend(temp_y_train)
        
    # 표정에 맞게 가져온 데이터를 sequence 길이에 맞게 자르기 (test data set)
    x_test = []
    y_test = []
    print(); print('=' * 100) ; print(); print('sequence size: %d으로 test data를 만드는 중입니다.' %seq_length)
    print(); print('선택한 EMG channel은 %s 입니다.' %(emg_channel,))
    for i in range(len(tmp_test_emg)):
        # 선택한 emg 채널만 사용하도록 선택
        temp1 = tmp_test_emg[i,:,:]
        temp1 = temp1[:,chan_idx]
        # markert 사용하도록 선택
        temp2 = tmp_test_mark[i,:,:]
        
        # sequence window에 맞게 자르기
        temp_x_test = seq2dataset(temp1,seq_length) 
        temp_y_test = temp2[seq_length-1:,:] 
        
        # 자른 데이터 붙이기
        x_test.extend(temp_x_test)
        y_test.extend(temp_y_test)
            
      
    print(); print('권장하는 batch size는 %d 입니다.' %len(temp_x_train))   
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    
    return x_train, y_train, x_test, y_test, train_trial, test_trial

  
# =============================================================================
# 
# =============================================================================





































