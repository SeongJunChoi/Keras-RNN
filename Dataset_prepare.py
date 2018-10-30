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
# random 함수
import random
import keras
from keras.utils.vis_utils import plot_model






# =============================================================================
# 이번 실험의 data의 이름과 위치를 읽어 옴
# =============================================================================
def find_file_address (main_location, marker_num, cordinate_num, emg_pair) :

    """

        Input
            main_location       :   모든 data들이 들어있는 폴더의 주소
            marker_num          :   몇 번 marker를 사용할 것인지 선택
            cordinate_num       :   marker 좌표 x:1, y:2, z:3
            emg_pair            :   emg 채널 선택 조합
            want_load_type      :   불어올 data들의 확장자 type   ex) csv

        Output
            files_dir           :   모든 파일의 전체 주소가 저장된 list
            files_name          :   모든 파일 이름이 저장된 list

    """

    ##  변수 초기화
    #   모든 파일의 주소를 저장할 dictionary 변수 초기화
    files_dir = []
    #   모든 파일의 이름을 저장할 list 변수 초기화
    files_name = []
    
    # marker 번호
    num_marker = 'mark_' + marker_num + '_'
    
    # marker 좌표
    cor_marker = 'xyz_' + cordinate_num + '_'
    
    #emg 조합
    emg = 'emg_pair_' + emg_pair + '_'
    
    #해당 data folder 주소
    folder = num_marker + cor_marker + emg + 'mv_size_5'
    
    ##  해당 실험 data가 들어있는 폴더 주소
    now_data_location = main_location + '\\' + folder
    
    # data 이름과 위치를 리스트로 받아옴 
    for (path, dir, files) in os.walk(now_data_location,topdown=True):
        files.sort()
    for filename in files:
        files_name.append(filename)
        filepath = os.path.join(now_data_location,filename)
        files_dir.append(filepath)
    print('=' * 100) ; print('선택된 폴더는 다음과 같습니다.');print(folder); print();
    print('선택된 파일은 다음과 같습니다.') ; print()
    print(files_name)
        
    return files_dir, files_name



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
def making_train_validation_dataset(dep_ind,kfold,ext,files_name,files_location,emo,seq_length,emg_channel,chan_idx):
    
        
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
    
    # 불러올 파일 이름 
    sel_file = dep_ind + '_kfold_' + kfold + ext
    
    # 불러올 파일 이름의 list의 index
    idx = files_name.index(sel_file)
    
    # index에 맞는 파일 불러오기
    data = sio.loadmat(files_location[idx])
    
    #################### training data 표정별로 나누기 #########################
    # mat 파일 읽기
    train_idx = data['train_fe_index']
    train_emg = data['xtrain']
    train_mark = data['ytrain']
    
    # 선택한 표정의 index에 맞는 표정만 원래 데이터에서 가져오기 (EMG)
    ## 선택한 표정의 index를 읽은후 train data에서 index에 맞는 데이터만 불러옴
    temp_train_emg = []
    print('=' * 100); print(); print('Train Data & Validation Data Set 만들기 시작합니다.')
    for i in range(len(emo)):
        print(); print('%d번 표정의 EMG data를 선택중입니다.' %emo[i]); 
        tem_idx = np.where((train_idx==emo[i]))
        temp = train_emg[tem_idx[0]]
        temp_train_emg.extend(temp)    
    temp_train_emg = np.array(temp_train_emg)
    
    # 선택한 표정의 index에 맞는 표정만 원래 데이터에서 가져오기 (Marker)
    temp_train_mark = []
    for i in range(len(emo)):
        print(); print('%d번 표정의 Marker data를 선택중입니다.' %emo[i]); 
        tem_idx = np.where((train_idx==emo[i]))
        temp = train_mark[tem_idx[0]]
        temp_train_mark.extend(temp)    
    temp_train_mark = np.array(temp_train_mark)
    
    
    ### training set & validation set으로 나누기 ###
    # random index 생성
    ratio = int(len(temp_train_emg) * 0.2)
    rand_idx = np.random.permutation(temp_train_emg.shape[0])
    
    # random index대로 섞기
    temp_train_emg = temp_train_emg[rand_idx,:]
    temp_train_mark = temp_train_mark[rand_idx,:]
    
    # validation set 생성
    temp_val_train_emg = temp_train_emg[:ratio,:]
    temp_val_train_mark = temp_train_mark[0:ratio,:]
    
    # training set 생성
    temp_train_emg = temp_train_emg[ratio:,:]
    temp_train_mark = temp_train_mark[ratio:,:]
    
    
     # 표정에 맞게 가져온 데이터를 sequence 길이에 맞게 자르기 (train data set)
    x_train = []
    y_train = []

    print(); print('=' * 100) ; print(); print('sequence size: %d으로 train data를 만드는 중입니다.' %seq_length)
    print(); print('선택한 EMG channel은 %s 입니다.' %(emg_channel,))
    for i in range(len(temp_train_emg)):
        temp1 = temp_train_emg[i,0]
        # 선택한 emg 채널만 사용하도록 선택
        temp1 = temp1[:,chan_idx]
        temp2 = temp_train_mark[i,0]
        
        # sequence window에 맞게 자르기
        temp_x_train = seq2dataset(temp1,seq_length) 
        temp_y_train = temp2[seq_length-1:,:] 
        
        # 자른 데이터 붙이기
        x_train.extend(temp_x_train)
        y_train.extend(temp_y_train)
        
        
    # 표정에 맞게 가져온 데이터를 sequence 길이에 맞게 자르기 (validation data set)   
    val_x_train = []
    val_y_train= []    
    print(); print('20% data로 validation data를 자르는 중입니다.')
    for i in range(len(temp_val_train_emg)):
        temp3 = temp_val_train_emg[i,0]
        # 선택한 emg 채널만 사용하도록 선택
        temp3 = temp3[:,chan_idx]
        temp4 = temp_val_train_mark[i,0]      
        
        # sequence window에 맞게 자르기
        temp_val_x_train = seq2dataset(temp3, seq_length)
        temp_val_y_train = temp4[seq_length-1:,:]
        
        # 자른 데이터 붙이기
        val_x_train.extend(temp_val_x_train)
        val_y_train.extend(temp_val_y_train)

        
    print(); print('권장하는 batch size는 %d 입니다.' %len(temp_x_train))   
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    val_x_train = np.array(val_x_train)
    val_y_train = np.array(val_y_train)
    
    return x_train, y_train, val_x_train, val_y_train

# =============================================================================
# training data set 만 만드는 함수
# =============================================================================
def making_train_dataset(dep_ind,kfold,ext,files_name,files_location,emo,seq_length,emg_channel,chan_idx):
    
        
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
    
    # 불러올 파일 이름 
    sel_file = dep_ind + '_kfold_' + kfold + ext
    
    # 불러올 파일 이름의 list의 index
    idx = files_name.index(sel_file)
    
    # index에 맞는 파일 불러오기
    data = sio.loadmat(files_location[idx])
    
    #################### training data 표정별로 나누기 #########################
    # mat 파일 읽기
    train_idx = data['train_fe_index']
    train_emg = data['xtrain']
    train_mark = data['ytrain']
    
    # 선택한 표정의 index에 맞는 표정만 원래 데이터에서 가져오기 (EMG)
    ## 선택한 표정의 index를 읽은후 train data에서 index에 맞는 데이터만 불러옴
    temp_train_emg = []
    print('=' * 100); print(); print('Train Data Set 만들기 시작합니다.')
    for i in range(len(emo)):
        print(); print('%d번 표정의 EMG data를 선택중입니다.' %emo[i]); 
        tem_idx = np.where((train_idx==emo[i]))
        temp = train_emg[tem_idx[0]]
        temp_train_emg.extend(temp)    
    temp_train_emg = np.array(temp_train_emg)
    
    # 선택한 표정의 index에 맞는 표정만 원래 데이터에서 가져오기 (Marker)
    temp_train_mark = []
    for i in range(len(emo)):
        print(); print('%d번 표정의 Marker data를 선택중입니다.' %emo[i]); 
        tem_idx = np.where((train_idx==emo[i]))
        temp = train_mark[tem_idx[0]]
        temp_train_mark.extend(temp)    
    temp_train_mark = np.array(temp_train_mark)
    
    
     # 표정에 맞게 가져온 데이터를 sequence 길이에 맞게 자르기 (train data set)
    x_train = []
    y_train = []

    print(); print('=' * 100) ; print(); print('sequence size: %d으로 train data를 만드는 중입니다.' %seq_length)
    print(); print('선택한 EMG channel은 %s 입니다.' %(emg_channel,))
    for i in range(len(temp_train_emg)):
        temp1 = temp_train_emg[i,0]
        # 선택한 emg 채널만 사용하도록 선택
        temp1 = temp1[:,chan_idx]
        temp2 = temp_train_mark[i,0]
        
        # sequence window에 맞게 자르기
        temp_x_train = seq2dataset(temp1,seq_length) 
        temp_y_train = temp2[seq_length-1:,:]  
        
        # 자른 데이터 붙이기
        x_train.extend(temp_x_train)
        y_train.extend(temp_y_train)
        
    print(); print('권장하는 batch size는 %d 입니다.' %len(temp_x_train))   
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    return x_train, y_train

# =============================================================================
# test data set 만들기
# =============================================================================
def making_test_dataset(dep_ind,kfold,ext,files_name,files_location,emo,seq_length,emg_channel,chan_idx,test_sample_rate,cut_rate):

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

    # 불러올 파일 이름 
    sel_file = dep_ind + '_kfold_' + kfold + ext
    
    # 불러올 파일 이름의 list의 index
    idx = files_name.index(sel_file)
    
    # index에 맞는 파일 불러오기
    data = sio.loadmat(files_location[idx])
    
    #################### test data 표정별로 나누기 #########################
    # mat 파일 읽기
    test_idx = data['test_fe_index']
    test_trigger = data['ytest_valid']
    test_emg = data['xtest']
    test_mark = data['ytest']
    
    x_test = []
    y_test = []
    print('=' * 100); print(); print('Test Data Set 만들기 시작합니다.')
    print(); print('선택한 EMG channel은 %s 입니다.' %(emg_channel,))
    for i in range(len(test_emg)):
        temp_idx = test_idx[i,0]

        temp_trigger = test_trigger[i,0]
        
        # test할 표정 index에 맞는 trigger 찾음
        temp_idx = np.where((temp_idx==emo))[0]
        temp_trigger = temp_trigger[temp_idx]
        
        ################ trigger에 맞게 test data 자름 ####################
        temp_test_emg = test_emg[i,0]
        temp_test_mark = test_mark[i,0]
        
        # 선택한 채널 data 선택
        
        temp_test_emg = temp_test_emg[:,chan_idx]
        
        # 표정 갯수만큼 trigger 자르기
        temp_e=[]
        temp_m=[]
        st = test_sample_rate*1
        en = test_sample_rate*cut_rate
        for k in range(len(temp_trigger)):
            pt = temp_trigger[k,0] 
            tempemg = temp_test_emg[pt-st:pt+en]
            tempmark = temp_test_mark[pt-st:pt+en]
            
            # sequence 길이 만큼 자르기
            seq_emg = seq2dataset(tempemg,seq_length)
            seq_mark = tempmark[seq_length-1:,:]
            
            temp_e.extend(seq_emg)
            temp_m.extend(seq_mark)
            
        # 표정마다 자른 sequence 붙이기    
        temp_e = np.array(temp_e) 
        temp_m = np.array(temp_m)  
        
        # trial 마다 붙이기
        x_test.extend(temp_e)
        y_test.extend(temp_m)
    
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    return x_test, y_test

# =============================================================================
# 
# =============================================================================
def saving_data(custom_hist,y_test,y_pred,model,para,newpath):
    
    
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
    
        
    # # marker 결과 저장 위치
    # save_path = os.getcwd() + '\\result\\predict'
    #
    # # paratmeter에 따른 결과 저장 위치 생성
    # foldername = str(test_sample_rate) + 'Hz_EMO_' + ' '.join(map(str,(emo))) + '_seq_' + str(seq_length) + '_epoch_' + str(epoch) + '_stack_' + str(stack) + '_hidden_' + str(hidden_size)
    # newpath = save_path + '\\'+ foldername
    # if not os.path.exists(newpath):
    #     os.makedirs(newpath)

    p_filename = para + '_predicted'
    l_filename = para + '_loss'
    
#    output = np.hstack((y_test,y_pred))
    loss = np.vstack((custom_hist.train_loss, custom_hist.val_loss));
    loss = loss.transpose()
    po_save_name = newpath + '\\predicted_original.csv'
    pp_save_name = newpath + '\\predicted_predicted.csv'
    l_save_name = newpath + '\\loss.csv'
    print('=' * 100);
    print();
    print('데이터를 저장중입니다.')

    # data 저장
    np.savetxt(po_save_name, y_test, fmt='%.6f', delimiter=',', header=" original")
    np.savetxt(pp_save_name, y_pred, fmt='%.6f', delimiter=',', header=" predict")
    np.savetxt(l_save_name, loss, fmt='%.6f', delimiter=',', header="train_loss,  val_loss")

    # model 저장
    plot_save_name = newpath + '\\model.png'
    plot_model(model, to_file=plot_save_name, show_shapes=True, show_layer_names=True)



# =============================================================================
#
# =============================================================================
def param_string(test_sample_rate,emo,seq_length,epoch,stack,hidden_size):
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
    para = str(test_sample_rate) + 'Hz_EMO_' + ' '.join(map(str,(emo))) + '_seq_' + str(seq_length) + '_epoch_' + str(epoch) + '_stack_' + str(stack) + '_hidden_' + str(hidden_size)

    # 저장 위치
    save_path = os.getcwd() + '\\result\\predict'

    # paratmeter에 따른 결과 저장 위치 생성
    newpath = save_path + '\\'+ para
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    return para,newpath


# =============================================================================
#
# =============================================================================
def MDN_param_string(test_sample_rate, emo, seq_length, epoch, stack, hidden_size,N_MIXES,temperature):
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
    para = str(test_sample_rate) + 'Hz_EMO_' + ' '.join(map(str, (emo))) + '_seq_' + str(seq_length) + '_epoch_' + str(
        epoch) + '_stack_' + str(stack) + '_hidden_' + str(hidden_size)  +'_n_mixes_' + str(N_MIXES) +'_temeperature_' + str(temperature)

    # 저장 위치
    save_path = os.getcwd() + '\\result\\predict'

    # paratmeter에 따른 결과 저장 위치 생성
    newpath = save_path + '\\' + para
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    return para, newpath




























