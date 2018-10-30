from Dataset_prepare import *
import scipy.io as sio
import random


main_location = 'D:\\CSJ\\데이터 선별\\marker_regression_코드(호승형)\\DB\\DB_processed2\\DB_raw2_marker_wsize_1_winc_1_emg_wsize_408_winc_17_delay_0\\regression_no_smoothing\\DB'

marker_num = '1'

cordinate_num = '1'

emg_pair = '1'

# 불러올 파일 이름 
dep_ind = 'dep'; kfold = '3'; ext = '.mat'

seq_length = 20

# 사용할 emg channel 선택
emg_channel = [1,2,4]

#chan_idx = 0,1,2,3
chan_idx = np.array(emg_channel) - 1


# training 할 표정 선택
# 화남: 1, 어금니깨물기:2, 비웃음(왼):3, 비웃음(오):4, 눈세게감기:5, 두려움:6, 행복:7
# 키스:8, 무표정:9, 슬픔:10, 놀람:11
emo = 7,11

test_sample_rate = 120


# 파일 위치 불러오기
files_location, files_name = find_file_address(main_location, marker_num, cordinate_num, emg_pair)

# training data set 만들기
x_train, y_train = making_train_dataset(dep_ind,kfold,ext,files_name,files_location,emo,seq_length,emg_channel,chan_idx)

# test data set 만들기
x_test, y_test = making_test_dataset(dep_ind,kfold,ext,files_name,files_location,emo,seq_length,emg_channel,chan_idx,test_sample_rate)





    





























