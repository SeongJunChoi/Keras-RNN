import mdn
import keras

# keras
from sklearn.linear_model import LinearRegression
from keras.layers.normalization import BatchNormalization
from keras.layers import LSTM, Dense, Dropout, LeakyReLU, ELU, PReLU, Bidirectional
from keras.models import Sequential
import keras.backend as K


# =============================================================================
#
# =============================================================================
def stateful_stacked_RNN(seq_length, stack, hidden_size, drop_out):
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

    # 상태유지 n stack RNN model
    K.clear_session()
    model = Sequential()
    # n stack
    for i in range(stack):
        model.add(LSTM(hidden_size, batch_input_shape=(1, seq_length, 4), stateful=True, return_sequences=True,
                       kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(ELU(alpha=1.0))
        model.add(Dropout(drop_out))
    model.add(LSTM(hidden_size, batch_input_shape=(1, seq_length, 1), stateful=True))
    model.add(Dropout(drop_out))
    model.add(Dense(1, kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(ELU(alpha=0.1))

    # model 학습 과정 설정하기
    model.compile(loss='mean_squared_error', optimizer='adam')

    # 생성된 model 확인
    model.summary()

    return model


# =============================================================================
#
# =============================================================================
def lstm(seq_length, stack, hidden_size, drop_out):
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

    # n stack RNN model
    K.clear_session()
    model = Sequential()
    # n stack
    for i in range(stack):
        model.add(LSTM(hidden_size, input_shape=(seq_length, 4), return_sequences=True, kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        # model.add(ELU(alpha=1.0))
        model.add(Dropout(drop_out))
    model.add(LSTM(10))
    model.add(Dense(1, kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    # model.add(LeakyReLU(alpha=0.1))
    model.add(ELU(alpha=1.0))

    # model 학습 과정 설정하기
    model.compile(loss='mean_squared_error', optimizer='adam')

    # 생성된 model 확인
    model.summary()

    return model


# =============================================================================
#
# =============================================================================
def original_lstm(seq_length, hidden_size, drop_out):
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

    K.clear_session()
    model = Sequential()  # Sequeatial Model
    model.add(LSTM(hidden_size, return_sequences=True, input_shape=(seq_length, 4)))  # (timestep, feature)\
    model.add(Dropout(drop_out))
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(LSTM(10))
    model.add(Dense(1))  # output = 1
    model.add(LeakyReLU(alpha=0.1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()

    return model


# =============================================================================
#
# =============================================================================
def fclone_lstm(seq_length, stack, hidden_size, drop_out):
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

    # n stack RNN model
    K.clear_session()
    model = Sequential()
    # n stack
    for i in range(stack):
        model.add(LSTM(hidden_size, input_shape=(seq_length, 4), return_sequences=True, kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        # model.add(ELU(alpha=1.0))
        model.add(Dropout(drop_out))
    model.add(LSTM(10))
    model.add(Dense(1, kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    # model.add(LeakyReLU(alpha=0.1))
    model.add(ELU(alpha=1.0))

    # model 학습 과정 설정하기
    model.compile(loss='mean_squared_error', optimizer='adam')

    # 생성된 model 확인
    model.summary()

    return model


# =============================================================================
#
# =============================================================================
def fclone_RMDN(seq_length, stack, hidden_size, drop_out, N_MIXES, OUTPUT_DIMS, mark_position):
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

    # n stack RNN model
    K.clear_session()
    model = Sequential()
    # n stack
    for i in range(stack):
        model.add(LSTM(hidden_size, input_shape=(seq_length, 4), return_sequences=True, kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        # model.add(ELU(alpha=1.0))
        model.add(Dropout(drop_out))
    model.add(LSTM(256))
    # model.add(Dense(len(mark_position), kernel_initializer='he_normal'))
    # model.add(ELU(alpha=1.0))
    model.add(mdn.MDN(OUTPUT_DIMS, N_MIXES))

    # model 학습 과정 설정하기
    model.compile(loss=mdn.get_mixture_loss_func(OUTPUT_DIMS, N_MIXES), optimizer=keras.optimizers.Adam())

    # 생성된 model 확인
    model.summary()

    return model

# =============================================================================
#
# =============================================================================

def fclone_bidirectional_RMDN(seq_length, stack, hidden_size, drop_out, N_MIXES, OUTPUT_DIMS, mark_position):
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

    # n stack RNN model
    K.clear_session()
    model = Sequential()
    # n stack
    for i in range(stack):
        model.add(Bidirectional(LSTM(hidden_size, input_shape=(seq_length, 4), return_sequences=True, kernel_initializer='he_normal')))
        model.add(BatchNormalization())
        # model.add(ELU(alpha=1.0))
        model.add(Dropout(drop_out))
    model.add(Bidirectional(LSTM(256)))
    # model.add(Dense(len(mark_position), kernel_initializer='he_normal'))
    # model.add(ELU(alpha=1.0))
    model.add(mdn.MDN(OUTPUT_DIMS, N_MIXES))

    # model 학습 과정 설정하기
    model.compile(loss=mdn.get_mixture_loss_func(OUTPUT_DIMS, N_MIXES), optimizer=keras.optimizers.Adam())

    # 생성된 model 확인
#    model.summary()

    return model

