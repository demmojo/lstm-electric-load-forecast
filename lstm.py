import time
from keras.layers.recurrent import LSTM
from keras.layers.core import Activation, Dense, Dropout
from keras.models import Sequential, load_model
from data import load_data
from numpy.random import seed
seed(1234)  # seed random numbers for Keras
from tensorflow import set_random_seed
set_random_seed(2)  # seed random numbers for Tensorflow backend
from plot import plot_predictions


def build_model(prediction_steps):
    model = Sequential()
    layers = [1, 75, 100, prediction_steps]
    model.add(CuDNNLSTM(layers[1], input_shape=(None, layers[0]), return_sequences=True))  # add first layer
    model.add(Dropout(0.2))  # add dropout for first layer
    model.add(CuDNNLSTM(layers[2], return_sequences=True))  # add second layer
    model.add(Dropout(0.2))  # add dropout for second layer
    '''model.add(CuDNNLSTM(layers[2], return_sequences=True))
    model.add(Dropout(0.2))'''
    model.add(CuDNNLSTM(layers[2], return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32,activation='linear'))
    model.add(Dropout(0.2))
    #model.add(Dense(32,activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(layers[3]))  # add output layer
    model.add(Activation('linear'))  # output layer with linear activation
    start = time.time()
    #model.compile(loss="mse", optimizer="rmsprop")
    opt=optimizers.Adam(lr=0.001,decay=1e-5) 
    #opt=optimizers.SGD(lr=0.01, nesterov=True)
    #opt=optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    #opt=optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    #opt=optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    #model.compile(loss='mean_absolute_error',optimizer=opt,metrics =["accuracy"])
    model.compile(optimizer = opt, loss = root_mean_squared_error,metrics =["accuracy"])
    print('Compilation Time : ', time.time() - start)
    return model


def run_lstm(model, sequence_length, prediction_steps):
    data = None
    global_start_time = time.time()
    epochs = 1
    ratio_of_data = 1  # ratio of data to use from 2+ million data points
    path_to_dataset = 'data/household_power_consumption.txt'

    if data is None:
        print('Loading data... ')
        x_train, y_train, x_test, y_test, result_mean = load_data(path_to_dataset, sequence_length,
                                                                  prediction_steps, ratio_of_data)
    else:
        x_train, y_train, x_test, y_test = data

    print('\nData Loaded. Compiling...\n')

    if model is None:
        model = build_model(prediction_steps)
        try:
            model.fit(x_train, y_train, batch_size=128, epochs=epochs, validation_split=0.05)
            predicted = model.predict(x_test)
            # predicted = np.reshape(predicted, (predicted.size,))
            model.save('LSTM_power_consumption_model.h5')  # save LSTM model
        except KeyboardInterrupt:  # save model if training interrupted by user
            print('Duration of training (s) : ', time.time() - global_start_time)
            model.save('LSTM_power_consumption_model.h5')
            return model, y_test, 0
    else:  # previously trained mode is given
        print('Loading model...')
        predicted = model.predict(x_test)
    plot_predictions(result_mean, prediction_steps, predicted, y_test, global_start_time)

    return None

