import time
from keras.layers.recurrent import LSTM
from keras.layers.core import Activation, Dense, Dropout
from keras.models import Sequential, load_model
from data import load_data
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed
seed(1234)  # seed random numbers for Keras
from tensorflow import set_random_seed
set_random_seed(2)  # seed random numbers for Tensorflow backend

def build_model(prediction_steps):
    model = Sequential()
    layers = [1, 75, 100, prediction_steps]
    model.add(LSTM(layers[1], input_shape=(None, layers[0]), return_sequences=True))  # add first layer
    model.add(Dropout(0.2))  # add dropout for first layer
    model.add(LSTM(layers[2], return_sequences=False))  # add second layer
    model.add(Dropout(0.2))  # add dropout for second layer
    model.add(Dense(layers[3]))  # add output layer
    model.add(Activation('linear'))  # output layer with linear activation
    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print('Compilation Time : ', time.time() - start)
    return model

def run_lstm(model=None, data=None):
    global_start_time = time.time()
    epochs = 1
    ratio_of_data = 1  # ratio of data to use from 2+ million data points
    sequence_length = 10  # number of past minutes of data for model to consider
    prediction_steps = 5  # number of future minutes of data for model to predict
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

    # plot results
    try:
        test_hours_to_plot = 2
        t0 = 20  # time to start plot of predictions
        skip = 15  # skip prediction plots by specified minutes
        print('Plotting predictions...')
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(y_test[:test_hours_to_plot * 60, 0] + result_mean, label='Raw data') # plot actual test series

        # plot predicted values from t0 to t0+prediction_steps
        plt.plot(np.arange(t0 - 1, t0 + prediction_steps),
                 np.insert(predicted[t0, :], 0, y_test[t0 - 1, 0]) + result_mean,
                 color='red', label='t+{0} evolution of predictions'.format(prediction_steps))
        for i in range(t0, test_hours_to_plot * 60, skip):
            t0 += skip
            if t0 + prediction_steps > test_hours_to_plot * 60:  # check plot does not exceed boundary
                break
            plt.plot(np.arange(t0 - 1, t0 + prediction_steps),
                     np.insert(predicted[t0, :], 0, y_test[t0 - 1, 0]) + result_mean, color='red')

        # plot predicted value of t+prediction_steps as series
        plt.plot(predicted[:test_hours_to_plot * 60, prediction_steps - 1] + result_mean,
                 label='t+{0} prediction series'.format(prediction_steps))

        plt.legend(loc='lower left')
        plt.ylabel('Actual Power in kilowatt')
        plt.xlabel('Time in minutes')
        plt.title('Predictions for first {0} minutes in test set'.format(test_hours_to_plot * 60))
        plt.show()
    except Exception as e:
        print(str(e))
    print('Duration of training (s) : ', time.time() - global_start_time)

    return model, y_test, predicted

