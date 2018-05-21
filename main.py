from keras.models import load_model
from lstm import run_lstm

if __name__ == '__main__':
    loading_model = False
    if loading_model:
        model = load_model('LSTM_power_consumption_model.h5')
    else:
        model = None
    sequence_length = 10  # number of past minutes of data for model to consider
    prediction_steps = 5  # number of future minutes of data for model to predict
    run_lstm(model, sequence_length, prediction_steps)
