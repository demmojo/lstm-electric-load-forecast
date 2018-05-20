from keras.models import load_model
from lstm import run_lstm

if __name__ == '__main__':
    loading_model = False
    if loading_model:
        model = load_model('LSTM_power_consumption_model.h5')
    else:
        model = None
    run_lstm(model)
