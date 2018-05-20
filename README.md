# lstm-electric-load-forecast
The task is to predict values for a timeseries of the history of over two million minutes of the power consumption of a household. The dataset can be found in:
https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption#

I used a three layer multiple-input multiple-output LSTM recurrent neural network to predict future 5 minutes using previous 10 minutes. The code is modular so you can specify the number of minutes to consider in one step for prediction as well as the number of predictions. For e.g: Use the previous 30 minutes to predict the next 15 minutes. 
