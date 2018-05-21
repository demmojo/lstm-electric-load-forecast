import time
import numpy as np
import matplotlib.pyplot as plt


# plot results


def plot_predictions(result_mean, prediction_steps, predicted, y_test, global_start_time):
    try:
        test_hours_to_plot = 2
        t0 = 20  # time to start plot of predictions
        skip = 15  # skip prediction plots by specified minutes
        print('Plotting predictions...')
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(y_test[:test_hours_to_plot * 60, 0] + result_mean, label='Raw data')  # plot actual test series

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

    return None