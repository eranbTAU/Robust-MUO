import serial as ser
import os
from tqdm.auto import tqdm
import re
import matplotlib.pyplot as plt
import time
import pickle
import numpy as np


def countdown(time_factor):
    # Gives the user a few seconds to prepare for a
    # data recording session
    time.sleep(0.5)
    print(" ")
    print("The recording will begin in:")
    for i in range(0, 3):
        print(3 - i)
        time.sleep(time_factor)
    print(" ")


def data_recorder(userpref, live_data, show_data):
    ard1 = userpref[2]
    recordedData = []
    while len(recordedData) < userpref[1]:
        while True:
            if ard1.inWaiting() > 0:
                current_data = str(ard1.readline())
                break

        while current_data.find('aaaa') == (-1) and (ard1.inWaiting() > 0):
            current_data = str(ard1.readline())

        i = 0
        while len(recordedData) < userpref[1]:
            if ard1.inWaiting() > 0:
                current_data = str(ard1.readline())

                try:
                    current_data = float(re.search(r'\d+', current_data).group())
                    current_sample = current_data / 1023

                    if current_sample > 1:
                        current_sample = live_data[i]
                except type(current_data) is None or BaseException:
                    current_sample = live_data[i]
                recordedData.append(current_sample)
                i = i + 1
            if i > userpref[1]:
                i = 0
    if show_data == 1:
        print(recordedData)
    return recordedData


def training_dataset_maker(userpref, show_data):
    data_set_for_training = [data_recorder(userpref, [0] * userpref[1], show_data)]
    if show_data:
        for i in range(userpref[0] - 1):
            print(i)
            print(" ", end='\r')
            data_set_for_training.append(data_recorder(userpref, data_set_for_training[i], show_data))
    else:
        for i in tqdm(range(userpref[0])):
            data_set_for_training.append(data_recorder(userpref, data_set_for_training[i], show_data))
    return data_set_for_training


def nested_list_flipper(nested):
    if len(nested) == 0:
        print("\nThe list is empty, Unable to flip.\n")
        return nested
    return_object = []
    for i in range(len(nested[0])):
        sensor = []
        for j in range(len(nested)):
            sensor.append(nested[j][i])
        return_object.append(sensor)
    return return_object


def simple_graph_saver(data, user_pref, suffix_num, predictions, legend_num, pose, save_path, numn):
    suffix = [' - Recorded data Graph', ' - Prediction data Graph', ' - Training data Graph']
    legend = ['Sensor', 'Delta']

    color_sequence = ['red', 'blue', 'green', 'magenta', 'yellow',
                      'maroon', 'black', 'aqua', 'darkslategray', 'lime',
                      'cornflowerblue', 'indigo', 'plum', 'goldenrod', 'grey',
                      'orangered']


    plt.figure(num=1, figsize=(15,15))
    for i in range(0, user_pref[1]):
        current_sensor = legend[legend_num] + str(i + 1)
        sample = list(range(1, len(data[i]) + 1))
        voltage = [x * user_pref[3] for x in data[i]]
        # plt.plot(sample, voltage, label=current_sensor)
        plt.plot(sample, voltage, color=color_sequence[i], label=current_sensor)
    plt.xlabel("Sample [#]")
    plt.ylabel("Voltage [V]")
    if type(predictions) == list:
        for i in range(len(predictions)):
            predictions[i] = pose[predictions[i]]
    plt.title(str(pose[user_pref[5]]) + " Voltage as a function of sample\n" + str(predictions))
    plt.legend(loc='best', bbox_to_anchor=(0.673, 0.5, 0.5, 0.5),  prop={"size":20})
    t = str(time.asctime())
    file_name = str(user_pref[5]) + " " + str(pose[user_pref[5]]) + ' ' + str(re.sub('[:!@#$]', '_', t)) \
                + suffix[suffix_num] + '_' + str(numn) + '.jpeg'

    save_path = save_path + '/' + pose[user_pref[5]]
    completeName = os.path.join(save_path, file_name)
    plt.savefig(completeName, bbox_inches="tight", pad_inches=1)
    plt.close()


def file_saver(user_pref, suffix_num, data, file_type_num, pose, save_path):
    file_type = ['.pkl', '.jpeg']
    suffix = [' - Training dataset file ', ' - Graph ', ' - Live data ']
    t = str(time.asctime())
    file_name = str(user_pref[5]) + " " + str(pose[user_pref[5]]) + " " + str(re.sub('[:!@#$]', '_', t)) + \
                suffix[suffix_num] + file_type[file_type_num]
    save_path = save_path + '/' + pose[user_pref[5]]
    completeName = os.path.join(save_path, file_name)
    with open(completeName, "wb") as f:
        pickle.dump(data, f)
        del data


if __name__ == '__main__':
    # utilis
    ard = ser.Serial('COM5', 500000)
    categories = sorted(['Mug', 'Screwdriver', 'Bottle', 'Scissors', 'Plate', 'Cellphone', 'Book&Notebook',
                         'Fork', 'Hammer', 'Ruler', 'Spoon'])
    longest_category = len(max(categories)) + 3
    save_path = r'E:\Gitpro\ReaLTime\data'
    save_plot_path = r'E:\Gitpro\ReaLTime\plot\data_plot'
    numn = 5                    #[0 3 4 7 10]  #[1 2 5 6 8 9]
    user_pref = [10000, 16, ard, 5, 1, 0,
                 'name']  # 10000 [sample_num, sensor_num, arduino_path, max_voltage, number_of_sets, pose, name]

    print('\n')
    for category in categories:
        print(str(categories.index(category)) + ' = ' + str(category))
    print('-1 = to end the recording loop\n')
    user_pref[5] = int(input("\nplease select the recorded pose: "))
    while user_pref[5] != -1:
        all_samples_per_session = []
        print("\n#############   Loose hand.   #############")
        countdown(0.5)
        training_dataset_maker([500, user_pref[1], ard, 5, 1, 0, 'Eran'], 0)
        sample = []
        curr_data = [0] * user_pref[1]
        for i in range(100):
            sample.append(data_recorder(user_pref, curr_data, 0))

        means = []
        for i in range(user_pref[1]):
            sensor = []
            for cell in sample:
                sensor.append(cell[i])
            means.append(np.array(sensor).mean())

        means = np.array(means)
        print("\n#############   Grab the " + str(categories[user_pref[5]]) + ".   #############")
        countdown(0.5)
        training_dataset_maker([1, user_pref[1], ard, 5, 1, 0, 'Eran'], 1)
        print("")
        training_dataset_maker([499, user_pref[1], ard, 5, 1, 0, 'Eran'], 0)

        for _ in range(user_pref[0]):
            print(_)
            all_samples_per_session.append(data_recorder(user_pref, curr_data, 1))

        if str(input("Would you like to save the data? press 'y' if so: ")) == 'y' or \
                str(input("Would you like to save the data? press 'y' if so: ")) == 'Y':
            simple_graph_saver(nested_list_flipper(all_samples_per_session), user_pref, 0, " ", 0, categories, save_plot_path, numn)
            file_saver(user_pref, 0, (all_samples_per_session, means), 0, categories, save_path)

        print('\n')
        for category in categories:
            print(str(categories.index(category)) + ' = ' + str(category))
        print('-1 = to end the recording loop\n')
        user_pref[5] = int(input("\nplease select the recorded pose: "))
