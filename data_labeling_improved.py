import pandas as pd
import numpy as np

# SELECTION = INPUT_FORMAT = 'vld'
SELECTION = INPUT_FORMAT = 'sel'
LABELING = OUTPUT_FORMAT = 'lai'
LOW_SPEED = 50
MEDIUM_SPEED = 70
HIGH_SPEED = 90

precipitation_ranges = {'none': [0, 0.0, 0.0], 'light': [1, 0.1, 5.0], 'moderate': [2, 5.1, 15.0],
                        'heavy': [3, 15.1, 50.0], 'torrential': [4, 50.1, 100.0]}
visibility_ranges = {'none': [0, 0.0, 0.0], 'bad': [4, 0, 2.0], 'poor': [3, 2.1, 4.0], 'moderate': [2, 4.1, 10.0],
                     'good': [1, 10.1, 40.0], 'excellent': [0, 40.1, 100.0]}
weather_types = {'Sunny': [1, 1], 'Mostly sunny': [1, 2], 'Partly sunny': [1, 3], 'Hazy sunshine': [3, 5],
                 'Mostly cloudy': [3, 6], 'Cloudy': [2, 7], 'Clouds and sun': [2, 9], 'Fog': [4, 11], 'Rain': [4, 18],
                 'Partly cloudy': [2, 35]}
heart_rate_ranges = {'bradycardia': [1, 0, 59], 'sinus': [0, 60, 100], 'tachycardia_moderate': [1, 101, 140],
                     'tachycardia_severe': [2, 141, 180]}
speed_ranges = {'normal': [0, 0, 90], 'moderate': [1, 91, 120], 'high': [3, 121, 140], 'very_high': [4, 141, 180]}
engine_temperature_ranges = {'low': [1, 0, 82], 'normal': [0, 83, 94], 'high': [1, 95, 104],
                             'overheating': [2, 105, 200]}
engine_load_ranges = {'low': [0, 0.0, 70.0], 'high': [1, 70.1, 100.0]}
rpm_ranges = {'low': [1, 0, 1500], 'normal': [0, 1501, 3000], 'moderate': [1, 3001, 4000], 'very_high': [2, 4001, 6000],
              'over_revving': [3, 6001, 8000]}
throttle_position_ranges = {'low': [0, 0.0, 60.0], 'high': [1, 60.1, 85.0], 'wide_open': [2, 85.1, 100.0]}
design_speed_ranges = {'normal': [0, 0, 0], 'slight': [1, 1, 10], 'moderate': [2, 11, 20], 'serious': [3, 21, 40],
                       'very_serious': [4, 41, 100]}
accidents_onsite_ranges = {'none': [0, 0, 0], 'low': [1, 1, 8], 'moderate': [2, 9, 30], 'high': [3, 31, 132],
                           'very_high': [4, 133, 1000]}
accidents_time_ranges = {'none': [0, 0, 0], 'low': [1, 1, 4], 'moderate': [2, 5, 9], 'high': [3, 10, 30],
                         'very_high': [4, 31, 100]}
acceleration_ranges = {'extreme': [2, 3.01, 5.00], 'high': [1, 1.51, 3.00], 'moderate': [0, 0.00, 1.50]}


def assign_risk_level_single(df, ranges):
    print(df)
    levels = list()

    for (index, value) in df.items():
        flag = False
        for i, v in ranges.items():
            if value == v[1]:
                levels.append(v[0])
                flag = True
                break
        if not flag:
            print("index:{} value:{}".format(index, value))
            print(value)
    print(len(levels))
    print('end')

    return np.array(levels)


def assign_risk_level_compound(df, ranges):
    levels = list()
    for (index, value) in df.items():
        flag = False
        for i, v in ranges.items():
            if (value >= v[1]) and (value <= v[2]):
                levels.append(v[0])
                flag = True
                break
        if not flag:
            print("index:{} value:{}".format(index, value))
            print(value)
    return np.array(levels)


def main(data_path):
    path = '{0}_{1}.csv'.format(data_path, INPUT_FORMAT)
    data = pd.read_csv(path, index_col=0)
    data.drop_duplicates(keep='first', inplace=True)

    speed_difference = data['speed'] - data['design_speed']
    speed_difference = speed_difference.where(speed_difference >= 0, other=0)

    data['accidents_time_risk'] = assign_risk_level_compound(data['accidents_time'], accidents_time_ranges)
    data['rpm_risk'] = assign_risk_level_compound(data['rpm'], rpm_ranges)
    data['engine_temperature_risk'] = assign_risk_level_compound(data['engine_temperature'], engine_temperature_ranges)
    data['heart_rate_risk'] = assign_risk_level_compound(data['heart_rate'], heart_rate_ranges)
    data['weather_risk'] = assign_risk_level_single(data['current_weather'], weather_types)
    data['visibility_risk'] = assign_risk_level_compound(data['visibility'], visibility_ranges)
    data['precipitation_risk'] = assign_risk_level_compound(data['precipitation'], precipitation_ranges)
    data['accidents_onsite_risk'] = assign_risk_level_compound(data['accidents_onsite'], accidents_onsite_ranges)
    data['design_speed_risk'] = assign_risk_level_compound(speed_difference, design_speed_ranges)
    data['engine_load_risk'] = assign_risk_level_compound(data['engine_load_value'], engine_load_ranges)
    data['acceleration_risk'] = assign_risk_level_compound(data['acceleration'].abs(), acceleration_ranges)
    data['throttle_position_risk'] = assign_risk_level_compound(data['throttle_position'], throttle_position_ranges)
    data['speed_risk'] = assign_risk_level_compound(data['speed'], speed_ranges)

    data['risk_total'] = data['rpm_risk'] + data['engine_temperature_risk'] + data['heart_rate_risk'] + \
        data['weather_risk'] + data['visibility_risk'] + data['precipitation_risk'] + \
        data['accidents_onsite_risk'] + data['design_speed_risk'] + data['accidents_time_risk'] + \
        data['engine_load_risk'] + data['acceleration_risk'] + data['throttle_position_risk'] + \
        data['speed_risk']

    data['risk_level_improved'] = np.where(data['risk_total'] >= 19, 4,
                                           (np.where((data['risk_total'] <= 18) * (data['risk_total'] >= 14), 3,
                                                     (np.where((data['risk_total'] <= 13) * (data['risk_total'] >= 11),
                                                               2, 1)))))

    data.drop(['rpm_risk'], axis=1, inplace=True)
    data.drop(['engine_temperature_risk'], axis=1, inplace=True)
    data.drop(['heart_rate_risk'], axis=1, inplace=True)
    data.drop(['weather_risk'], axis=1, inplace=True)
    data.drop(['visibility_risk'], axis=1, inplace=True)
    data.drop(['precipitation_risk'], axis=1, inplace=True)
    data.drop(['accidents_onsite_risk'], axis=1, inplace=True)
    data.drop(['design_speed_risk'], axis=1, inplace=True)
    data.drop(['accidents_time_risk'], axis=1, inplace=True)
    data.drop(['risk_total'], axis=1, inplace=True)
    data.drop(['speed_risk'], axis=1, inplace=True)
    data.drop(['throttle_position_risk'], axis=1, inplace=True)
    data.drop(['acceleration_risk'], axis=1, inplace=True)
    data.drop(['engine_load_risk'], axis=1, inplace=True)

    path = '{0}_{1}.csv'.format(data_path, OUTPUT_FORMAT)
    data.to_csv(path)
    print('file:{} created.'.format(path))
