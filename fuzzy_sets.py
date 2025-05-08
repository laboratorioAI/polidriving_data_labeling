import skfuzzy as fuzz
import numpy as np
import pandas as pd
from skfuzzy import control as ctrl
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

LABELING_IMPROVED = INPUT_FORMAT = 'lai'
LABELING = OUTPUT_FORMAT = 'fse'


def main(data_path):

    path = '{0}_{1}.csv'.format(data_path, INPUT_FORMAT)
    data = pd.read_csv(path, index_col=0)
    #data.drop_duplicates(keep='first', inplace=True)

    # inputs
    heart_rate = ctrl.Antecedent(np.arange(39, 202, 1), label='heart_rate')
    precipitation = ctrl.Antecedent(np.arange(-0.1, 100.2, 0.1), label='precipitation')
    visibility = ctrl.Antecedent(np.arange(-0.1, 100.2, 0.1), label='visibility')
    engine_temperature = ctrl.Antecedent(np.arange(-1, 202, 1), label='engine_temperature')
    rpm = ctrl.Antecedent(np.arange(-1, 8002, 1), label='rpm')
    design_speed = ctrl.Antecedent(np.arange(-1, 72, 1), label='design_speed')
    accidents_site = ctrl.Antecedent(np.arange(-1, 1002, 1), label='accidents_site')
    accidents_time = ctrl.Antecedent(np.arange(-1, 102, 1), label='accidents_time')
    weather = ctrl.Antecedent(np.arange(-0.1, 35.2, 0.1), label='weather')
    speed = ctrl.Antecedent(np.arange(-1, 202, 1), label='speed')
    throttle = ctrl.Antecedent(np.arange(-0.1, 100.2, 0.1), label='throttle')
    engine_load = ctrl.Antecedent(np.arange(-0.1, 100.2, 0.1), label='engine_load')
    acceleration = ctrl.Antecedent(np.arange(-0.01, 5.02, 0.01), label='acceleration')

    # membership functions
    # heart_rate['bradycardia'] = fuzz.trapmf(heart_rate.universe, [40, 44, 58, 62])
    # heart_rate['normal'] = fuzz.trapmf(heart_rate.universe, [58, 62, 98, 102])
    # heart_rate['tachycardia'] = fuzz.trapmf(heart_rate.universe, [98, 102, 196, 200])
    heart_rate['bradycardia'] = fuzz.trimf(heart_rate.universe, [39, 52, 65])
    heart_rate['sinus'] = fuzz.trimf(heart_rate.universe, [55, 80, 105])
    heart_rate['tachycardia'] = fuzz.trimf(heart_rate.universe, [95, 147, 201])

    # precipitation['light'] = fuzz.trapmf(precipitation.universe, [0.0, 0.4, 2.0, 2.4])
    # precipitation['moderate'] = fuzz.trapmf(precipitation.universe, [2.0, 2.4, 9.6, 10.0])
    # precipitation['heavy'] = fuzz.trapmf(precipitation.universe, [9.6, 10.0, 49.6, 50.0])
    # precipitation['torrential'] = fuzz.trapmf(precipitation.universe, [49.6, 50.0, 99.6, 100.0])
    precipitation['light'] = fuzz.trimf(precipitation.universe, [-0.1, 1.3, 2.8])
    precipitation['moderate'] = fuzz.trimf(precipitation.universe, [2.0, 6.2, 12])
    precipitation['heavy'] = fuzz.trimf(precipitation.universe, [8, 30.0, 55.0])
    precipitation['torrential'] = fuzz.trimf(precipitation.universe, [45.0, 74.9, 100.1])

    # visibility['bad'] = fuzz.trapmf(visibility.universe, [0.0, 0.4, 1.6, 2.0])
    # visibility['poor'] = fuzz.trapmf(visibility.universe, [1.6, 2.0, 3.6, 4.0])
    # visibility['moderate'] = fuzz.trapmf(visibility.universe, [3.6, 4.0, 9.6, 10.0])
    # visibility['good'] = fuzz.trapmf(visibility.universe, [9.6, 10.0, 39.6, 40.0])
    # visibility['excellent'] = fuzz.trapmf(visibility.universe, [39.6, 40.0, 99.6, 100.0])
    visibility['bad'] = fuzz.trimf(visibility.universe, [-0.1, 1.1, 2.4])
    visibility['poor'] = fuzz.trimf(visibility.universe, [1.6, 3.0, 4.4])
    visibility['moderate'] = fuzz.trimf(visibility.universe, [3.6, 7.0, 12.0])
    visibility['good'] = fuzz.trimf(visibility.universe, [8.0, 25.0, 45.0])
    visibility['excellent'] = fuzz.trimf(visibility.universe, [35.0, 69.9, 100.1])

    # engine_temperature['low'] = fuzz.trapmf(engine_temperature.universe, [40, 44, 80, 84])
    # engine_temperature['normal'] = fuzz.trapmf(engine_temperature.universe, [80, 84, 92, 96])
    # engine_temperature['high'] = fuzz.trapmf(engine_temperature.universe, [92, 96, 102, 106])
    # engine_temperature['overheating'] = fuzz.trapmf(engine_temperature.universe, [102, 106, 196, 200])
    engine_temperature['low'] = fuzz.trimf(engine_temperature.universe, [-1, 43, 86])
    engine_temperature['normal'] = fuzz.trimf(engine_temperature.universe, [82, 90, 98])
    engine_temperature['high'] = fuzz.trimf(engine_temperature.universe, [94, 101, 108])
    engine_temperature['overheating'] = fuzz.trimf(engine_temperature.universe, [104, 152, 201])

    # rpm['low'] = fuzz.trapmf(rpm.universe, [0, 40, 1480, 1520])
    # rpm['normal'] = fuzz.trapmf(rpm.universe, [1480, 1520, 2980, 3020])
    # rpm['high'] = fuzz.trapmf(rpm.universe, [2980, 3020, 4980, 5020])
    # rpm['very_high'] = fuzz.trapmf(rpm.universe, [4980, 5020, 7960, 8000])
    rpm['low'] = fuzz.trimf(rpm.universe, [-1, 800, 1600])
    rpm['normal'] = fuzz.trimf(rpm.universe, [1400, 2250, 3100])
    rpm['high'] = fuzz.trimf(rpm.universe, [2900, 4000, 5100])
    rpm['very_high'] = fuzz.trimf(rpm.universe, [4900, 6450, 8001])

    # design_speed['abnormal'] = fuzz.trapmf(design_speed.universe, [-70, -66, -22, -18])
    # design_speed['normal'] = fuzz.trapmf(design_speed.universe, [-22, -18, -2, 2])
    # design_speed['slight'] = fuzz.trapmf(design_speed.universe, [-2, 2, 8, 12])
    # design_speed['moderate'] = fuzz.trapmf(design_speed.universe, [8, 12, 18, 22])
    # design_speed['serious'] = fuzz.trapmf(design_speed.universe, [18, 22, 38, 42])
    # design_speed['very_serious'] = fuzz.trapmf(design_speed.universe, [38, 42, 66, 70])
    design_speed['normal'] = fuzz.trimf(design_speed.universe, [-1, 1, 2])
    design_speed['slight'] = fuzz.trimf(design_speed.universe, [1, 6, 12])
    design_speed['moderate'] = fuzz.trimf(design_speed.universe, [8, 15, 22])
    design_speed['serious'] = fuzz.trimf(design_speed.universe, [18, 30, 42])
    design_speed['very_serious'] = fuzz.trimf(design_speed.universe, [38, 54, 71])

    # accidents_site['low'] = fuzz.trapmf(accidents_site.universe, [0, 4,  6, 10])
    # accidents_site['moderate'] = fuzz.trapmf(accidents_site.universe, [6, 10, 28, 32])
    # accidents_site['high'] = fuzz.trapmf(accidents_site.universe, [28, 32, 130, 134])
    # accidents_site['very_high'] = fuzz.trapmf(accidents_site.universe, [130, 134, 996, 1000])
    accidents_site['low'] = fuzz.trimf(accidents_site.universe, [-1, 5, 10])
    accidents_site['moderate'] = fuzz.trimf(accidents_site.universe, [6, 19, 34])
    accidents_site['high'] = fuzz.trimf(accidents_site.universe, [26, 84, 140])
    accidents_site['very_high'] = fuzz.trimf(accidents_site.universe, [120, 560, 1001])

    # accidents_time['low'] = fuzz.trapmf(accidents_time.universe, [0, 0, 1, 3])
    # accidents_time['moderate'] = fuzz.trapmf(accidents_time.universe, [1, 3, 8, 12])
    # accidents_time['high'] = fuzz.trapmf(accidents_time.universe, [8, 12, 196, 200])
    accidents_time['low'] = fuzz.trimf(accidents_time.universe, [-1, 1, 3])
    accidents_time['moderate'] = fuzz.trimf(accidents_time.universe, [1, 6, 10])
    accidents_time['high'] = fuzz.trimf(accidents_time.universe, [8, 20, 32])
    accidents_time['very_high'] = fuzz.trimf(accidents_time.universe, [28, 64, 101])

    speed['normal'] = fuzz.trimf(speed.universe, [-1, 47, 95])
    speed['moderate'] = fuzz.trimf(speed.universe, [85, 100, 115])
    speed['high'] = fuzz.trimf(speed.universe, [105, 125, 145])
    speed['very_high'] = fuzz.trimf(speed.universe, [135, 168, 201])

    weather['sunny'] = fuzz.trimf(weather.universe, [0.9, 1, 1.1])
    weather['mostly_sunny'] = fuzz.trimf(weather.universe, [1.9, 2, 2.1])
    weather['partly_sunny'] = fuzz.trimf(weather.universe, [2.9, 3, 3.1])
    weather['hazy_sunshine'] = fuzz.trimf(weather.universe, [4.9, 5, 5.1])
    weather['mostly_cloudy'] = fuzz.trimf(weather.universe, [5.9, 6, 6.1])
    weather['cloudy'] = fuzz.trimf(weather.universe, [6.9, 7, 7.1])
    weather['clouds_sun'] = fuzz.trimf(weather.universe, [8.9, 9, 9.1])
    weather['fog'] = fuzz.trimf(weather.universe, [10.9, 11, 11.1])
    weather['rain'] = fuzz.trimf(weather.universe, [17.9, 18, 18.1])
    weather['partly_cloudy'] = fuzz.trimf(weather.universe, [34.9, 35, 35.1])

    throttle['low'] = fuzz.trimf(throttle.universe, [-0.1, 32.0, 65.0])
    throttle['high'] = fuzz.trimf(throttle.universe, [55.0, 72.0, 90.0])
    throttle['wide_open'] = fuzz.trimf(throttle.universe, [85.0, 92.0, 100.1])

    engine_load['low'] = fuzz.trimf(engine_load.universe, [-0.1, 37.0, 75.0])
    engine_load['high'] = fuzz.trimf(engine_load.universe, [65.0, 82.0, 100.1])

    acceleration['moderate'] = fuzz.trimf(acceleration.universe, [-0.01, 0.80, 1.60])
    acceleration['high'] = fuzz.trimf(acceleration.universe, [1.40, 2.25, 3.10])
    acceleration['extreme'] = fuzz.trimf(acceleration.universe, [2.90, 3.95, 5.01])

    # output
    penalty = ctrl.Consequent(np.arange(-0.1, 3.2, 0.1), label='penalty')
    penalty['low'] = fuzz.trimf(penalty.universe, [-0.1, 0.0, 0.2])
    penalty['medium'] = fuzz.trimf(penalty.universe, [0.1, 0.6, 1.1])
    penalty['high'] = fuzz.trimf(penalty.universe, [0.9, 1.5, 2.1])
    penalty['very_high'] = fuzz.trimf(penalty.universe, [1.9, 2.5, 3.1])

    #penalty['low'] = fuzz.trapmf(penalty.universe, [-0.1, 0.0, 0.1, 0.2])
    #penalty['medium'] = fuzz.trapmf(penalty.universe, [0.1, 0.3, 0.9, 1.1])
    #penalty['high'] = fuzz.trapmf(penalty.universe, [0.9, 1.1, 1.9, 2.1])
    #penalty['very_high'] = fuzz.trapmf(penalty.universe, [1.9, 2.1, 2.9, 3.1])

    # rules
    #precipitation
    rule_1 = ctrl.Rule(precipitation['light'], penalty['low'])
    rule_2 = ctrl.Rule(precipitation['moderate'], penalty['medium'])
    rule_3 = ctrl.Rule(precipitation['heavy'], penalty['high'])
    rule_4 = ctrl.Rule(precipitation['torrential'], penalty['very_high'])

    # visibility
    rule_5 = ctrl.Rule(visibility['bad'], penalty['very_high'])
    rule_6 = ctrl.Rule(visibility['poor'], penalty['very_high'])
    rule_7 = ctrl.Rule(visibility['moderate'], penalty['high'])
    rule_8 = ctrl.Rule(visibility['good'], penalty['medium'])
    rule_9 = ctrl.Rule(visibility['excellent'], penalty['low'])

    # engine_temperature
    rule_10 = ctrl.Rule(engine_temperature['low'], penalty['medium'])
    rule_11 = ctrl.Rule(engine_temperature['normal'], penalty['low'])
    rule_12 = ctrl.Rule(engine_temperature['high'], penalty['high'])
    rule_13 = ctrl.Rule(engine_temperature['overheating'], penalty['very_high'])

    # heart_rate
    rule_14 = ctrl.Rule(heart_rate['bradycardia'], penalty['medium'])
    rule_15 = ctrl.Rule(heart_rate['sinus'], penalty['low'])
    rule_16 = ctrl.Rule(heart_rate['tachycardia'], penalty['high'])

    # design_speed
    rule_17 = ctrl.Rule(design_speed['normal'], penalty['low'])
    rule_18 = ctrl.Rule(design_speed['slight'], penalty['low'])
    rule_19 = ctrl.Rule(design_speed['moderate'], penalty['medium'])
    rule_20 = ctrl.Rule(design_speed['serious'], penalty['high'])
    rule_21 = ctrl.Rule(design_speed['very_serious'], penalty['very_high'])

    # accidents_site
    rule_22 = ctrl.Rule(accidents_site['low'], penalty['low'])
    rule_23 = ctrl.Rule(accidents_site['moderate'], penalty['medium'])
    rule_24 = ctrl.Rule(accidents_site['high'], penalty['high'])
    rule_25 = ctrl.Rule(accidents_site['very_high'], penalty['very_high'])

    # accidents_time
    rule_26 = ctrl.Rule(accidents_time['low'], penalty['low'])
    rule_27 = ctrl.Rule(accidents_time['moderate'], penalty['medium'])
    rule_28 = ctrl.Rule(accidents_time['high'], penalty['high'])
    rule_29 = ctrl.Rule(accidents_time['very_high'], penalty['very_high'])

    # speed
    rule_30 = ctrl.Rule(speed['normal'], penalty['low'])
    rule_31 = ctrl.Rule(speed['moderate'], penalty['medium'])
    rule_32 = ctrl.Rule(speed['high'], penalty['high'])
    rule_33 = ctrl.Rule(speed['very_high'], penalty['very_high'])

    # weather
    rule_34 = ctrl.Rule(weather['sunny'], penalty['low'])
    rule_35 = ctrl.Rule(weather['mostly_sunny'], penalty['low'])
    rule_36 = ctrl.Rule(weather['partly_sunny'], penalty['low'])
    rule_37 = ctrl.Rule(weather['hazy_sunshine'], penalty['high'])
    rule_38 = ctrl.Rule(weather['mostly_cloudy'], penalty['high'])
    rule_39 = ctrl.Rule(weather['cloudy'], penalty['medium'])
    rule_40 = ctrl.Rule(weather['clouds_sun'], penalty['medium'])
    rule_41 = ctrl.Rule(weather['fog'], penalty['very_high'])
    rule_42 = ctrl.Rule(weather['rain'], penalty['very_high'])
    rule_43 = ctrl.Rule(weather['partly_cloudy'], penalty['medium'])

    # throttle
    rule_44 = ctrl.Rule(throttle['low'], penalty['low'])
    rule_45 = ctrl.Rule(throttle['high'], penalty['medium'])
    rule_46 = ctrl.Rule(throttle['wide_open'], penalty['high'])

    # engine_load
    rule_47 = ctrl.Rule(engine_load['low'], penalty['low'])
    rule_48 = ctrl.Rule(engine_load['high'], penalty['medium'])

    # acceleration
    rule_49 = ctrl.Rule(acceleration['moderate'], penalty['medium'])
    rule_50 = ctrl.Rule(acceleration['high'], penalty['high'])
    rule_51 = ctrl.Rule(acceleration['extreme'], penalty['very_high'])

    # rpm
    rule_52 = ctrl.Rule(rpm['low'], penalty['medium'])
    rule_53 = ctrl.Rule(rpm['normal'], penalty['low'])
    rule_54 = ctrl.Rule(rpm['high'], penalty['high'])
    rule_55 = ctrl.Rule(rpm['very_high'], penalty['very_high'])

    control_system_precipitation = ctrl.ControlSystem([rule_1, rule_2, rule_3, rule_4])
    control_system_visibility = ctrl.ControlSystem([rule_5, rule_6, rule_7, rule_8, rule_9])
    control_system_engine_temperature = ctrl.ControlSystem([rule_10, rule_11, rule_12, rule_13])
    control_system_heart_rate = ctrl.ControlSystem([rule_14, rule_15, rule_16])
    control_system_design_speed = ctrl.ControlSystem([rule_17, rule_18, rule_19, rule_20, rule_21])
    control_system_accidents_site = ctrl.ControlSystem([rule_22, rule_23, rule_24, rule_25])
    control_system_accidents_time = ctrl.ControlSystem([rule_26, rule_27, rule_28, rule_29])
    control_system_speed = ctrl.ControlSystem([rule_30, rule_31, rule_32, rule_33])
    control_system_weather = ctrl.ControlSystem([rule_34, rule_35, rule_36, rule_36, rule_37, rule_38, rule_39, rule_40,
                                                 rule_41, rule_42, rule_43])
    control_system_throttle = ctrl.ControlSystem([rule_44, rule_45, rule_46])
    control_system_engine_load = ctrl.ControlSystem([rule_47, rule_48])
    control_system_acceleration = ctrl.ControlSystem([rule_49, rule_50, rule_51])
    control_system_rpm = ctrl.ControlSystem([rule_52, rule_53, rule_54, rule_55])

    control_precipitation = ctrl.ControlSystemSimulation(control_system_precipitation)
    control_visibility = ctrl.ControlSystemSimulation(control_system_visibility)
    control_engine_temperature = ctrl.ControlSystemSimulation(control_system_engine_temperature)
    control_heart_rate = ctrl.ControlSystemSimulation(control_system_heart_rate)
    control_design_speed = ctrl.ControlSystemSimulation(control_system_design_speed)
    control_accidents_site = ctrl.ControlSystemSimulation(control_system_accidents_site)
    control_accidents_time = ctrl.ControlSystemSimulation(control_system_accidents_time)
    control_speed = ctrl.ControlSystemSimulation(control_system_speed)
    control_weather = ctrl.ControlSystemSimulation(control_system_weather)
    control_throttle = ctrl.ControlSystemSimulation(control_system_throttle)
    control_engine_load = ctrl.ControlSystemSimulation(control_system_engine_load)
    control_acceleration = ctrl.ControlSystemSimulation(control_system_acceleration)
    control_rpm = ctrl.ControlSystemSimulation(control_system_rpm)

    columns = ['observation_hour', 'speed', 'rpm', 'acceleration', 'throttle_position', 'engine_temperature',
               'engine_load_value', 'heart_rate', 'current_weather', 'visibility', 'precipitation',
               'accidents_onsite', 'design_speed', 'accidents_time']

    data['accidents_time_risk'] = np.nan
    data['rpm_risk'] = np.nan
    data['engine_temperature_risk'] = np.nan
    data['heart_rate_risk'] = np.nan
    data['current_weather_risk'] = np.nan
    data['visibility_risk'] = np.nan
    data['precipitation_risk'] = np.nan
    data['accidents_onsite_risk'] = np.nan
    data['design_speed_risk'] = np.nan
    data['engine_load_risk'] = np.nan
    data['acceleration_risk'] = np.nan
    data['throttle_position_risk'] = np.nan
    data['speed_risk'] = np.nan

    speed_difference = data['speed'] - data['design_speed']
    speed_difference = speed_difference.where(speed_difference >= 0, other=0)
    data['speed_difference'] = speed_difference

    for key, value in data.iterrows():
        control_design_speed.input['design_speed'] = value['speed_difference']
        control_design_speed.compute()
        data.at[key, 'design_speed_risk'] = round(control_design_speed.output.get('penalty'), 2)
        control_design_speed.reset()

        control_engine_load.input['engine_load'] = value['engine_load_value']
        control_engine_load.compute()
        data.at[key, 'engine_load_risk'] = round(control_engine_load.output.get('penalty'), 2)
        control_engine_load.reset()

        control_heart_rate.input['heart_rate'] = value['heart_rate']
        control_heart_rate.compute()
        data.at[key, 'heart_rate_risk'] = round(control_heart_rate.output.get('penalty'), 2)
        control_heart_rate.reset()

        control_weather.input['weather'] = value['current_weather']
        control_weather.compute()
        data.at[key, 'current_weather_risk'] = round(control_weather.output.get('penalty'), 2)
        control_weather.reset()

        control_visibility.input['visibility'] = value['visibility']
        control_visibility.compute()
        data.at[key, 'visibility_risk'] = round(control_visibility.output.get('penalty'), 2)
        control_visibility.reset()

        control_precipitation.input['precipitation'] = value['precipitation']
        control_precipitation.compute()
        data.at[key, 'precipitation_risk'] = round(control_precipitation.output.get('penalty'), 2)
        control_precipitation.reset()

        control_engine_temperature.input['engine_temperature'] = value['engine_temperature']
        control_engine_temperature.compute()
        data.at[key, 'engine_temperature_risk'] = round(control_engine_temperature.output.get('penalty'), 2)
        control_engine_temperature.reset()

        control_rpm.input['rpm'] = value['rpm']
        control_rpm.compute()
        data.at[key, 'rpm_risk'] = round(control_rpm.output.get('penalty'), 2)
        control_engine_temperature.reset()

        control_acceleration.input['acceleration'] = abs(value['acceleration'])
        control_acceleration.compute()
        data.at[key, 'acceleration_risk'] = round(control_acceleration.output.get('penalty'), 2)
        control_acceleration.reset()

        control_throttle.input['throttle'] = value['throttle_position']
        control_throttle.compute()
        data.at[key, 'throttle_position_risk'] = round(control_throttle.output.get('penalty'), 2)
        control_throttle.reset()

        control_speed.input['speed'] = value['speed']
        control_speed.compute()
        data.at[key, 'speed_risk'] = round(control_speed.output.get('penalty'), 2)
        control_speed.reset()

        control_accidents_time.input['accidents_time'] = value['accidents_time']
        control_accidents_time.compute()
        data.at[key, 'accidents_time_risk'] = round(control_accidents_time.output.get('penalty'), 2)
        control_accidents_time.reset()

        control_accidents_site.input['accidents_site'] = value['accidents_onsite']
        control_accidents_site.compute()
        data.at[key, 'accidents_onsite_risk'] = round(control_accidents_site.output.get('penalty'), 2)
        control_accidents_site.reset()

    data['risk_total'] = data['rpm_risk'] + data['engine_temperature_risk'] + data['heart_rate_risk'] + \
        data['current_weather_risk'] + data['visibility_risk'] + data['precipitation_risk'] + \
        data['accidents_onsite_risk'] + data['design_speed_risk'] + data['accidents_time_risk'] + \
        data['engine_load_risk'] + data['acceleration_risk'] + data['throttle_position_risk'] + \
        data['speed_risk']

    data['risk_level_fuzzy_sets'] = np.where(data['risk_total'] >= 13, 4,
                                             (np.where((data['risk_total'] < 13) * (data['risk_total'] >= 10), 3,
                                                       (np.where((data['risk_total'] < 10) * (data['risk_total'] >= 7),
                                                                 2, 1)))))

    data.drop(['rpm_risk'], axis=1, inplace=True)
    data.drop(['engine_temperature_risk'], axis=1, inplace=True)
    data.drop(['heart_rate_risk'], axis=1, inplace=True)
    data.drop(['current_weather_risk'], axis=1, inplace=True)
    data.drop(['visibility_risk'], axis=1, inplace=True)
    data.drop(['precipitation_risk'], axis=1, inplace=True)
    data.drop(['accidents_onsite_risk'], axis=1, inplace=True)
    data.drop(['design_speed_risk'], axis=1, inplace=True)
    data.drop(['accidents_time_risk'], axis=1, inplace=True)
    data.drop(['speed_risk'], axis=1, inplace=True)
    data.drop(['throttle_position_risk'], axis=1, inplace=True)
    data.drop(['acceleration_risk'], axis=1, inplace=True)
    data.drop(['engine_load_risk'], axis=1, inplace=True)
    data.drop(['speed_difference'], axis=1, inplace=True)

    path = '{0}_{1}.csv'.format(data_path, OUTPUT_FORMAT)
    data.to_csv(path)
    print('file:{} created.'.format(path))