import skfuzzy as fuzz
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, confusion_matrix

# SELECTION = INPUT_FORMAT = 'res'
SELECTION = INPUT_FORMAT = 'lai'
OUTPUT_FORMAT = 'fuz'


def main(data_path):

    path = '{0}_{1}.csv'.format(data_path, INPUT_FORMAT)
    data = pd.read_csv(path, index_col=0)

    columns = ['observation_hour', 'speed', 'rpm', 'acceleration', 'throttle_position', 'engine_temperature',
               'engine_load_value', 'heart_rate', 'current_weather', 'visibility', 'precipitation',
               'accidents_onsite', 'design_speed', 'accidents_time']
    tmp = data.copy()

    for col in columns:
        data[col] = MinMaxScaler().fit_transform(data[[col]].values)

    cntr, u, _, _, _, _, _ = fuzz.cmeans(data[columns].T, 4, m=2, error=0.005, maxiter=1000, seed=42)

    y_fcm = np.argmax(u, axis=0)
    # y_manual = data['risk_level'] - 1
    y_manual = data['risk_level_improved'] - 1

    conf_matrix = confusion_matrix(y_manual, y_fcm)
    row_ind, col_ind = linear_sum_assignment(-conf_matrix)

    mapping = {col_ind[i]: row_ind[i] for i in range(len(row_ind))}
    y_fcm_aligned = np.array([mapping[label] for label in y_fcm])

    ari_score = adjusted_rand_score(y_manual, y_fcm_aligned)
    print(f"ARI Score: {ari_score:.4f}")

    # data['expecting'] = data['risk_level']
    data['risk_level_fuzz'] = y_fcm_aligned + 1
    tmp['risk_level_fuzz'] = data['risk_level_fuzz']

    path = '{0}_{1}_.csv'.format(data_path, OUTPUT_FORMAT)
    tmp.to_csv(path)
    print('file:{} created.'.format(path))

