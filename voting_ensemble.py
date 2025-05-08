import numpy as np
import pandas as pd
import random
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from statistics import mode

INPUT_FORMAT = 'unr'
OUTPUT_FORMAT = 'res'
VOTING_FORMAT = 'vot'
TEST_FORMAT = 'tst'
VERIFIED_FORMAT = 'vld'

LOW = 1
MEDIUM = 2
HIGH = 3
VERY_HIGH = 4


def main(data_path):
    print('begin program')
    path = '{0}_{1}.csv'.format(data_path, INPUT_FORMAT)
    data = pd.read_csv(path, index_col=0)
    # performing label propagation and label spreading

    columns = ['observation_hour', 'speed', 'rpm', 'acceleration', 'throttle_position', 'engine_temperature',
               'engine_load_value', 'heart_rate', 'current_weather', 'visibility', 'precipitation', 'accidents_onsite',
               'design_speed', 'accidents_time']

    x = data[columns]

    y = data[['risk_level']]

    labeled_list = list()
    labeled_class = len(data[data['risk_level'] == VERY_HIGH])
    sub_data = data[data['risk_level'] == VERY_HIGH]
    random.seed(42)
    random_list = random.sample(sub_data.index.tolist(), labeled_class)
    labeled_list.extend(random_list)
    sub_data = data[data['risk_level'] == HIGH]
    random.seed(42)
    random_list = random.sample(sub_data.index.tolist(), labeled_class)
    labeled_list.extend(random_list)

    sub_data = data[data['risk_level'] == MEDIUM]
    random.seed(42)
    random_list = random.sample(sub_data.index.tolist(), labeled_class)
    labeled_list.extend(random_list)

    sub_data = data[data['risk_level'] == LOW]
    random.seed(42)
    random_list = random.sample(sub_data.index.tolist(), labeled_class)
    labeled_list.extend(random_list)

    data['risk_level_train'] = data['risk_level'].copy()

    for index in data.index.tolist():
        if index not in labeled_list:
            data['risk_level_train'].at[index] = -1

    lab_data = data[data['risk_level_train'] != -1]
    unl_data = data[data['risk_level_train'] == -1]

    x_lab_data = lab_data[['observation_hour', 'speed', 'rpm', 'acceleration', 'throttle_position',
                           'engine_temperature', 'engine_load_value', 'heart_rate', 'current_weather', 'visibility',
                           'precipitation', 'accidents_onsite', 'design_speed', 'accidents_time']]
    y_lab_data = lab_data[['risk_level_train']]

    x_train_lab_data, x_test_lab_data, y_train_lab_data, y_test_lab_data = train_test_split(x_lab_data, y_lab_data,
                                                                                            train_size=0.70,
                                                                                            random_state=42)

    x_unl_data = unl_data[['observation_hour', 'speed', 'rpm', 'acceleration', 'throttle_position',
                           'engine_temperature', 'engine_load_value', 'heart_rate', 'current_weather', 'visibility',
                           'precipitation', 'accidents_onsite', 'design_speed', 'accidents_time']]
    y_unl_data = unl_data[['risk_level_train']]

    x_train = pd.concat([x_unl_data, x_train_lab_data])
    y_train = pd.concat([y_unl_data, y_train_lab_data])
    x_test = x_test_lab_data
    y_test = y_test_lab_data

    # Hyperparameter tuning
    paramgrid = {'kernel': ['knn'], 'gamma': [0.1, 0.2, 0.3, 0.5, 1, 2, 3],
                 'n_neighbors': [10, 20, 30, 50, 100],
                 'max_iter': [3000, 4000, 5000]}
    grid = GridSearchCV(LabelPropagation(), param_grid=paramgrid, refit=True, verbose=10,
                        n_jobs=-1)
    grid.fit(x_train, y_train)
    print(grid.best_params_)
    print(grid.best_score_)

    xy_train = x_train.copy()

    xy_test_tmp = x_test.copy()
    xy_test_tmp['risk_level_real'] = y_test

    # LabelPropagation
    model_lbpr = LabelPropagation(kernel='knn', gamma=0.1, n_neighbors=10, max_iter=10000)
    model_lbpr.fit(x_train, y_train.squeeze())
    y_pred = model_lbpr.predict(x_test)
    xy_test_tmp['risk_level_lbpr'] = y_pred
    xy_train['risk_level_lbpr'] = model_lbpr.transduction_
    accuracy = accuracy_score(y_test, y_pred)
    print("Label Propagation")
    print("accuracy={}".format(accuracy))

    # LabelSpreading
    model_lbsp = LabelSpreading(kernel='knn', alpha=0.2, gamma=0.1, n_neighbors=15, max_iter=10000)
    model_lbsp.fit(x_train, y_train.squeeze())
    y_pred = model_lbsp.predict(x_test)
    xy_test_tmp['risk_level_lbsp'] = y_pred
    xy_train['risk_level_lbsp'] = model_lbsp.transduction_
    accuracy = accuracy_score(y_test, y_pred)
    print("Label Spreading")
    print("accuracy={}".format(accuracy))

    # Self Training (SVC)
    svm = SVC(kernel='rbf', probability=True, gamma=0.1, random_state=42)
    model_svm = SelfTrainingClassifier(svm, max_iter=None)
    model_svm.fit(x_train, y_train.squeeze())
    y_pred = model_svm.predict(x_test)
    xy_train['risk_level_svm'] = model_svm.transduction_
    accuracy = accuracy_score(y_test, y_pred)
    print("Self Training SVM")
    print("accuracy={}".format(accuracy))

    # Self Training (MLP)
    mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30), random_state=42)
    model_mlp = SelfTrainingClassifier(mlp, max_iter=None)
    model_mlp.fit(x_train, y_train.squeeze())
    y_pred = model_mlp.predict(x_test)
    xy_test_tmp['risk_level_mlp'] = y_pred
    xy_train['risk_level_mlp'] = model_mlp.transduction_
    accuracy = accuracy_score(y_test, y_pred)
    print("Self Training MLP")
    print("accuracy={}".format(accuracy))

    # Self Training (Random Forest)
    ran = RandomForestClassifier(n_estimators=50, random_state=42)
    model_ran = SelfTrainingClassifier(ran, max_iter=None)
    model_ran.fit(x_train, y_train.squeeze())
    y_pred = model_ran.predict(x_test)
    xy_test_tmp['risk_level_ran'] = y_pred
    xy_train['risk_level_ran'] = model_ran.transduction_
    accuracy = accuracy_score(y_test, y_pred)
    print("Self Training Random")
    print("accuracy={}".format(accuracy))

    # Self Training (Gradient Boosting)
    gbc = GradientBoostingClassifier(learning_rate=0.8, max_depth=30, n_estimators=100, random_state=42)
    model_gbc = SelfTrainingClassifier(gbc, max_iter=None)
    model_gbc.fit(x_train, y_train.squeeze())
    y_pred = model_gbc.predict(x_test)
    xy_test_tmp['risk_level_gbc'] = y_pred
    xy_train['risk_level_gbc'] = model_gbc.transduction_
    accuracy = accuracy_score(y_test, y_pred)
    print("Self Training Gradient")
    print(model_gbc.get_params())
    print("accuracy={}".format(accuracy))

    # Voting Classifier
    # Examples using VotingClassifier
    # model = VotingClassifier(estimators=[('lab_pro', mlp), ('lab_spr', ran)],
    #                         voting='hard')
    # model.fit(x_train, y_train)
    # y_pred = model.predict(x_test)
    # score = accuracy_score(y_test, y_pred)
    # print("Hard Voting Score: {}".format(score))

    # model = VotingClassifier(estimators=[('lab_pro', mlp), ('lab_spr', ran)],
    #                         voting='soft')
    # model.fit(x_train, y_train)
    # y_pred = model.predict(x_test)
    #score = accuracy_score(y_test, y_pred)
    # print("Soft Voting Score: {}".format(score))

    xy_train['voting'] = np.nan
    xy_test_tmp['voting'] = np.nan

    for key, value in xy_train.iterrows():
        risk_list = list()
        if value['risk_level_lbpr'] != -1:
            risk_list.append(value['risk_level_lbpr'])
        if value['risk_level_lbsp'] != -1:
            risk_list.append(value['risk_level_lbsp'])
        #if value['risk_level_svm'] != -1:
        #    risk_list.append(value['risk_level_svm'])
        if value['risk_level_mlp'] != -1:
            risk_list.append(value['risk_level_mlp'])
        if value['risk_level_ran'] != -1:
            risk_list.append(value['risk_level_ran'])
        if value['risk_level_gbc'] != -1:
            risk_list.append(value['risk_level_gbc'])
        xy_train.at[key, 'voting'] = mode(risk_list)

    for key, value in xy_test_tmp.iterrows():
        risk_list = list()
        if value['risk_level_lbpr'] != -1:
            risk_list.append(value['risk_level_lbpr'])
        if value['risk_level_lbsp'] != -1:
            risk_list.append(value['risk_level_lbsp'])
        #if value['risk_level_svm'] != -1:
        #    risk_list.append(value['risk_level_svm'])
        if value['risk_level_mlp'] != -1:
            risk_list.append(value['risk_level_mlp'])
        if value['risk_level_ran'] != -1:
            risk_list.append(value['risk_level_ran'])
        if value['risk_level_gbc'] != -1:
            risk_list.append(value['risk_level_gbc'])
        xy_test_tmp.at[key, 'voting'] = mode(risk_list)

    accuracy = accuracy_score(xy_test_tmp['risk_level_real'].squeeze(), xy_test_tmp['voting'].squeeze())
    print("Voting ensemble")
    print("accuracy={}".format(accuracy))

    path = '{0}_{1}.csv'.format(data_path, TEST_FORMAT)
    xy_test_tmp.to_csv(path)
    print('file:{} created.'.format(path))

    path = '{0}_{1}.csv'.format(data_path, VOTING_FORMAT)
    xy_train.to_csv(path)
    print('file:{} created.'.format(path))

    xy_train.drop('risk_level_lbpr', axis=1, inplace=True)
    xy_train.drop('risk_level_lbsp', axis=1, inplace=True)
    xy_train.drop('risk_level_mlp', axis=1, inplace=True)
    xy_train.drop('risk_level_ran', axis=1, inplace=True)
    xy_train.drop('risk_level_gbc', axis=1, inplace=True)
    xy_train.rename(columns={'voting': 'risk_level'}, inplace=True)

    xy_test = x_test.copy()
    xy_test['risk_level'] = y_test

    data_tmp = pd.concat([xy_train, xy_test], ignore_index=True, sort=False)

    path = '{0}_{1}.csv'.format(data_path, OUTPUT_FORMAT)
    data_tmp.to_csv(path)
    print('file:{} created.'.format(path))
