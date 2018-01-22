import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def gaussian_model(X_train, X_test, y_train, y_test):
    pred = GaussianNB().fit(X_train, y_train)
    y_train_pred = pred.predict(X_train)
    y_test_pred = pred.predict(X_test)

    pred_train = accuracy_score(y_train, y_train_pred)
    pred_test = accuracy_score(y_test, y_test_pred)
    print('GaussianNB accuracies:train: %.3f,test: %.3f' % (pred_train, pred_test))

    # 用cross_val_score評估模型
    scores = cross_val_score(estimator=pred,
                             X=X_train,
                             y=y_train,
                             cv=20,
                             scoring="accuracy",
                             n_jobs=-1)
    print('CV accuracy:%.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

    bag = BaggingClassifier(base_estimator=pred,
                            n_estimators=500,
                            max_samples=1.0,
                            max_features=1.0,
                            bootstrap=True,
                            bootstrap_features=False,
                            n_jobs=-1)

    bag = bag.fit(X_train, y_train)
    bag_y_train_pred = bag.predict(X_train)
    bag_y_test_pred = bag.predict(X_test)

    bag_train = accuracy_score(y_train, bag_y_train_pred)
    bag_test = accuracy_score(y_test, bag_y_test_pred)
    print('BaggingClassifier  accuracies: train: %.3f,test: %.3f' % (bag_train, bag_test))

    ada = AdaBoostClassifier(base_estimator=pred,
                             n_estimators=5000)
    ada = ada.fit(X_train, y_train)
    ada_y_train_pred = ada.predict(X_train)
    ada_y_test_pred = ada.predict(X_test)

    ada_train = accuracy_score(y_train, ada_y_train_pred)
    ada_test = accuracy_score(y_test, ada_y_test_pred)

    print('AdaBoostClassifier accuracies: train: %.3f,test: %.3f' % (ada_train, ada_test))


def logistic_regression_model(X_train, X_test, y_train, y_test):
    lr = LogisticRegression(C=30.0).fit(X_train, y_train)
    y_train_pred = lr.predict(X_train)
    y_test_pred = lr.predict(X_test)

    pred_train = accuracy_score(y_train, y_train_pred)
    pred_test = accuracy_score(y_test, y_test_pred)
    print('LogisticRegression: train: %.3f,test: %.3f' % (pred_train, pred_test))

    # 用cross_val_score評估模型
    scores = cross_val_score(estimator=lr,
                             X=X_train,
                             y=y_train,
                             cv=20,
                             scoring="accuracy",
                             n_jobs=-1)
    print('CV accuracy:%.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

    bag = BaggingClassifier(base_estimator=lr,
                            n_estimators=500,
                            max_samples=1.0,
                            max_features=1.0,
                            bootstrap=True,
                            bootstrap_features=False,
                            n_jobs=-1)

    bag = bag.fit(X_train, y_train)
    bag_y_train_pred = bag.predict(X_train)
    bag_y_test_pred = bag.predict(X_test)

    bag_train = accuracy_score(y_train, bag_y_train_pred)
    bag_test = accuracy_score(y_test, bag_y_test_pred)
    print('BaggingClassifier  accuracies: train: %.3f,test: %.3f' % (bag_train, bag_test))

    ada = AdaBoostClassifier(base_estimator=lr,
                             n_estimators=5000)
    ada = ada.fit(X_train, y_train)
    ada_y_train_pred = ada.predict(X_train)
    ada_y_test_pred = ada.predict(X_test)

    ada_train = accuracy_score(y_train, ada_y_train_pred)
    ada_test = accuracy_score(y_test, ada_y_test_pred)

    print('AdaBoostClassifier accuracies: train: %.3f,test: %.3f' % (ada_train, ada_test))


def svm_model(X_train, X_test, y_train, y_test):
    svm = SVC(kernel='rbf', gamma=0.5, C=20).fit(X_train, y_train)
    y_train_pred = svm.predict(X_train)
    y_test_pred = svm.predict(X_test)

    pred_train = accuracy_score(y_train, y_train_pred)
    pred_test = accuracy_score(y_test, y_test_pred)
    print('SVC:train: %.3f,test: %.3f' % (pred_train, pred_test))

    # 用cross_val_score評估模型
    scores = cross_val_score(estimator=svm,
                             X=X_train,
                             y=y_train,
                             cv=20,
                             scoring="accuracy",
                             n_jobs=-1)

    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

    bag = BaggingClassifier(base_estimator=svm,
                            n_estimators=500,
                            max_samples=1.0,
                            max_features=1.0,
                            bootstrap=True,
                            bootstrap_features=False,
                            n_jobs=-1)

    bag = bag.fit(X_train, y_train)
    bag_y_train_pred = bag.predict(X_train)
    bag_y_test_pred = bag.predict(X_test)
    bag_train = accuracy_score(y_train, bag_y_train_pred)
    bag_test = accuracy_score(y_test, bag_y_test_pred)
    print('BaggingClassifier  accuracies: train: %.3f,test: %.3f' % (bag_train, bag_test))


def random_forest_model(X_train, X_test, y_train, y_test):

    forest = RandomForestClassifier(criterion='gini',
                                    n_estimators=20,
                                    max_features='auto',
                                    n_jobs=-1)
    forest.fit(X_train, y_train)
    y_train_pred = forest.predict(X_train)
    y_test_pred = forest.predict(X_test)

    pred_train = accuracy_score(y_train, y_train_pred)
    pred_test = accuracy_score(y_test, y_test_pred)
    print('RandomForestClassifier: train: %.3f,test: %.3f' % (pred_train, pred_test))

    # 用cross_val_score評估模型
    scores = cross_val_score(estimator=forest,
                             X=X_train,
                             y=y_train,
                             cv=20,
                             scoring="accuracy",
                             n_jobs=-1)
    print('CV accuracy:  %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


def knn_model(X_train, X_test, y_train, y_test):
    for K in range(40):
        K_value = K + 1
        neigh = KNeighborsClassifier(n_neighbors=K_value, weights='uniform', algorithm='auto')
        neigh.fit(X_train, y_train)
        y_pred = neigh.predict(X_test)

        print("Accuracy is ", accuracy_score(y_test, y_pred), " for K-Value:", K_value)
    print("================================")

    scores = cross_val_score(estimator=neigh,
                             X=X_train,
                             y=y_train,
                             cv=20,
                             scoring="accuracy",
                             n_jobs=-1)

    print('CV accuracy:  %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))



if __name__ == "__main__":
    dir_path = "data/slope/"
    os.listdir(dir_path)
    count = 0
    for file in os.listdir(dir_path):
        if count == 0:
            df = pd.read_csv(dir_path + file)
        else:
            df = df.append(pd.read_csv(dir_path + file), ignore_index=True)
        count += 1

    df = df.loc[:, ["Timestamp_RPI","Gyro_x", "Gyro_y", "Gyro_z", "Accel_x", "Accel_y", "Accel_z", "Motion_Label"]]
    df["Timestamp_RPI"] = df["Timestamp_RPI"].apply(lambda x: x[:-5].replace("T", " "))
    df = df.sort_values(by=['Motion_Label', "Timestamp_RPI"])

    fine_df = pd.DataFrame([], columns=["Gyro_x", "Gyro_y", "Gyro_z", "Accel_x", "Accel_y", "Accel_z","Motion_Label"])
    seconds_per_data = 5
    for _tuple in [(7, "-1"), (8, "1"), (10, "0")]:   #"-1"下坡, "0"平坡,  "1"上坡,
        df_lable = df[df.Motion_Label == _tuple[0]]
        df_lable = df_lable.drop(df_lable.index[df_lable.shape[0]-df_lable.shape[0]%100:])
        count = 0
        for i in range(0,df_lable.shape[0],seconds_per_data*5):
            # normalize後，每個特徵每組(每秒)各筆資料算1個變異數，當成新的1筆資料
            count += 1
            normal = normalize(df_lable.iloc[i:i+seconds_per_data*5,1:-1], norm='l2', axis=1, return_norm=True)
            df_TMP = pd.DataFrame(normal[0], columns=["Gyro_x", "Gyro_y", "Gyro_z", "Accel_x", "Accel_y", "Accel_z"]).var()
            df_TMP["Motion_Label"] = _tuple[1]
            fine_df = fine_df.append(df_TMP, ignore_index=True)

    ##準備好訓練、測試集
    X = fine_df.loc[:,["Gyro_x", "Gyro_y", "Gyro_z", "Accel_x", "Accel_y", "Accel_z"]].values
    y = fine_df.loc[:,"Motion_Label"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    gaussian_model(X_train, X_test, y_train, y_test)
    print("=======================================================")
    logistic_regression_model(X_train, X_test, y_train, y_test)
    print("=======================================================")
    random_forest_model(X_train, X_test, y_train, y_test)
    print("=======================================================")
    svm_model(X_train, X_test, y_train, y_test)
    print("=======================================================")
    knn_model(X_train, X_test, y_train, y_test)