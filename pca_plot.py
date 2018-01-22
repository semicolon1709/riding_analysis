import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.decomposition import PCA


def pca_dimensionality_reduction(X_train, X_test, n_components):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    pca.explained_variance_ratio_

    pca = PCA(n_components=n_components)
    X_test_pca = pca.fit_transform(X_test)
    #print(pca.explained_variance_ratio_)
    return X_train_pca, X_test_pca

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

    df = df.loc[:, ["Timestamp_RPI", "Gyro_x", "Gyro_y", "Gyro_z", "Accel_x", "Accel_y", "Accel_z", "Motion_Label"]]
    df["Timestamp_RPI"] = df["Timestamp_RPI"].apply(lambda x: x[:-5].replace("T", " "))
    df = df.sort_values(by=['Motion_Label', "Timestamp_RPI"])

    count = 0
    seconds_per_data = 5
    for _tuple in [(7, "-1"), (8, "1"), (10, "0")]:  # "-1"下坡, "0"平坡,  "1"上坡,
        df_lable = df[df.Motion_Label == _tuple[0]]
        df_lable = df_lable.drop(df_lable.index[df_lable.shape[0] - df_lable.shape[0] % 100:])
        for i in range(0, df_lable.shape[0], seconds_per_data * 5):
            count += 1
            # normalize後，每個特徵每組(每5秒)各筆資料轉置，串接成新的1筆資料
            normal = normalize(df_lable.iloc[i:i + (seconds_per_data * 5), 1:-1], norm='l2', axis=1, return_norm=True)
            df_vector = pd.DataFrame(normal[0].reshape(1, normal[0].shape[0] * normal[0].shape[1]))
            df_vector["Motion_Label"] = _tuple[1]
            if count == 1:
                fine_df = df_vector
            else:
                fine_df = fine_df.append(df_vector, ignore_index=True)

    # 準備好訓練、測試集
    X = fine_df.iloc[:, :-1].values
    y = fine_df.loc[:, "Motion_Label"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    cov_mat = np.cov(X_train.T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    tot = sum(eigen_vals)
    # 將特徵值排序  並計算百分比
    var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
    # 加總
    cum_var_exp = np.cumsum(var_exp)
    # 製圖觀察特徵值解譯量

    plt.bar(range(1, seconds_per_data * 5 * (df.shape[1] - 2) + 1), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
    plt.step(range(1, seconds_per_data * 5 * (df.shape[1] - 2) + 1), cum_var_exp, where='mid',
             label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()

    plt.show()
    # list(zip(eigen_vals, fine_df.columns))

    # 觀察特徵值後，將特徵值使用PCA降維
    X_train_pca, X_test_pca = pca_dimensionality_reduction(X_train, X_test, n_components=40)

    # 作圖查看降維結果
    colors = ['r', 'b', 'g']
    markers = ['s', 'x', 'o']

    for l, c, m in zip(np.unique(y_train), colors, markers):
        plt.scatter(X_train_pca[y_train == l, 0],
                    X_train_pca[y_train == l, 1],
                    c=c, label=l, marker=m)

    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.tight_layout()

    plt.show()
