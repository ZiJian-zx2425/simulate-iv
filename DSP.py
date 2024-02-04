import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_packages = [
    "numpy",
    "statsmodels",
    "scikit-learn",
    "tensorflow"
]

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        install(package)

import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

import os
import tensorflow as tf


def create_toeplitz_matrix(n, r):
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = r ** abs(i - j)
    return matrix


# def calculate_f_statistic(rss0, rss, n, p):
#     """
#     计算 F-statistic。
#
#     参数:
#     rss0 -- 只包含控制变量的模型的残差平方和
#     rss -- 包含控制变量和工具变量的模型的残差平方和
#     n -- 样本量
#     p -- 工具变量的数量
#
#     返回:
#     F-statistic 值
#     """
#     return ((rss0 - rss) / p) / (rss / (n - p - 1))


beta1 = 1


def calculate_rss(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)


# def simulate_and_calculate_variances(n=2000, n_training=10000, beta1=beta1):
#     alpha_0, alpha_3, beta_1, beta_6 =0, 4, beta1, 60
#     beta3 = 0
#     beta_X_to_D = np.array([1, 1.2, 1.4, 1.6, 1.8, 1, 1.2, 1.4, 1.6, 1.8, 1, 1.2, 1.4, 1.6, 1.8, 1, 1.2, 1.4, 1.6, 1.8,1, 1.2, 1.4, 1.6, 1.8, 1, 1.2, 1.4, 1.6, 1.8, 1, 1.2, 1.4, 1.6, 1.8, 1, 1.2, 1.4, 1.6, 1.8])
#     beta_X_to_D=beta_X_to_D*15
#     beta_X_to_Y = np.array([0.8, 0.9, 1, 1.1, 1.2, 0.8, 0.9, 1, 1.1, 1.2, 0.8, 0.9, 1, 1.1, 1.2, 0.8, 0.9, 1, 1.1, 1.2, 0.8, 0.9, 1, 1.1, 1.2, 0.8, 0.9, 1, 1.1, 1.2, 0.8, 0.9, 1, 1.1, 1.2, 0.8, 0.9, 1, 1.1, 1.2])
#
#     # Generate Z
#     nc = 10  # predictor numbers
#     rc = 0.8  # covariance hyper parameter
#     cov_matrix = create_toeplitz_matrix(nc, rc)
#     Z = np.random.multivariate_normal(np.zeros(nc), cov_matrix, size=n)
#     Z_test = np.random.multivariate_normal(np.zeros(nc), cov_matrix, size=n)
#     Z_training = np.random.multivariate_normal(np.zeros(nc), cov_matrix, size=n_training)
#
#     # Generate training parameter Z
#     beta_Z = np.zeros(nc)
#     beta_Z[0:4] = 3
#     beta_Z[4:5] = 0
#     beta_Z[5:9] = 3
#     beta_Z[10:10] = 0
#     # print(beta_Z)
#
#     # Generate Test Parameter Z
#     beta_Z_training = beta_Z.copy()
#     # print(beta_Z_training)
#     random_index_1 = np.random.randint(1, 5)
#     random_index_11 = np.random.randint(1, 5)
#     random_index_2 = np.random.randint(6, 10)
#     random_index_22 = np.random.randint(6, 10)
#     beta_Z_training[random_index_1] -= 3
#     beta_Z_training[random_index_11] -= 3
#     beta_Z_training[random_index_2] -= 3
#     beta_Z_training[random_index_22] -= 3
#     # random_index_111 = np.random.randint(0, 20)
#     # random_index_1111 = np.random.randint(0, 20)
#     # random_index_222 = np.random.randint(21, 40)
#     # random_index_2222 = np.random.randint(21, 40)
#     # beta_Z_training[random_index_111] -= 0.7
#     # beta_Z_training[random_index_1111] -= 0.7
#     # beta_Z_training[random_index_222] -= 0.3
#     # beta_Z_training[random_index_2222] -= 0.3
#     # random_index_11111 = np.random.randint(0, 20)
#     # random_index_111111 = np.random.randint(0, 20)
#     # random_index_22222 = np.random.randint(21, 40)
#     # random_index_222222 = np.random.randint(21, 40)
#     # beta_Z_training[random_index_11111] -= 0.7
#     # beta_Z_training[random_index_111111] -= 0.7
#     # beta_Z_training[random_index_22222] -= 0.3
#     # beta_Z_training[random_index_222222] -= 0.3
#     # random_index_1111111 = np.random.randint(0, 20)
#     # random_index_11111111 = np.random.randint(0, 20)
#     # random_index_2222222 = np.random.randint(21, 40)
#     # random_index_22222222 = np.random.randint(21, 40)
#     # beta_Z_training[random_index_1111111] -= 0.7
#     # beta_Z_training[random_index_11111111] -= 0.7
#     # beta_Z_training[random_index_2222222] -= 0.3
#     # beta_Z_training[random_index_22222222] -= 0.3
#
#     # Generate X
#     X = np.random.multivariate_normal(np.zeros(40), np.eye(40), n)
#     X_training = np.random.multivariate_normal(np.zeros(40), np.eye(40), n_training)
#     X_test = np.random.multivariate_normal(np.zeros(40), np.eye(40), n)
#
#     # Generate C
#     C = np.random.normal(0, 3, n)
#     C_training = np.random.normal(0, 3, n_training)
#     C_test = np.random.normal(0, 3, n)
#
#     # Generate Error Term
#     u = np.random.normal(0, 1, n)
#     u_training = np.random.normal(0, 1, n_training)
#     u_test = np.random.normal(0, 1, n)
#     # print("now:--------------^,^--------------")
#     # print(alpha_0,Z[:, :2],Z[:, 3:],beta_Z[:2],beta_Z[3:],beta_X_to_D)
#     # print("now:--------------^,^--------------")
#     # Generate D
#     # D = alpha_0 + np.sin(np.dot(Z[:, :5], beta_Z[:5])) + np.abs(np.dot(Z[:, 5:], beta_Z[5:])) + np.sin(np.dot(X,beta_X_to_D)) + alpha_3 * C + u
#     # D_test = alpha_0 + np.sin(np.dot(Z_test[:, :5], beta_Z[:5])) + np.abs(np.dot(Z_test[:, 5:], beta_Z[5:])) + np.sin(np.dot(X_test, beta_X_to_D)) + alpha_3 * C_test + u_test
#     # D_training = alpha_0 + np.sin(np.dot(Z_training[:, :5], beta_Z_training[:5]) )+ np.abs(np.dot(Z_training[:, 5:],beta_Z_training[5:])) + np.sin(np.dot(X_training, beta_X_to_D)) + alpha_3 * C_training + u_training
#
#     # 假设 beta_X_to_D 的长度是 40
#     D = alpha_0 + np.sin(np.dot(Z[:, :5], beta_Z[:5])) + np.abs(np.dot(Z[:, 5:], beta_Z[5:])) + np.sin(np.dot(X[:, :20], beta_X_to_D[:20])) + np.abs(np.dot(X[:, 20:], beta_X_to_D[20:])) + alpha_3 * C + u
#     D_test = alpha_0 + np.sin(np.dot(Z_test[:, :5], beta_Z[:5])) + np.abs(np.dot(Z_test[:, 5:], beta_Z[5:])) + np.sin(np.dot(X_test[:, :20], beta_X_to_D[:20])) + np.abs(np.dot(X_test[:, 20:], beta_X_to_D[20:])) + alpha_3 * C_test + u_test
#     D_training = alpha_0 + np.sin(np.dot(Z_training[:, :5], beta_Z_training[:5])) + np.abs(np.dot(Z_training[:, 5:], beta_Z_training[5:])) + np.sin(np.dot(X_training[:, :20], beta_X_to_D[:20])) + np.abs(np.dot(X_training[:, 20:], beta_X_to_D[20:])) + alpha_3 * C_training + u_training
#
#
#     # Generate Y
#     v = np.random.normal(0, 1, n)
#     Y = beta3 + beta_1 * D + np.dot(X, beta_X_to_Y) + beta_6 * C + v
#     v_training = np.random.normal(0, 1, n_training)
#     Y_training = beta3 + beta_1 * D_training + np.dot(X_training, beta_X_to_Y) + beta_6 * C_training + v_training
#     v_test = np.random.normal(0, 1, n)
#     Y_test = beta3 + beta_1 * D_test + np.dot(X_test, beta_X_to_Y) + beta_6 * C_test + v_test
#
#     # 计算方差
#     Y_pred_intercept = np.full(n, alpha_0)  # 只有截距项的模型
#     Y_pred_X = np.sin(np.dot(X[:, :20], beta_X_to_D[:20])) + np.abs(np.dot(X[:, 20:], beta_X_to_D[20:]))  # 包含 X 的模型
#     Y_pred_XZ = alpha_0 + np.sin(np.dot(Z[:, :5], beta_Z[:5])) + np.abs(np.dot(Z[:, 5:], beta_Z[5:])) + np.sin(np.dot(X[:, :20], beta_X_to_D[:20])) + np.abs(np.dot(X[:, 20:], beta_X_to_D[20:]))  #+ alpha_3 * C  # 包含 X 和 Z 的模型
#
#     var_I_true = np.var(Y - Y_pred_intercept, ddof=1)
#     var_X_true = np.var(Y - Y_pred_X, ddof=1)
#     var_XZ_true = np.var(Y - Y_pred_XZ, ddof=1)
#     # print(var_I_true)
#     # print(var_X_true)
#     # print(var_XZ_true)
#
#     V_X_true = var_I_true - var_X_true
#     V_XZ_true = var_I_true - var_XZ_true
#     V_Z_true = V_XZ_true - V_X_true
#
#     # print("V_X:", V_X_true)
#     # print("V_XZ:", V_XZ_true)
#     # print("V_Z:", V_Z_true)
#
#     # 计算 IVStrength
#     IVStrength = V_Z_true / V_XZ_true
#
#     print("IVStrength:", IVStrength)
#     print("IVStrength ratio:", IVStrength*n)
#
#     return (Z, X, sm.add_constant(np.column_stack((Z, X))),
#             Z_training, X_training, sm.add_constant(np.column_stack((Z_training, X_training))),
#             Z_test, X_test, sm.add_constant(np.column_stack((Z_test, X_test))),
#             D, D_training, D_test, Y, Y_training, Y_test,IVStrength*n)

def simulate_and_calculate_variances(n=2000, n_training=10000, beta1=beta1):
    alpha_0, alpha_3, beta_1, beta_6 = 0, 4, beta1, 40
    beta3 = 0
    beta_X_to_D = np.array(
        [1, 1.2, 1.4, 1.6, 1.8, 1, 1.2, 1.4, 1.6, 1.8, 1, 1.2, 1.4, 1.6, 1.8, 1, 1.2, 1.4, 1.6, 1.8, 1, 1.2, 1.4, 1.6,
         1.8, 1, 1.2, 1.4, 1.6, 1.8, 1, 1.2, 1.4, 1.6, 1.8, 1, 1.2, 1.4, 1.6, 1.8])
    beta_X_to_D = beta_X_to_D * 15
    beta_X_to_Y = np.array(
        [0.8, 0.9, 1, 1.1, 1.2, 0.8, 0.9, 1, 1.1, 1.2, 0.8, 0.9, 1, 1.1, 1.2, 0.8, 0.9, 1, 1.1, 1.2, 0.8, 0.9, 1, 1.1,
         1.2, 0.8, 0.9, 1, 1.1, 1.2, 0.8, 0.9, 1, 1.1, 1.2, 0.8, 0.9, 1, 1.1, 1.2])

    # Generate Z
    nc = 10  # predictor numbers
    rc = 0.8  # covariance hyper parameter
    cov_matrix = create_toeplitz_matrix(nc, rc)
    Z = np.random.multivariate_normal(np.zeros(nc), cov_matrix, size=n)
    Z_test = np.random.multivariate_normal(np.zeros(nc), cov_matrix, size=n)
    Z_training = np.random.multivariate_normal(np.zeros(nc), cov_matrix, size=n_training)

    # Generate training parameter Z
    beta_Z = np.zeros(nc)
    beta_Z[0:4] = 1.75
    beta_Z[4:5] = 0
    beta_Z[5:9] = 1.75
    beta_Z[10:10] = 0
    # print(beta_Z)

    # Generate Test Parameter Z
    beta_Z_training = beta_Z.copy()
    # print(beta_Z_training)
    random_index_1 = np.random.randint(1, 5)
    # random_index_11 = np.random.randint(1, 5)
    random_index_2 = np.random.randint(6, 10)
    # random_index_22 = np.random.randint(6, 10)
    beta_Z_training[random_index_1] -= 3.5
    # beta_Z_training[random_index_11] -= 3
    beta_Z_training[random_index_2] -= 3.5
    # beta_Z_training[random_index_22] -= 3
    # random_index_111 = np.random.randint(0, 20)
    # random_index_1111 = np.random.randint(0, 20)
    # random_index_222 = np.random.randint(21, 40)
    # random_index_2222 = np.random.randint(21, 40)
    # beta_Z_training[random_index_111] -= 0.7
    # beta_Z_training[random_index_1111] -= 0.7
    # beta_Z_training[random_index_222] -= 0.3
    # beta_Z_training[random_index_2222] -= 0.3
    # random_index_11111 = np.random.randint(0, 20)
    # random_index_111111 = np.random.randint(0, 20)
    # random_index_22222 = np.random.randint(21, 40)
    # random_index_222222 = np.random.randint(21, 40)
    # beta_Z_training[random_index_11111] -= 0.7
    # beta_Z_training[random_index_111111] -= 0.7
    # beta_Z_training[random_index_22222] -= 0.3
    # beta_Z_training[random_index_222222] -= 0.3
    # random_index_1111111 = np.random.randint(0, 20)
    # random_index_11111111 = np.random.randint(0, 20)
    # random_index_2222222 = np.random.randint(21, 40)
    # random_index_22222222 = np.random.randint(21, 40)
    # beta_Z_training[random_index_1111111] -= 0.7
    # beta_Z_training[random_index_11111111] -= 0.7
    # beta_Z_training[random_index_2222222] -= 0.3
    # beta_Z_training[random_index_22222222] -= 0.3

    # Generate X
    X = np.random.multivariate_normal(np.zeros(40), np.eye(40), n)
    X_training = np.random.multivariate_normal(np.zeros(40), np.eye(40), n_training)
    X_test = np.random.multivariate_normal(np.zeros(40), np.eye(40), n)

    v = np.array([0, 1])

    # Generate C
    C = np.random.normal(0, 1, (n, 2))
    C_training = np.random.normal(0, 1, (n_training, 2))
    C_test = np.random.normal(0, 1, (n, 2))

    C = np.dot(C, v)
    C_training = np.dot(C_training, v)
    C_test = np.dot(C_test, v)

    # Generate Error Term
    u = np.random.normal(0, 1, n)
    u_training = np.random.normal(0, 1, n_training)
    u_test = np.random.normal(0, 1, n)
    # print("now:--------------^,^--------------")
    # print(alpha_0,Z[:, :2],Z[:, 3:],beta_Z[:2],beta_Z[3:],beta_X_to_D)
    # print("now:--------------^,^--------------")
    # Generate D
    # D = alpha_0 + np.sin(np.dot(Z[:, :5], beta_Z[:5])) + np.abs(np.dot(Z[:, 5:], beta_Z[5:])) + np.sin(np.dot(X,beta_X_to_D)) + alpha_3 * C + u
    # D_test = alpha_0 + np.sin(np.dot(Z_test[:, :5], beta_Z[:5])) + np.abs(np.dot(Z_test[:, 5:], beta_Z[5:])) + np.sin(np.dot(X_test, beta_X_to_D)) + alpha_3 * C_test + u_test
    # D_training = alpha_0 + np.sin(np.dot(Z_training[:, :5], beta_Z_training[:5]) )+ np.abs(np.dot(Z_training[:, 5:],beta_Z_training[5:])) + np.sin(np.dot(X_training, beta_X_to_D)) + alpha_3 * C_training + u_training

    # 假设 beta_X_to_D 的长度是 40
    D = alpha_0 + np.sin(np.dot(Z[:, :5], beta_Z[:5])) + np.abs(np.dot(Z[:, 5:], beta_Z[5:])) + np.sin(
        np.dot(X[:, :20], beta_X_to_D[:20])) + np.abs(np.dot(X[:, 20:], beta_X_to_D[20:])) + alpha_3 * C + u
    D_test = alpha_0 + np.sin(np.dot(Z_test[:, :5], beta_Z[:5])) + np.abs(np.dot(Z_test[:, 5:], beta_Z[5:])) + np.sin(
        np.dot(X_test[:, :20], beta_X_to_D[:20])) + np.abs(
        np.dot(X_test[:, 20:], beta_X_to_D[20:])) + alpha_3 * C_test + u_test
    D_training = alpha_0 + np.sin(np.dot(Z_training[:, :5], beta_Z_training[:5])) + np.abs(
        np.dot(Z_training[:, 5:], beta_Z_training[5:])) + np.sin(np.dot(X_training[:, :20], beta_X_to_D[:20])) + np.abs(
        np.dot(X_training[:, 20:], beta_X_to_D[20:])) + alpha_3 * C_training + u_training

    # Generate Y
    v = np.random.normal(0, 1, n)
    Y = beta3 + beta_1 * D + np.dot(X, beta_X_to_Y) + beta_6 * C + v
    v_training = np.random.normal(0, 1, n_training)
    Y_training = beta3 + beta_1 * D_training + np.dot(X_training, beta_X_to_Y) + beta_6 * C_training + v_training
    v_test = np.random.normal(0, 1, n)
    Y_test = beta3 + beta_1 * D_test + np.dot(X_test, beta_X_to_Y) + beta_6 * C_test + v_test

    # 计算方差
    Y_pred_intercept = np.full(n, alpha_0)  # 只有截距项的模型
    Y_pred_X = np.sin(np.dot(X[:, :20], beta_X_to_D[:20])) + np.abs(np.dot(X[:, 20:], beta_X_to_D[20:]))  # 包含 X 的模型
    Y_pred_XZ = alpha_0 + np.sin(np.dot(Z[:, :5], beta_Z[:5])) + np.abs(np.dot(Z[:, 5:], beta_Z[5:])) + np.sin(
        np.dot(X[:, :20], beta_X_to_D[:20])) + np.abs(
        np.dot(X[:, 20:], beta_X_to_D[20:]))  # + alpha_3 * C  # 包含 X 和 Z 的模型

    var_I_true = np.var(Y - Y_pred_intercept, ddof=1)
    var_X_true = np.var(Y - Y_pred_X, ddof=1)
    var_XZ_true = np.var(Y - Y_pred_XZ, ddof=1)
    # print(var_I_true)
    # print(var_X_true)
    # print(var_XZ_true)

    V_X_true = var_I_true - var_X_true
    V_XZ_true = var_I_true - var_XZ_true
    V_Z_true = V_XZ_true - V_X_true

    # print("V_X:", V_X_true)
    # print("V_XZ:", V_XZ_true)
    # print("V_Z:", V_Z_true)

    # 计算 IVStrength
    IVStrength = V_Z_true / V_XZ_true

    print("IVStrength:", IVStrength)
    print("IVStrength ratio:", IVStrength * n)

    return (Z, X, sm.add_constant(np.column_stack((Z, X))),
            Z_training, X_training, sm.add_constant(np.column_stack((Z_training, X_training))),
            Z_test, X_test, sm.add_constant(np.column_stack((Z_test, X_test))),
            D, D_training, D_test, Y, Y_training, Y_test, IVStrength * n)


def simulate_and_calculate_variances2(n=2000, n_training=10000, beta1=beta1):
    alpha_0, alpha_3, beta_1, beta_6 = 0, 4, beta1, 40
    beta3 = 0
    beta_X_to_D = np.array(
        [1, 1.2, 1.4, 1.6, 1.8, 1, 1.2, 1.4, 1.6, 1.8, 1, 1.2, 1.4, 1.6, 1.8, 1, 1.2, 1.4, 1.6, 1.8, 1, 1.2, 1.4, 1.6,
         1.8, 1, 1.2, 1.4, 1.6, 1.8, 1, 1.2, 1.4, 1.6, 1.8, 1, 1.2, 1.4, 1.6, 1.8])
    beta_X_to_D = beta_X_to_D * 15
    beta_X_to_Y = np.array(
        [0.8, 0.9, 1, 1.1, 1.2, 0.8, 0.9, 1, 1.1, 1.2, 0.8, 0.9, 1, 1.1, 1.2, 0.8, 0.9, 1, 1.1, 1.2, 0.8, 0.9, 1, 1.1,
         1.2, 0.8, 0.9, 1, 1.1, 1.2, 0.8, 0.9, 1, 1.1, 1.2, 0.8, 0.9, 1, 1.1, 1.2])

    # Generate Z
    nc = 10  # predictor numbers
    rc = 0.8  # covariance hyper parameter
    cov_matrix = create_toeplitz_matrix(nc, rc)
    Z = np.random.multivariate_normal(np.zeros(nc), cov_matrix, size=n)
    Z_test = np.random.multivariate_normal(np.zeros(nc), cov_matrix, size=n)
    Z_training = np.random.multivariate_normal(np.zeros(nc), cov_matrix, size=n_training)

    # Generate training parameter Z
    beta_Z = np.zeros(nc)
    beta_Z[0:4] = 1.75
    beta_Z[4:5] = 0
    beta_Z[5:9] = 1.75
    beta_Z[10:10] = 0
    # print(beta_Z)

    # Generate Test Parameter Z
    beta_Z_training = beta_Z.copy()
    # print(beta_Z_training)
    random_index_1 = np.random.randint(1, 5)
    random_index_11 = np.random.randint(1, 5)
    random_index_2 = np.random.randint(6, 10)
    random_index_22 = np.random.randint(6, 10)
    beta_Z_training[random_index_1] -= 3.5
    beta_Z_training[random_index_11] -= 3.5
    beta_Z_training[random_index_2] -= 3.5
    beta_Z_training[random_index_22] -= 3.5
    # random_index_111 = np.random.randint(0, 20)
    # random_index_1111 = np.random.randint(0, 20)
    # random_index_222 = np.random.randint(21, 40)
    # random_index_2222 = np.random.randint(21, 40)
    # beta_Z_training[random_index_111] -= 0.7
    # beta_Z_training[random_index_1111] -= 0.7
    # beta_Z_training[random_index_222] -= 0.3
    # beta_Z_training[random_index_2222] -= 0.3
    # random_index_11111 = np.random.randint(0, 20)
    # random_index_111111 = np.random.randint(0, 20)
    # random_index_22222 = np.random.randint(21, 40)
    # random_index_222222 = np.random.randint(21, 40)
    # beta_Z_training[random_index_11111] -= 0.7
    # beta_Z_training[random_index_111111] -= 0.7
    # beta_Z_training[random_index_22222] -= 0.3
    # beta_Z_training[random_index_222222] -= 0.3
    # random_index_1111111 = np.random.randint(0, 20)
    # random_index_11111111 = np.random.randint(0, 20)
    # random_index_2222222 = np.random.randint(21, 40)
    # random_index_22222222 = np.random.randint(21, 40)
    # beta_Z_training[random_index_1111111] -= 0.7
    # beta_Z_training[random_index_11111111] -= 0.7
    # beta_Z_training[random_index_2222222] -= 0.3
    # beta_Z_training[random_index_22222222] -= 0.3

    # Generate X
    X = np.random.multivariate_normal(np.zeros(40), np.eye(40), n)
    X_training = np.random.multivariate_normal(np.zeros(40), np.eye(40), n_training)
    X_test = np.random.multivariate_normal(np.zeros(40), np.eye(40), n)

    v = np.array([0, 1])

    # Generate C
    C = np.random.normal(0, 1, (n, 2))
    C_training = np.random.normal(0, 1, (n_training, 2))
    C_test = np.random.normal(0, 1, (n, 2))

    C = np.dot(C, v)
    C_training = np.dot(C_training, v)
    C_test = np.dot(C_test, v)

    # Generate Error Term
    u = np.random.normal(0, 1, n)
    u_training = np.random.normal(0, 1, n_training)
    u_test = np.random.normal(0, 1, n)
    # print("now:--------------^,^--------------")
    # print(alpha_0,Z[:, :2],Z[:, 3:],beta_Z[:2],beta_Z[3:],beta_X_to_D)
    # print("now:--------------^,^--------------")
    # Generate D
    # D = alpha_0 + np.sin(np.dot(Z[:, :5], beta_Z[:5])) + np.abs(np.dot(Z[:, 5:], beta_Z[5:])) + np.sin(np.dot(X,beta_X_to_D)) + alpha_3 * C + u
    # D_test = alpha_0 + np.sin(np.dot(Z_test[:, :5], beta_Z[:5])) + np.abs(np.dot(Z_test[:, 5:], beta_Z[5:])) + np.sin(np.dot(X_test, beta_X_to_D)) + alpha_3 * C_test + u_test
    # D_training = alpha_0 + np.sin(np.dot(Z_training[:, :5], beta_Z_training[:5]) )+ np.abs(np.dot(Z_training[:, 5:],beta_Z_training[5:])) + np.sin(np.dot(X_training, beta_X_to_D)) + alpha_3 * C_training + u_training

    # 假设 beta_X_to_D 的长度是 40
    D = alpha_0 + np.sin(np.dot(Z[:, :5], beta_Z[:5])) + np.abs(np.dot(Z[:, 5:], beta_Z[5:])) + np.sin(
        np.dot(X[:, :20], beta_X_to_D[:20])) + np.abs(np.dot(X[:, 20:], beta_X_to_D[20:])) + alpha_3 * C + u
    D_test = alpha_0 + np.sin(np.dot(Z_test[:, :5], beta_Z[:5])) + np.abs(np.dot(Z_test[:, 5:], beta_Z[5:])) + np.sin(
        np.dot(X_test[:, :20], beta_X_to_D[:20])) + np.abs(
        np.dot(X_test[:, 20:], beta_X_to_D[20:])) + alpha_3 * C_test + u_test
    D_training = alpha_0 + np.sin(np.dot(Z_training[:, :5], beta_Z_training[:5])) + np.abs(
        np.dot(Z_training[:, 5:], beta_Z_training[5:])) + np.sin(np.dot(X_training[:, :20], beta_X_to_D[:20])) + np.abs(
        np.dot(X_training[:, 20:], beta_X_to_D[20:])) + alpha_3 * C_training + u_training

    # Generate Y
    v = np.random.normal(0, 1, n)
    Y = beta3 + beta_1 * D + np.dot(X, beta_X_to_Y) + beta_6 * C + v
    v_training = np.random.normal(0, 1, n_training)
    Y_training = beta3 + beta_1 * D_training + np.dot(X_training, beta_X_to_Y) + beta_6 * C_training + v_training
    v_test = np.random.normal(0, 1, n)
    Y_test = beta3 + beta_1 * D_test + np.dot(X_test, beta_X_to_Y) + beta_6 * C_test + v_test

    # 计算方差
    Y_pred_intercept = np.full(n, alpha_0)  # 只有截距项的模型
    Y_pred_X = np.sin(np.dot(X[:, :20], beta_X_to_D[:20])) + np.abs(np.dot(X[:, 20:], beta_X_to_D[20:]))  # 包含 X 的模型
    Y_pred_XZ = alpha_0 + np.sin(np.dot(Z[:, :5], beta_Z[:5])) + np.abs(np.dot(Z[:, 5:], beta_Z[5:])) + np.sin(
        np.dot(X[:, :20], beta_X_to_D[:20])) + np.abs(
        np.dot(X[:, 20:], beta_X_to_D[20:]))  # + alpha_3 * C  # 包含 X 和 Z 的模型

    var_I_true = np.var(Y - Y_pred_intercept, ddof=1)
    var_X_true = np.var(Y - Y_pred_X, ddof=1)
    var_XZ_true = np.var(Y - Y_pred_XZ, ddof=1)
    # print(var_I_true)
    # print(var_X_true)
    # print(var_XZ_true)

    V_X_true = var_I_true - var_X_true
    V_XZ_true = var_I_true - var_XZ_true
    V_Z_true = V_XZ_true - V_X_true

    # print("V_X:", V_X_true)
    # print("V_XZ:", V_XZ_true)
    # print("V_Z:", V_Z_true)

    # 计算 IVStrength
    IVStrength = V_Z_true / V_XZ_true

    print("IVStrength:", IVStrength)
    print("IVStrength ratio:", IVStrength * n)

    return (Z, X, sm.add_constant(np.column_stack((Z, X))),
            Z_training, X_training, sm.add_constant(np.column_stack((Z_training, X_training))),
            Z_test, X_test, sm.add_constant(np.column_stack((Z_test, X_test))),
            D, D_training, D_test, Y, Y_training, Y_test, IVStrength * n)


def simulate_and_calculate_variances3(n=2000, n_training=10000, beta1=beta1):
    alpha_0, alpha_3, beta_1, beta_6 = 0, 4, beta1, 40
    beta3 = 0
    beta_X_to_D = np.array(
        [1, 1.2, 1.4, 1.6, 1.8, 1, 1.2, 1.4, 1.6, 1.8, 1, 1.2, 1.4, 1.6, 1.8, 1, 1.2, 1.4, 1.6, 1.8, 1, 1.2, 1.4, 1.6,
         1.8, 1, 1.2, 1.4, 1.6, 1.8, 1, 1.2, 1.4, 1.6, 1.8, 1, 1.2, 1.4, 1.6, 1.8])
    beta_X_to_D = beta_X_to_D * 15
    beta_X_to_Y = np.array(
        [0.8, 0.9, 1, 1.1, 1.2, 0.8, 0.9, 1, 1.1, 1.2, 0.8, 0.9, 1, 1.1, 1.2, 0.8, 0.9, 1, 1.1, 1.2, 0.8, 0.9, 1, 1.1,
         1.2, 0.8, 0.9, 1, 1.1, 1.2, 0.8, 0.9, 1, 1.1, 1.2, 0.8, 0.9, 1, 1.1, 1.2])

    # Generate Z
    nc = 10  # predictor numbers
    rc = 0.8  # covariance hyper parameter
    cov_matrix = create_toeplitz_matrix(nc, rc)
    Z = np.random.multivariate_normal(np.zeros(nc), cov_matrix, size=n)
    Z_test = np.random.multivariate_normal(np.zeros(nc), cov_matrix, size=n)
    Z_training = np.random.multivariate_normal(np.zeros(nc), cov_matrix, size=n_training)

    # Generate training parameter Z
    beta_Z = np.zeros(nc)
    beta_Z[0:4] = 1.75
    beta_Z[4:5] = 0
    beta_Z[5:9] = 1.75
    beta_Z[10:10] = 0
    # print(beta_Z)

    # Generate Test Parameter Z
    beta_Z_training = beta_Z.copy()
    # print(beta_Z_training)
    random_index_1 = np.random.randint(1, 5)
    random_index_11 = np.random.randint(1, 5)
    random_index_2 = np.random.randint(6, 10)
    random_index_22 = np.random.randint(6, 10)
    beta_Z_training[random_index_1] -= 3.5
    beta_Z_training[random_index_11] -= 3.5
    beta_Z_training[random_index_2] -= 3.5
    beta_Z_training[random_index_22] -= 3.5
    random_index_111 = np.random.randint(1, 5)
    # random_index_1111 = np.random.randint(0, 20)
    random_index_222 = np.random.randint(6, 10)
    # random_index_2222 = np.random.randint(21, 40)
    beta_Z_training[random_index_111] -= 3.5
    # beta_Z_training[random_index_1111] -= 0.7
    beta_Z_training[random_index_222] -= 3.5
    # beta_Z_training[random_index_2222] -= 0.3
    # random_index_11111 = np.random.randint(0, 20)
    # random_index_111111 = np.random.randint(0, 20)
    # random_index_22222 = np.random.randint(21, 40)
    # random_index_222222 = np.random.randint(21, 40)
    # beta_Z_training[random_index_11111] -= 0.7
    # beta_Z_training[random_index_111111] -= 0.7
    # beta_Z_training[random_index_22222] -= 0.3
    # beta_Z_training[random_index_222222] -= 0.3
    # random_index_1111111 = np.random.randint(0, 20)
    # random_index_11111111 = np.random.randint(0, 20)
    # random_index_2222222 = np.random.randint(21, 40)
    # random_index_22222222 = np.random.randint(21, 40)
    # beta_Z_training[random_index_1111111] -= 0.7
    # beta_Z_training[random_index_11111111] -= 0.7
    # beta_Z_training[random_index_2222222] -= 0.3
    # beta_Z_training[random_index_22222222] -= 0.3

    # Generate X
    X = np.random.multivariate_normal(np.zeros(40), np.eye(40), n)
    X_training = np.random.multivariate_normal(np.zeros(40), np.eye(40), n_training)
    X_test = np.random.multivariate_normal(np.zeros(40), np.eye(40), n)

    v = np.array([0, 1])

    # Generate C
    C = np.random.normal(0, 1, (n, 2))
    C_training = np.random.normal(0, 1, (n_training, 2))
    C_test = np.random.normal(0, 1, (n, 2))

    C = np.dot(C, v)
    C_training = np.dot(C_training, v)
    C_test = np.dot(C_test, v)

    # Generate Error Term
    u = np.random.normal(0, 1, n)
    u_training = np.random.normal(0, 1, n_training)
    u_test = np.random.normal(0, 1, n)
    # print("now:--------------^,^--------------")
    # print(alpha_0,Z[:, :2],Z[:, 3:],beta_Z[:2],beta_Z[3:],beta_X_to_D)
    # print("now:--------------^,^--------------")
    # Generate D
    # D = alpha_0 + np.sin(np.dot(Z[:, :5], beta_Z[:5])) + np.abs(np.dot(Z[:, 5:], beta_Z[5:])) + np.sin(np.dot(X,beta_X_to_D)) + alpha_3 * C + u
    # D_test = alpha_0 + np.sin(np.dot(Z_test[:, :5], beta_Z[:5])) + np.abs(np.dot(Z_test[:, 5:], beta_Z[5:])) + np.sin(np.dot(X_test, beta_X_to_D)) + alpha_3 * C_test + u_test
    # D_training = alpha_0 + np.sin(np.dot(Z_training[:, :5], beta_Z_training[:5]) )+ np.abs(np.dot(Z_training[:, 5:],beta_Z_training[5:])) + np.sin(np.dot(X_training, beta_X_to_D)) + alpha_3 * C_training + u_training

    # 假设 beta_X_to_D 的长度是 40
    D = alpha_0 + np.sin(np.dot(Z[:, :5], beta_Z[:5])) + np.abs(np.dot(Z[:, 5:], beta_Z[5:])) + np.sin(
        np.dot(X[:, :20], beta_X_to_D[:20])) + np.abs(np.dot(X[:, 20:], beta_X_to_D[20:])) + alpha_3 * C + u
    D_test = alpha_0 + np.sin(np.dot(Z_test[:, :5], beta_Z[:5])) + np.abs(np.dot(Z_test[:, 5:], beta_Z[5:])) + np.sin(
        np.dot(X_test[:, :20], beta_X_to_D[:20])) + np.abs(
        np.dot(X_test[:, 20:], beta_X_to_D[20:])) + alpha_3 * C_test + u_test
    D_training = alpha_0 + np.sin(np.dot(Z_training[:, :5], beta_Z_training[:5])) + np.abs(
        np.dot(Z_training[:, 5:], beta_Z_training[5:])) + np.sin(np.dot(X_training[:, :20], beta_X_to_D[:20])) + np.abs(
        np.dot(X_training[:, 20:], beta_X_to_D[20:])) + alpha_3 * C_training + u_training

    # Generate Y
    v = np.random.normal(0, 1, n)
    Y = beta3 + beta_1 * D + np.dot(X, beta_X_to_Y) + beta_6 * C + v
    v_training = np.random.normal(0, 1, n_training)
    Y_training = beta3 + beta_1 * D_training + np.dot(X_training, beta_X_to_Y) + beta_6 * C_training + v_training
    v_test = np.random.normal(0, 1, n)
    Y_test = beta3 + beta_1 * D_test + np.dot(X_test, beta_X_to_Y) + beta_6 * C_test + v_test

    # 计算方差
    Y_pred_intercept = np.full(n, alpha_0)  # 只有截距项的模型
    Y_pred_X = np.sin(np.dot(X[:, :20], beta_X_to_D[:20])) + np.abs(np.dot(X[:, 20:], beta_X_to_D[20:]))  # 包含 X 的模型
    Y_pred_XZ = alpha_0 + np.sin(np.dot(Z[:, :5], beta_Z[:5])) + np.abs(np.dot(Z[:, 5:], beta_Z[5:])) + np.sin(
        np.dot(X[:, :20], beta_X_to_D[:20])) + np.abs(
        np.dot(X[:, 20:], beta_X_to_D[20:]))  # + alpha_3 * C  # 包含 X 和 Z 的模型

    var_I_true = np.var(Y - Y_pred_intercept, ddof=1)
    var_X_true = np.var(Y - Y_pred_X, ddof=1)
    var_XZ_true = np.var(Y - Y_pred_XZ, ddof=1)
    # print(var_I_true)
    # print(var_X_true)
    # print(var_XZ_true)

    V_X_true = var_I_true - var_X_true
    V_XZ_true = var_I_true - var_XZ_true
    V_Z_true = V_XZ_true - V_X_true

    # print("V_X:", V_X_true)
    # print("V_XZ:", V_XZ_true)
    # print("V_Z:", V_Z_true)

    # 计算 IVStrength
    IVStrength = V_Z_true / V_XZ_true

    print("IVStrength:", IVStrength)
    print("IVStrength ratio:", IVStrength * n)

    return (Z, X, sm.add_constant(np.column_stack((Z, X))),
            Z_training, X_training, sm.add_constant(np.column_stack((Z_training, X_training))),
            Z_test, X_test, sm.add_constant(np.column_stack((Z_test, X_test))),
            D, D_training, D_test, Y, Y_training, Y_test, IVStrength * n)


# # 调用函数并打印结果
# V_X, V_XZ, V_Z, IVStrength = simulate_and_calculate_variances()
# print("V_X:", V_X)
# print("V_XZ:", V_XZ)
# print("V_Z:", V_Z)
# print("IVStrength:", IVStrength)


def simulate_and_calculate_variances4(n=2000, n_training=10000, beta1=beta1):
    alpha_0, alpha_3, beta_1, beta_6 = 0, 4, beta1, 40
    beta3 = 0
    beta_X_to_D = np.array(
        [1, 1.2, 1.4, 1.6, 1.8, 1, 1.2, 1.4, 1.6, 1.8, 1, 1.2, 1.4, 1.6, 1.8, 1, 1.2, 1.4, 1.6, 1.8, 1, 1.2, 1.4, 1.6,
         1.8, 1, 1.2, 1.4, 1.6, 1.8, 1, 1.2, 1.4, 1.6, 1.8, 1, 1.2, 1.4, 1.6, 1.8])
    beta_X_to_D = beta_X_to_D * 15
    beta_X_to_Y = np.array(
        [0.8, 0.9, 1, 1.1, 1.2, 0.8, 0.9, 1, 1.1, 1.2, 0.8, 0.9, 1, 1.1, 1.2, 0.8, 0.9, 1, 1.1, 1.2, 0.8, 0.9, 1, 1.1,
         1.2, 0.8, 0.9, 1, 1.1, 1.2, 0.8, 0.9, 1, 1.1, 1.2, 0.8, 0.9, 1, 1.1, 1.2])

    # Generate Z
    nc = 10  # predictor numbers
    rc = 0.8  # covariance hyper parameter
    cov_matrix = create_toeplitz_matrix(nc, rc)
    Z = np.random.multivariate_normal(np.zeros(nc), cov_matrix, size=n)
    Z_test = np.random.multivariate_normal(np.zeros(nc), cov_matrix, size=n)
    Z_training = np.random.multivariate_normal(np.zeros(nc), cov_matrix, size=n_training)

    # Generate training parameter Z
    beta_Z = np.zeros(nc)
    beta_Z[0:4] = 1.75
    beta_Z[4:5] = 0
    beta_Z[5:9] = 1.75
    beta_Z[10:10] = 0
    # print(beta_Z)

    # Generate Test Parameter Z
    beta_Z_training = beta_Z.copy()
    # print(beta_Z_training)
    random_index_1 = np.random.randint(1, 5)
    random_index_11 = np.random.randint(1, 5)
    random_index_2 = np.random.randint(6, 10)
    random_index_22 = np.random.randint(6, 10)
    beta_Z_training[random_index_1] -= 3.5
    beta_Z_training[random_index_11] -= 3.5
    beta_Z_training[random_index_2] -= 3.5
    beta_Z_training[random_index_22] -= 3.5
    random_index_111 = np.random.randint(1, 5)
    random_index_1111 = np.random.randint(1, 5)
    random_index_222 = np.random.randint(6, 10)
    random_index_2222 = np.random.randint(6, 10)
    beta_Z_training[random_index_111] -= 3.5
    beta_Z_training[random_index_1111] -= 3.5
    beta_Z_training[random_index_222] -= 3.5
    beta_Z_training[random_index_2222] -= 3.5
    # random_index_11111 = np.random.randint(0, 20)
    # random_index_111111 = np.random.randint(0, 20)
    # random_index_22222 = np.random.randint(21, 40)
    # random_index_222222 = np.random.randint(21, 40)
    # beta_Z_training[random_index_11111] -= 0.7
    # beta_Z_training[random_index_111111] -= 0.7
    # beta_Z_training[random_index_22222] -= 0.3
    # beta_Z_training[random_index_222222] -= 0.3
    # random_index_1111111 = np.random.randint(0, 20)
    # random_index_11111111 = np.random.randint(0, 20)
    # random_index_2222222 = np.random.randint(21, 40)
    # random_index_22222222 = np.random.randint(21, 40)
    # beta_Z_training[random_index_1111111] -= 0.7
    # beta_Z_training[random_index_11111111] -= 0.7
    # beta_Z_training[random_index_2222222] -= 0.3
    # beta_Z_training[random_index_22222222] -= 0.3

    # Generate X
    X = np.random.multivariate_normal(np.zeros(40), np.eye(40), n)
    X_training = np.random.multivariate_normal(np.zeros(40), np.eye(40), n_training)
    X_test = np.random.multivariate_normal(np.zeros(40), np.eye(40), n)

    v = np.array([0, 1])

    # Generate C
    C = np.random.normal(0, 1, (n, 2))
    C_training = np.random.normal(0, 1, (n_training, 2))
    C_test = np.random.normal(0, 1, (n, 2))

    C = np.dot(C, v)
    C_training = np.dot(C_training, v)
    C_test = np.dot(C_test, v)

    # Generate Error Term
    u = np.random.normal(0, 1, n)
    u_training = np.random.normal(0, 1, n_training)
    u_test = np.random.normal(0, 1, n)
    # print("now:--------------^,^--------------")
    # print(alpha_0,Z[:, :2],Z[:, 3:],beta_Z[:2],beta_Z[3:],beta_X_to_D)
    # print("now:--------------^,^--------------")
    # Generate D
    # D = alpha_0 + np.sin(np.dot(Z[:, :5], beta_Z[:5])) + np.abs(np.dot(Z[:, 5:], beta_Z[5:])) + np.sin(np.dot(X,beta_X_to_D)) + alpha_3 * C + u
    # D_test = alpha_0 + np.sin(np.dot(Z_test[:, :5], beta_Z[:5])) + np.abs(np.dot(Z_test[:, 5:], beta_Z[5:])) + np.sin(np.dot(X_test, beta_X_to_D)) + alpha_3 * C_test + u_test
    # D_training = alpha_0 + np.sin(np.dot(Z_training[:, :5], beta_Z_training[:5]) )+ np.abs(np.dot(Z_training[:, 5:],beta_Z_training[5:])) + np.sin(np.dot(X_training, beta_X_to_D)) + alpha_3 * C_training + u_training

    # 假设 beta_X_to_D 的长度是 40
    D = alpha_0 + np.sin(np.dot(Z[:, :5], beta_Z[:5])) + np.abs(np.dot(Z[:, 5:], beta_Z[5:])) + np.sin(
        np.dot(X[:, :20], beta_X_to_D[:20])) + np.abs(np.dot(X[:, 20:], beta_X_to_D[20:])) + alpha_3 * C + u
    D_test = alpha_0 + np.sin(np.dot(Z_test[:, :5], beta_Z[:5])) + np.abs(np.dot(Z_test[:, 5:], beta_Z[5:])) + np.sin(
        np.dot(X_test[:, :20], beta_X_to_D[:20])) + np.abs(
        np.dot(X_test[:, 20:], beta_X_to_D[20:])) + alpha_3 * C_test + u_test
    D_training = alpha_0 + np.sin(np.dot(Z_training[:, :5], beta_Z_training[:5])) + np.abs(
        np.dot(Z_training[:, 5:], beta_Z_training[5:])) + np.sin(np.dot(X_training[:, :20], beta_X_to_D[:20])) + np.abs(
        np.dot(X_training[:, 20:], beta_X_to_D[20:])) + alpha_3 * C_training + u_training

    # Generate Y
    v = np.random.normal(0, 1, n)
    Y = beta3 + beta_1 * D + np.dot(X, beta_X_to_Y) + beta_6 * C + v
    v_training = np.random.normal(0, 1, n_training)
    Y_training = beta3 + beta_1 * D_training + np.dot(X_training, beta_X_to_Y) + beta_6 * C_training + v_training
    v_test = np.random.normal(0, 1, n)
    Y_test = beta3 + beta_1 * D_test + np.dot(X_test, beta_X_to_Y) + beta_6 * C_test + v_test

    # 计算方差
    Y_pred_intercept = np.full(n, alpha_0)  # 只有截距项的模型
    Y_pred_X = np.sin(np.dot(X[:, :20], beta_X_to_D[:20])) + np.abs(np.dot(X[:, 20:], beta_X_to_D[20:]))  # 包含 X 的模型
    Y_pred_XZ = alpha_0 + np.sin(np.dot(Z[:, :5], beta_Z[:5])) + np.abs(np.dot(Z[:, 5:], beta_Z[5:])) + np.sin(
        np.dot(X[:, :20], beta_X_to_D[:20])) + np.abs(
        np.dot(X[:, 20:], beta_X_to_D[20:]))  # + alpha_3 * C  # 包含 X 和 Z 的模型

    var_I_true = np.var(Y - Y_pred_intercept, ddof=1)
    var_X_true = np.var(Y - Y_pred_X, ddof=1)
    var_XZ_true = np.var(Y - Y_pred_XZ, ddof=1)
    # print(var_I_true)
    # print(var_X_true)
    # print(var_XZ_true)

    V_X_true = var_I_true - var_X_true
    V_XZ_true = var_I_true - var_XZ_true
    V_Z_true = V_XZ_true - V_X_true

    # print("V_X:", V_X_true)
    # print("V_XZ:", V_XZ_true)
    # print("V_Z:", V_Z_true)

    # 计算 IVStrength
    IVStrength = V_Z_true / V_XZ_true

    print("IVStrength:", IVStrength)
    print("IVStrength ratio:", IVStrength * n)

    return (Z, X, sm.add_constant(np.column_stack((Z, X))),
            Z_training, X_training, sm.add_constant(np.column_stack((Z_training, X_training))),
            Z_test, X_test, sm.add_constant(np.column_stack((Z_test, X_test))),
            D, D_training, D_test, Y, Y_training, Y_test, IVStrength * n)


# # 调用函数并打印结果
# V_X, V_XZ, V_Z, IVStrength = simulate_and_calculate_variances()
# print("V_X:", V_X)
# print("V_XZ:", V_XZ)
# print("V_Z:", V_Z)
# print("IVStrength:", IVStrength)


def simulate_and_calculate_variances5(n=2000, n_training=10000, beta1=beta1):
    alpha_0, alpha_3, beta_1, beta_6 = 0, 4, beta1, 40
    beta3 = 0
    beta_X_to_D = np.array(
        [1, 1.2, 1.4, 1.6, 1.8, 1, 1.2, 1.4, 1.6, 1.8, 1, 1.2, 1.4, 1.6, 1.8, 1, 1.2, 1.4, 1.6, 1.8, 1, 1.2, 1.4, 1.6,
         1.8, 1, 1.2, 1.4, 1.6, 1.8, 1, 1.2, 1.4, 1.6, 1.8, 1, 1.2, 1.4, 1.6, 1.8])
    beta_X_to_D = beta_X_to_D * 15
    beta_X_to_Y = np.array(
        [0.8, 0.9, 1, 1.1, 1.2, 0.8, 0.9, 1, 1.1, 1.2, 0.8, 0.9, 1, 1.1, 1.2, 0.8, 0.9, 1, 1.1, 1.2, 0.8, 0.9, 1, 1.1,
         1.2, 0.8, 0.9, 1, 1.1, 1.2, 0.8, 0.9, 1, 1.1, 1.2, 0.8, 0.9, 1, 1.1, 1.2])

    # Generate Z
    nc = 10  # predictor numbers
    rc = 0.8  # covariance hyper parameter
    cov_matrix = create_toeplitz_matrix(nc, rc)
    Z = np.random.multivariate_normal(np.zeros(nc), cov_matrix, size=n)
    Z_test = np.random.multivariate_normal(np.zeros(nc), cov_matrix, size=n)
    Z_training = np.random.multivariate_normal(np.zeros(nc), cov_matrix, size=n_training)

    # Generate training parameter Z
    beta_Z = np.zeros(nc)
    beta_Z[0:4] = 1.75
    beta_Z[4:5] = 0
    beta_Z[5:9] = 1.75
    beta_Z[10:10] = 0
    # print(beta_Z)

    # Generate Test Parameter Z
    beta_Z_training = beta_Z.copy()
    # print(beta_Z_training)
    random_index_1 = np.random.randint(1, 5)
    random_index_11 = np.random.randint(1, 5)
    random_index_2 = np.random.randint(6, 10)
    random_index_22 = np.random.randint(6, 10)
    beta_Z_training[random_index_1] -= 3.5
    beta_Z_training[random_index_11] -= 3.5
    beta_Z_training[random_index_2] -= 3.5
    beta_Z_training[random_index_22] -= 3.5
    random_index_111 = np.random.randint(1, 5)
    random_index_1111 = np.random.randint(1, 5)
    random_index_222 = np.random.randint(6, 10)
    random_index_2222 = np.random.randint(6, 10)
    beta_Z_training[random_index_111] -= 3.5
    beta_Z_training[random_index_1111] -= 3.5
    beta_Z_training[random_index_222] -= 3.5
    beta_Z_training[random_index_2222] -= 3.5
    random_index_11111 = np.random.randint(1, 5)
    random_index_22222 = np.random.randint(6, 10)
    beta_Z_training[random_index_11111] -= 3.5
    beta_Z_training[random_index_22222] -= 3.5
    # random_index_1111111 = np.random.randint(0, 20)
    # random_index_11111111 = np.random.randint(0, 20)
    # random_index_2222222 = np.random.randint(21, 40)
    # random_index_22222222 = np.random.randint(21, 40)
    # beta_Z_training[random_index_1111111] -= 0.7
    # beta_Z_training[random_index_11111111] -= 0.7
    # beta_Z_training[random_index_2222222] -= 0.3
    # beta_Z_training[random_index_22222222] -= 0.3

    # Generate X
    X = np.random.multivariate_normal(np.zeros(40), np.eye(40), n)
    X_training = np.random.multivariate_normal(np.zeros(40), np.eye(40), n_training)
    X_test = np.random.multivariate_normal(np.zeros(40), np.eye(40), n)

    v = np.array([0, 1])

    # Generate C
    C = np.random.normal(0, 1, (n, 2))
    C_training = np.random.normal(0, 1, (n_training, 2))
    C_test = np.random.normal(0, 1, (n, 2))

    C = np.dot(C, v)
    C_training = np.dot(C_training, v)
    C_test = np.dot(C_test, v)

    # Generate Error Term
    u = np.random.normal(0, 1, n)
    u_training = np.random.normal(0, 1, n_training)
    u_test = np.random.normal(0, 1, n)
    # print("now:--------------^,^--------------")
    # print(alpha_0,Z[:, :2],Z[:, 3:],beta_Z[:2],beta_Z[3:],beta_X_to_D)
    # print("now:--------------^,^--------------")
    # Generate D
    # D = alpha_0 + np.sin(np.dot(Z[:, :5], beta_Z[:5])) + np.abs(np.dot(Z[:, 5:], beta_Z[5:])) + np.sin(np.dot(X,beta_X_to_D)) + alpha_3 * C + u
    # D_test = alpha_0 + np.sin(np.dot(Z_test[:, :5], beta_Z[:5])) + np.abs(np.dot(Z_test[:, 5:], beta_Z[5:])) + np.sin(np.dot(X_test, beta_X_to_D)) + alpha_3 * C_test + u_test
    # D_training = alpha_0 + np.sin(np.dot(Z_training[:, :5], beta_Z_training[:5]) )+ np.abs(np.dot(Z_training[:, 5:],beta_Z_training[5:])) + np.sin(np.dot(X_training, beta_X_to_D)) + alpha_3 * C_training + u_training

    # 假设 beta_X_to_D 的长度是 40
    D = alpha_0 + np.sin(np.dot(Z[:, :5], beta_Z[:5])) + np.abs(np.dot(Z[:, 5:], beta_Z[5:])) + np.sin(
        np.dot(X[:, :20], beta_X_to_D[:20])) + np.abs(np.dot(X[:, 20:], beta_X_to_D[20:])) + alpha_3 * C + u
    D_test = alpha_0 + np.sin(np.dot(Z_test[:, :5], beta_Z[:5])) + np.abs(np.dot(Z_test[:, 5:], beta_Z[5:])) + np.sin(
        np.dot(X_test[:, :20], beta_X_to_D[:20])) + np.abs(
        np.dot(X_test[:, 20:], beta_X_to_D[20:])) + alpha_3 * C_test + u_test
    D_training = alpha_0 + np.sin(np.dot(Z_training[:, :5], beta_Z_training[:5])) + np.abs(
        np.dot(Z_training[:, 5:], beta_Z_training[5:])) + np.sin(np.dot(X_training[:, :20], beta_X_to_D[:20])) + np.abs(
        np.dot(X_training[:, 20:], beta_X_to_D[20:])) + alpha_3 * C_training + u_training

    # Generate Y
    v = np.random.normal(0, 1, n)
    Y = beta3 + beta_1 * D + np.dot(X, beta_X_to_Y) + beta_6 * C + v
    v_training = np.random.normal(0, 1, n_training)
    Y_training = beta3 + beta_1 * D_training + np.dot(X_training, beta_X_to_Y) + beta_6 * C_training + v_training
    v_test = np.random.normal(0, 1, n)
    Y_test = beta3 + beta_1 * D_test + np.dot(X_test, beta_X_to_Y) + beta_6 * C_test + v_test

    # 计算方差
    Y_pred_intercept = np.full(n, alpha_0)  # 只有截距项的模型
    Y_pred_X = np.sin(np.dot(X[:, :20], beta_X_to_D[:20])) + np.abs(np.dot(X[:, 20:], beta_X_to_D[20:]))  # 包含 X 的模型
    Y_pred_XZ = alpha_0 + np.sin(np.dot(Z[:, :5], beta_Z[:5])) + np.abs(np.dot(Z[:, 5:], beta_Z[5:])) + np.sin(
        np.dot(X[:, :20], beta_X_to_D[:20])) + np.abs(
        np.dot(X[:, 20:], beta_X_to_D[20:]))  # + alpha_3 * C  # 包含 X 和 Z 的模型

    var_I_true = np.var(Y - Y_pred_intercept, ddof=1)
    var_X_true = np.var(Y - Y_pred_X, ddof=1)
    var_XZ_true = np.var(Y - Y_pred_XZ, ddof=1)
    # print(var_I_true)
    # print(var_X_true)
    # print(var_XZ_true)

    V_X_true = var_I_true - var_X_true
    V_XZ_true = var_I_true - var_XZ_true
    V_Z_true = V_XZ_true - V_X_true

    # print("V_X:", V_X_true)
    # print("V_XZ:", V_XZ_true)
    # print("V_Z:", V_Z_true)

    # 计算 IVStrength
    IVStrength = V_Z_true / V_XZ_true

    print("IVStrength:", IVStrength)
    print("IVStrength ratio:", IVStrength * n)

    return (Z, X, sm.add_constant(np.column_stack((Z, X))),
            Z_training, X_training, sm.add_constant(np.column_stack((Z_training, X_training))),
            Z_test, X_test, sm.add_constant(np.column_stack((Z_test, X_test))),
            D, D_training, D_test, Y, Y_training, Y_test, IVStrength * n)


# # 调用函数并打印结果
# V_X, V_XZ, V_Z, IVStrength = simulate_and_calculate_variances()
# print("V_X:", V_X)
# print("V_XZ:", V_XZ)
# print("V_Z:", V_Z)
# print("IVStrength:", IVStrength)

# Define the neural network model
def build_nn(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def train_models(X_train, D_train):
    # OLS
    model_ols = sm.OLS(D_train, X_train).fit()
    model_lasso = Lasso(alpha=0.1)
    model_lasso.fit(X_train, D_train)
    # Neural Network
    model_nn = build_nn(X_train.shape[1])
    model_nn.fit(X_train, D_train, epochs=10, batch_size=10, verbose=0)

    return model_ols, model_nn, model_lasso


def target_direct(x0, y0):
    input_dim = x0.shape[1]
    output_dim = 1  # Assuming the residual is a single value prediction

    # Use the refactored function to define the model architecture for residual neural network
    model_target = build_residual_nn(input_dim, output_dim)

    # Train the residual neural network on (X^(0), R^(m->0)) pairs
    model_target.fit(x0, y0, epochs=10, batch_size=10, verbose=0)
    return model_target


# def evaluate_models(model_ols, model_nn,model_lasso_initial,model_residual,model_oollss,model_lassoo_oollss, X, D, Y,model_nn_target):
#     D_hat_ols = model_ols.predict(X)
#     D_hat_nn_main = model_nn.predict(X).flatten()
#     D_hat_lasso_main = model_lasso_initial.predict(X).flatten()
#     D_hat_nn_residual = model_residual.predict(X).flatten()
#     D_hat_oollss_residual =model_oollss.predict(X)
#     D_hat_lassoo_residual = model_lassoo_oollss.predict(X)
#     D_hat_nn = D_hat_nn_main + D_hat_nn_residual
#     D_hat_nn_oollss=D_hat_nn_main+D_hat_oollss_residual
#     D_hat_lassoo_oollss = D_hat_lasso_main + D_hat_lassoo_residual
#     D_hat_model_target=model_nn_target.predict(X).flatten()
#
#     # Compute MSE
#     mse_ols = mean_squared_error(D, D_hat_ols)
#     mse_model_target = mean_squared_error(D, D_hat_model_target)
#     # Calculate MSE for D_hat_nn_main and D_hat_nn with true D
#     mse_nn_main = mean_squared_error(D, D_hat_nn_main)
#     mse_nn_combined = mean_squared_error(D, D_hat_nn)
#     mse_nn_oollss = mean_squared_error(D, D_hat_nn_oollss)
#     mse_lasso_main = mean_squared_error(D, D_hat_lasso_main)
#     mse_lasso_combine=mean_squared_error(D, D_hat_lassoo_oollss)
#     # Direct OLS Regression of Y on D
#     X_ols = sm.add_constant(D)
#     model_direct = sm.OLS(Y, X_ols).fit()
#     beta1_direct = model_direct.params[1]
#     beta1_direct_error = (1 - beta1_direct) ** 2
#
#     # Direct OLS Regression of Y on D
#     X_trans_Lasso = sm.add_constant(D_hat_lassoo_oollss)
#     model_trans_Lasso = sm.OLS(Y, X_trans_Lasso).fit()
#     beta1_trans_Lasso = model_trans_Lasso.params[1]
#
#     # Direct OLS Regression of Y on D
#     X_trans_nn = sm.add_constant(D_hat_nn)
#     model_trans_nn = sm.OLS(Y, X_trans_nn).fit()
#     beta1_trans_nn = model_trans_nn.params[1]
#
#     # 2SLS using OLS
#     n = len(Y)
#     numerator_2sls_ols = n * np.sum(D_hat_ols * Y) - np.sum(D_hat_ols) * np.sum(Y)
#     denominator_2sls_ols = n * np.sum(D_hat_ols ** 2) - np.sum(D_hat_ols) ** 2
#     beta1_2sls_ols = numerator_2sls_ols / denominator_2sls_ols
#     beta1_2sls_ols_error = (1 - beta1_2sls_ols) ** 2
#
#     # Model target
#     X_model_target = sm.add_constant(D_hat_model_target)
#     model_target = sm.OLS(Y, X_model_target).fit()
#     beta1_model_target = model_target.params[1]
#     beta1_model_target_error= (1 - beta1_model_target) ** 2
#
#
#     # 2SLS using Neural Network
#     numerator_2sls_nn = n * np.sum(D_hat_nn_main * Y) - np.sum(D_hat_nn_main) * np.sum(Y)
#     denominator_2sls_nn = n * np.sum(D_hat_nn_main ** 2) - np.sum(D_hat_nn_main) ** 2
#     beta1_2sls_nn = numerator_2sls_nn / denominator_2sls_nn
#     beta1_2sls_nn_main_error = (1 - beta1_2sls_nn) ** 2
#
#     # 2SLS using Neural Network
#     numerator_2sls_lasso = n * np.sum(D_hat_lasso_main * Y) - np.sum(D_hat_lasso_main) * np.sum(Y)
#     denominator_2sls_lasso = n * np.sum(D_hat_lasso_main ** 2) - np.sum(D_hat_lasso_main) ** 2
#     beta1_2sls_lasso = numerator_2sls_lasso / denominator_2sls_lasso
#
#     # 2SLS using Neural Network
#     numerator_2sls_nn_trans = n * np.sum(D_hat_nn * Y) - np.sum(D_hat_nn) * np.sum(Y)
#     denominator_2sls_nn_trans = n * np.sum(D_hat_nn ** 2) - np.sum(D_hat_nn) ** 2
#     beta1_2sls_nn_trans = numerator_2sls_nn_trans / denominator_2sls_nn_trans
#     beta1_2sls_nn_trans_error = (1 - beta1_2sls_nn_trans) ** 2
#
#
#     # 2SLS using Neural Network
#     numerator_2sls_lasso_trans = n * np.sum(D_hat_nn_oollss * Y) - np.sum(D_hat_nn_oollss) * np.sum(Y)
#     denominator_2sls_lasso_trans= n * np.sum(D_hat_nn_oollss ** 2) - np.sum(D_hat_nn_oollss) ** 2
#     beta1_2sls_lasso_trans = numerator_2sls_lasso_trans / denominator_2sls_lasso_trans
#     beta1_2sls_nn_trans_lasso_error = (1 - beta1_2sls_lasso_trans) ** 2
#
#     # 2SLS using Neural Network
#     numerator_2sls_llaassoo_trans = n * np.sum(D_hat_lassoo_oollss * Y) - np.sum(D_hat_lassoo_oollss) * np.sum(Y)
#     denominator_2sls_llaassoo__trans= n * np.sum(D_hat_lassoo_oollss ** 2) - np.sum(D_hat_lassoo_oollss) ** 2
#     beta1_2sls_llaassoo_trans = numerator_2sls_llaassoo_trans / denominator_2sls_llaassoo__trans
#     print(D)
#     print(D_hat_nn)
#     print(D_hat_nn_oollss)
#     squared_diff_sum = np.sum((D_hat_nn - D) ** 2)
#     squared_diff_summ = np.sum((D_hat_nn_oollss - D) ** 2)
#     print(squared_diff_sum)
#     print(squared_diff_summ)
#
#
#     results = {
#         #"mse_ols": mse_ols,
#         "mse_nn_main": mse_nn_main,            # added this
#         "mse_nn_combined": mse_nn_combined,    # added this
#         #"mse_lasso_main": mse_lasso_main,  # added this
#         #"mse_lasso_combine": mse_lasso_combine,  # added this
#         "mse_nn_oollss":mse_nn_oollss,
#         "mse_model_target": mse_model_target,
#         "beta1_direct": beta1_direct,
#         #"beta1_trans_Lasso": beta1_trans_Lasso,
#         #"beta1_trans_nn":beta1_trans_nn,
#         #"beta1_2sls_ols": beta1_2sls_ols,
#         "beta1_2sls_nn_main": beta1_2sls_nn,
#         #"beta1_2sls_lasso_main": beta1_2sls_lasso,
#         "beta1_2sls_nn_trans": beta1_2sls_nn_trans,
#         "beta1_2sls_lasso_trans": beta1_2sls_lasso_trans,
#         "beta1_model_target": beta1_model_target,
#         #"beta1_2sls_llaassoo_trans": beta1_2sls_llaassoo_trans,
#         "beta1_direct_error": beta1_direct_error,
#         "beta1_2sls_nn_main_error": beta1_2sls_nn_main_error,
#         "beta1_2sls_nn_trans_error": beta1_2sls_nn_trans_error,
#         "beta1_2sls_nn_trans_lasso_error": beta1_2sls_nn_trans_lasso_error,
#         "beta1_model_target_error": beta1_model_target_error,
#     }
#
#     return results
#
# def evaluate_models(model_ols, model_nn,model_lasso_initial,model_residual,model_oollss,model_lassoo_oollss, X, D, Y,model_nn_target):
#     D_hat_ols = model_ols.predict(X)
#     D_hat_nn_main = model_nn.predict(X,verbose=0).flatten()
#     D_hat_lasso_main = model_lasso_initial.predict(X).flatten()
#     D_hat_nn_residual = model_residual.predict(X,verbose=0).flatten()
#     D_hat_oollss_residual =model_oollss.predict(X)
#     D_hat_lassoo_residual = model_lassoo_oollss.predict(X)
#     D_hat_nn = D_hat_nn_main + D_hat_nn_residual
#     D_hat_nn_oollss=D_hat_nn_main+D_hat_oollss_residual
#     D_hat_lassoo_oollss = D_hat_lasso_main + D_hat_lassoo_residual
#     D_hat_model_target=model_nn_target.predict(X,verbose=0).flatten()
#
#     # Compute MSE
#     mse_ols = mean_squared_error(D, D_hat_ols)
#     mse_model_target = mean_squared_error(D, D_hat_model_target)
#     # Calculate MSE for D_hat_nn_main and D_hat_nn with true D
#     mse_nn_main = mean_squared_error(D, D_hat_nn_main)
#     mse_nn_combined = mean_squared_error(D, D_hat_nn)
#     mse_nn_oollss = mean_squared_error(D, D_hat_nn_oollss)
#     mse_lasso_main = mean_squared_error(D, D_hat_lasso_main)
#     mse_lasso_combine=mean_squared_error(D, D_hat_lassoo_oollss)
#     # Direct OLS Regression of Y on D
#     X_ols = sm.add_constant(D)
#     model_direct = sm.OLS(Y, X_ols).fit()
#     beta1_direct = model_direct.params[1]
#     beta1_direct_error = (1 - beta1_direct) ** 2
#
#     # Direct OLS Regression of Y on D
#     X_trans_Lasso = sm.add_constant(D_hat_lassoo_oollss)
#     model_trans_Lasso = sm.OLS(Y, X_trans_Lasso).fit()
#     beta1_trans_Lasso = model_trans_Lasso.params[1]
#
#     # Direct OLS Regression of Y on D
#     X_trans_nn = sm.add_constant(D_hat_nn)
#     model_trans_nn = sm.OLS(Y, X_trans_nn).fit()
#     beta1_trans_nn = model_trans_nn.params[1]
#
#     # 2SLS using OLS
#     n = len(Y)
#     numerator_2sls_ols = n * np.sum(D_hat_ols * Y) - np.sum(D_hat_ols) * np.sum(Y)
#     denominator_2sls_ols = n * np.sum(D_hat_ols ** 2) - np.sum(D_hat_ols) ** 2
#     beta1_2sls_ols = numerator_2sls_ols / denominator_2sls_ols
#     beta1_2sls_ols_error = (1 - beta1_2sls_ols) ** 2
#
#     # Model target
#     X_model_target = sm.add_constant(D_hat_model_target)
#     model_target = sm.OLS(Y, X_model_target).fit()
#     beta1_model_target = model_target.params[1]
#     beta1_model_target_error= (1 - beta1_model_target) ** 2
#
#
#     # 2SLS using Neural Network
#     numerator_2sls_nn = n * np.sum(D_hat_nn_main * Y) - np.sum(D_hat_nn_main) * np.sum(Y)
#     denominator_2sls_nn = n * np.sum(D_hat_nn_main ** 2) - np.sum(D_hat_nn_main) ** 2
#     beta1_2sls_nn = numerator_2sls_nn / denominator_2sls_nn
#     beta1_2sls_nn_main_error = (1 - beta1_2sls_nn) ** 2
#
#     # 2SLS using Neural Network
#     numerator_2sls_lasso = n * np.sum(D_hat_lasso_main * Y) - np.sum(D_hat_lasso_main) * np.sum(Y)
#     denominator_2sls_lasso = n * np.sum(D_hat_lasso_main ** 2) - np.sum(D_hat_lasso_main) ** 2
#     beta1_2sls_lasso = numerator_2sls_lasso / denominator_2sls_lasso
#
#     # 2SLS using Neural Network
#     numerator_2sls_nn_trans = n * np.sum(D_hat_nn * Y) - np.sum(D_hat_nn) * np.sum(Y)
#     denominator_2sls_nn_trans = n * np.sum(D_hat_nn ** 2) - np.sum(D_hat_nn) ** 2
#     beta1_2sls_nn_trans = numerator_2sls_nn_trans / denominator_2sls_nn_trans
#     beta1_2sls_nn_trans_error = (1 - beta1_2sls_nn_trans) ** 2
#
#
#     # 2SLS using Neural Network
#     numerator_2sls_lasso_trans = n * np.sum(D_hat_nn_oollss * Y) - np.sum(D_hat_nn_oollss) * np.sum(Y)
#     denominator_2sls_lasso_trans= n * np.sum(D_hat_nn_oollss ** 2) - np.sum(D_hat_nn_oollss) ** 2
#     beta1_2sls_lasso_trans = numerator_2sls_lasso_trans / denominator_2sls_lasso_trans
#     beta1_2sls_nn_trans_lasso_error = (1 - beta1_2sls_lasso_trans) ** 2
#
#     # 2SLS using Neural Network
#     numerator_2sls_llaassoo_trans = n * np.sum(D_hat_lassoo_oollss * Y) - np.sum(D_hat_lassoo_oollss) * np.sum(Y)
#     denominator_2sls_llaassoo__trans= n * np.sum(D_hat_lassoo_oollss ** 2) - np.sum(D_hat_lassoo_oollss) ** 2
#     beta1_2sls_llaassoo_trans = numerator_2sls_llaassoo_trans / denominator_2sls_llaassoo__trans
#     # print(D)
#     # print(D_hat_nn)
#     # print(D_hat_nn_oollss)
#     squared_diff_sum = np.sum((D_hat_nn - D) ** 2)
#     squared_diff_summ = np.sum((D_hat_nn_oollss - D) ** 2)
#     # print(squared_diff_sum)
#     # print(squared_diff_summ)
#
#
#     results = {
#         "beta1_direct_ols": beta1_direct,
#         #"beta1_trans_Lasso": beta1_trans_Lasso,
#         #"beta1_trans_nn":beta1_trans_nn,
#         #"beta1_2sls_ols": beta1_2sls_ols,
#         "beta1_direct_nn_main": beta1_2sls_nn,
#         #"beta1_2sls_lasso_main": beta1_2sls_lasso,
#         "beta1_direct_nn": beta1_2sls_nn_trans,
#         "beta1_direct_lassoo_oollss": beta1_2sls_lasso_trans,
#         "beta1_direct_model_target": beta1_model_target,
#         #"beta1_2sls_llaassoo_trans": beta1_2sls_llaassoo_trans,
#         "error_beta1_direct_ols": abs(1 - beta1_direct),
#         "error_beta1_direct_nn_main": abs(1 - beta1_2sls_nn),
#         "error_beta1_direct_nn": abs(1 - beta1_2sls_nn_trans),
#         "error_beta1_direct_lassoo_oollss": abs(1 - beta1_2sls_lasso_trans),
#         "error_beta1_direct_model_target": abs(1 - beta1_model_target),
#         "mse_D_hat_hat_ols": mean_squared_error(D, D_hat_ols),
#         "mse_D_hat_hat_nn_main": mean_squared_error(D, D_hat_nn_main),
#         "mse_D_hat_hat_nn": mean_squared_error(D, D_hat_nn),
#         "mse_D_hat_hat_lassoo_oollss": mean_squared_error(D, D_hat_lassoo_oollss),
#         "mse_D_hat_hat_model_target": mean_squared_error(D, D_hat_model_target)
#     }
#
#     return results

def evaluate_models(model_ols, model_nn, model_lasso_initial, model_residual, model_oollss, model_lassoo_oollss, X, D,
                    Y, model_nn_target):
    # Split the data
    # X_C, X, D_C, D, Y_C, Y = train_test_split(X, D, Y, test_size=0.5)

    D_hat_ols = model_ols.predict(X)
    D_hat_nn_main = model_nn.predict(X, verbose=0).flatten()
    D_hat_lasso_main = model_lasso_initial.predict(X).flatten()
    D_hat_nn_residual = model_residual.predict(X, verbose=0).flatten()
    D_hat_oollss_residual = model_oollss.predict(X)
    D_hat_lassoo_residual = model_lassoo_oollss.predict(X)
    D_hat_nn = D_hat_nn_main + D_hat_nn_residual
    D_hat_nn_oollss = D_hat_nn_main + D_hat_oollss_residual
    D_hat_lassoo_oollss = D_hat_lasso_main + D_hat_lassoo_residual
    D_hat_model_target = model_nn_target.predict(X, verbose=0).flatten()

    alpha_hat_ols = LinearRegression().fit(D.reshape(-1, 1), D).coef_[0]
    alpha_hat_nn_main = LinearRegression().fit(D_hat_nn_main.reshape(-1, 1), D).coef_[0]
    alpha_hat_nn = LinearRegression().fit(D_hat_nn.reshape(-1, 1), D).coef_[0]
    alpha_hat_lassoo_oollss = LinearRegression().fit(D_hat_nn_oollss.reshape(-1, 1), D).coef_[0]
    alpha_hat_model_target = LinearRegression().fit(D_hat_model_target.reshape(-1, 1), D).coef_[0]

    # D_hat_hat_ols = model_ols.predict(X_C)
    # D_hat_hat_nn_main = model_nn.predict(X_C).flatten()
    # D_hat_hat_nn_residual = model_residual.predict(X_C).flatten()
    # D_hat_hat_oollss_residual =model_oollss.predict(X_C)
    # D_hat_hat_nn = D_hat_hat_nn_main+D_hat_hat_nn_residual
    # D_hat_hat_lassoo_oollss = D_hat_hat_nn_main+D_hat_hat_oollss_residual
    # D_hat_hat_model_target = model_nn_target.predict(X_C).flatten()

    # Use alphas and D_hat_hat for OLS regression of Y on alpha*D
    def perform_direct_ols_regression(Y, D_hat, alpha):
        X_ols = sm.add_constant(alpha * D_hat)
        model_direct = sm.OLS(Y, X_ols).fit()
        # 检查模型参数的长度是否至少为2
        if len(model_direct.params) > 1:
            return model_direct.params[1]
        else:
            # 打印相关信息
            print("模型参数不足以进行回归。返回默认值 0。")
            print("Y:", Y)
            print("alpha:", alpha)
            print("D_hat:", D_hat)
            return 0  # 如果没有第二个参数，则返回0

    beta1_direct_ols = perform_direct_ols_regression(Y, D, alpha_hat_ols)
    beta1_direct_nn_main = perform_direct_ols_regression(Y, D_hat_nn_main, alpha_hat_nn_main)
    beta1_direct_nn = perform_direct_ols_regression(Y, D_hat_nn, alpha_hat_nn)
    beta1_direct_lassoo_oollss = perform_direct_ols_regression(Y, D_hat_nn_oollss, alpha_hat_lassoo_oollss)
    beta1_direct_model_target = perform_direct_ols_regression(Y, D_hat_model_target, alpha_hat_model_target)
    #
    # beta1_direct_ols = perform_direct_ols_regression(Y_C, D_hat_hat_ols, alpha_hat_ols)
    # beta1_direct_nn_main = perform_direct_ols_regression(Y_C, D_hat_hat_nn_main, alpha_hat_nn_main)
    # beta1_direct_nn = perform_direct_ols_regression(Y_C, D_hat_hat_nn, alpha_hat_nn)
    # beta1_direct_lassoo_oollss = perform_direct_ols_regression(Y_C, D_hat_hat_lassoo_oollss, alpha_hat_lassoo_oollss)
    # beta1_direct_model_target = perform_direct_ols_regression(Y_C, D_hat_hat_model_target, alpha_hat_model_target)

    # Compile results
    results = {
        "beta1_direct_ols": beta1_direct_ols,
        "beta1_direct_nn_main": beta1_direct_nn_main,
        "beta1_direct_nn": beta1_direct_nn,
        "beta1_direct_lassoo_oollss": beta1_direct_lassoo_oollss,
        "beta1_direct_model_target": beta1_direct_model_target,

        # Compute the absolute error from the truth (assuming truth is 1)
        "error_beta1_direct_ols": (1 - beta1_direct_ols) ** 2,
        "error_beta1_direct_nn_main": (1 - beta1_direct_ols) ** 2,
        "error_beta1_direct_nn": (1 - beta1_direct_ols) ** 2,
        "error_beta1_direct_lassoo_oollss": (1 - beta1_direct_ols) ** 2,
        "error_beta1_direct_model_target": (1 - beta1_direct_ols) ** 2,

        # Compute MSE between D_hat_hat and true D
        "mse_D_hat_hat_ols": mean_squared_error(D, D_hat_ols),
        "mse_D_hat_hat_nn_main": mean_squared_error(D, D_hat_nn_main),
        "mse_D_hat_hat_nn": mean_squared_error(D, D_hat_nn),
        "mse_D_hat_hat_lassoo_oollss": mean_squared_error(D, D_hat_nn_oollss),
        "mse_D_hat_hat_model_target": mean_squared_error(D, D_hat_model_target)
    }
    return results


def build_residual_nn(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(output_dim))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def compute_residuals(model_nn_initial, model_lasso_initial, x0, y0):
    residuals = []
    for i in range(len(x0)):
        X_i_0 = x0[i].reshape(1, -1)
        Y_i_0 = y0[i]
        # Printing is generally for debugging, consider removing for performance if not needed
        #        print(x0[i])
        #        print(X_i_0)
        #        print(Y_i_0)
        residual = Y_i_0 - model_nn_initial.predict(X_i_0, verbose=0).flatten()
        #        print(residual)
        residuals.append(residual)

    residuals = np.array(residuals)
    input_dim = x0.shape[1]
    output_dim = 1  # Assuming the residual is a single value prediction

    # Use the refactored function to define the model architecture for residual neural network
    model_residual = build_residual_nn(input_dim, output_dim)

    # Train the residual neural network on (X^(0), R^(m->0)) pairs
    model_residual.fit(x0, residuals, epochs=10, batch_size=10, verbose=0)

    # Define the Lasso regression model
    # You can specify the alpha parameter here; alpha is the regularization strength
    model_oollss = Lasso(alpha=0.1)

    # Fit the Lasso model to your data
    model_oollss.fit(x0, residuals)
    residuals = []
    for i in range(len(x0)):
        X_i_0 = x0[i].reshape(1, -1)
        Y_i_0 = y0[i]
        # Printing is generally for debugging, consider removing for performance if not needed
        #        print(x0[i])
        #        print(X_i_0)
        #        print(Y_i_0)
        residual = Y_i_0 - model_lasso_initial.predict(X_i_0).flatten()
        #        print(residual)
        residuals.append(residual)

    residuals = np.array(residuals)

    # You can specify the alpha parameter here; alpha is the regularization strength
    model_lassoo_oollss = Lasso(alpha=0.1)

    # Fit the Lasso model to your data
    model_lassoo_oollss.fit(x0, residuals)

    return model_nn_initial, model_residual, model_oollss, model_lassoo_oollss


from sklearn.utils import resample
import statsmodels.api as sm


def bootstrap_evaluate(model_ols, model_nn_initial, model_lasso_initial, model_residual, model_oollss,
                       model_lassoo_oollss, Z, D, Y, model_nn_target, n_bootstrap=500):
    bootstrap_results = {
        "beta1_direct_ols": [],
        "beta1_direct_nn_main": [],
        "beta1_direct_nn": [],
        "beta1_direct_lassoo_oollss": [],
        "beta1_direct_model_target": []
    }

    for _ in range(n_bootstrap):
        indices = resample(range(len(Z)), replace=True)
        Z_bs, D_bs, Y_bs = Z[indices], D[indices], Y[indices]
        results_bs = evaluate_models(model_ols, model_nn_initial, model_lasso_initial, model_residual, model_oollss,
                                     model_lassoo_oollss, Z_bs, D_bs, Y_bs, model_nn_target)

        for key in bootstrap_results:
            bootstrap_results[key].append(results_bs[key])

    # print("bootstrap_results:",bootstrap_results)
    summary = {}
    for key in bootstrap_results:
        estimates = bootstrap_results[key]
        mean_estimate = np.mean(estimates)
        std_dev = np.sqrt(np.var(bootstrap_results[key]))
        lower_bound = mean_estimate - 1.96 * std_dev
        upper_bound = mean_estimate + 1.96 * std_dev
        covers_truth = lower_bound <= beta1 <= upper_bound

        summary[key] = {"mean": mean_estimate, "CI": (lower_bound, upper_bound), "covers_truth": covers_truth}

    return bootstrap_results


if __name__ == "__main__":
    num_iterations = 500
    all_results = []
    IV_str = []
    covers_truth_dict = {
        "beta1_direct_ols": [],
        "beta1_direct_nn_main": [],
        "beta1_direct_nn": [],
        "beta1_direct_lassoo_oollss": [],
        "beta1_direct_model_target": []
    }
    for _ in range(num_iterations):
        Z, X, ZX_combined, Z_training, X_training, ZX_training_combined, Z_test, X_test, ZX_test_combine, D, D_training, D_test, Y, Y_training, Y_test, IVStrength = simulate_and_calculate_variances()
        model_ols, model_nn_initial, model_lasso_initial = train_models(ZX_training_combined, D_training)
        model_nn_initial, model_residual, model_oollss, model_lassoo_oollss = compute_residuals(model_nn_initial,
                                                                                                model_lasso_initial,
                                                                                                ZX_test_combine, D_test)
        model_nn_target = target_direct(ZX_test_combine, D_test)
        results = evaluate_models(model_ols, model_nn_initial, model_lasso_initial, model_residual, model_oollss,
                                  model_lassoo_oollss, ZX_combined, D, Y, model_nn_target)
        all_results.append(results)

        # print(all_results)
        IV_str.append(IVStrength)
        #
        # bootstrap_results = bootstrap_evaluate(model_ols, model_nn_initial, model_lasso_initial, model_residual, model_oollss,model_lassoo_oollss, ZX_combined, D, Y,model_nn_target)
        # # 新添加的绘制直方图的代码
        # import matplotlib.pyplot as plt
        #
        # # 绘制直方图
        # beta1_keys = ["beta1_direct_ols", "beta1_direct_nn_main", "beta1_direct_nn",
        #               "beta1_direct_lassoo_oollss", "beta1_direct_model_target"]
        #
        # # for key in beta1_keys:
        # #     plt.figure()
        # #     plt.hist(bootstrap_results[key], bins=100, alpha=0.7)
        # #     plt.title(f"Histogram of {key}")
        # #     plt.xlabel(key)
        # #     plt.ylabel("Frequency")
        # #     plt.show()
        #
        # for key in covers_truth_dict.keys():
        #     # 计算均值和置信区间
        #     mean_estimate = np.mean(bootstrap_results[key])
        #     std_dev = np.sqrt(np.var(bootstrap_results[key], ddof=1))
        #     print("std:",std_dev)
        #     ci_lower = mean_estimate - 1.96 * std_dev
        #     ci_upper = mean_estimate + 1.96 * std_dev
        #
        #     # 判断置信区间是否覆盖了 beta1 并记录结果
        #     print("CI:",ci_lower,ci_upper,beta1)
        #     covers_truth = ci_lower <= beta1 <= ci_upper
        #     covers_truth_dict[key].append(covers_truth)
        #     print(covers_truth_dict)

    # 假设 results 和 all_results 是已经定义好的变量
    # mean
    print(IV_str)
    mean_results = {}
    for key in results.keys():
        mean_results[key] = np.mean([result[key] for result in all_results])

    # variance
    var_results = {}
    for key in results.keys():
        var_results[key] = np.var([result[key] for result in all_results])

    # 将结果写入文件
    with open("IVD1.txt", "w") as file:
        file.write("mean：\n")
        for key, value in mean_results.items():
            file.write(f"{key}: {value}\n")

        file.write("\nvar：\n")
        for key, value in var_results.items():
            file.write(f"{key}: {value}\n")

    # 将 IV_str 写入另一个文件
    with open("IVDstr1.txt", "w") as file:
        for iv_value in IV_str:
            file.write(f"{iv_value}\n")

    all_results = []
    IV_str = []
    covers_truth_dict = {
        "beta1_direct_ols": [],
        "beta1_direct_nn_main": [],
        "beta1_direct_nn": [],
        "beta1_direct_lassoo_oollss": [],
        "beta1_direct_model_target": []
    }
    for _ in range(num_iterations):
        Z, X, ZX_combined, Z_training, X_training, ZX_training_combined, Z_test, X_test, ZX_test_combine, D, D_training, D_test, Y, Y_training, Y_test, IVStrength = simulate_and_calculate_variances2()
        model_ols, model_nn_initial, model_lasso_initial = train_models(ZX_training_combined, D_training)
        model_nn_initial, model_residual, model_oollss, model_lassoo_oollss = compute_residuals(model_nn_initial,
                                                                                                model_lasso_initial,
                                                                                                ZX_test_combine, D_test)
        model_nn_target = target_direct(ZX_test_combine, D_test)
        results = evaluate_models(model_ols, model_nn_initial, model_lasso_initial, model_residual, model_oollss,
                                  model_lassoo_oollss, ZX_combined, D, Y, model_nn_target)
        all_results.append(results)

        # print(all_results)
        IV_str.append(IVStrength)
        #
        # bootstrap_results = bootstrap_evaluate(model_ols, model_nn_initial, model_lasso_initial, model_residual, model_oollss,model_lassoo_oollss, ZX_combined, D, Y,model_nn_target)
        # # 新添加的绘制直方图的代码
        # import matplotlib.pyplot as plt
        #
        # # 绘制直方图
        # beta1_keys = ["beta1_direct_ols", "beta1_direct_nn_main", "beta1_direct_nn",
        #               "beta1_direct_lassoo_oollss", "beta1_direct_model_target"]
        #
        # # for key in beta1_keys:
        # #     plt.figure()
        # #     plt.hist(bootstrap_results[key], bins=100, alpha=0.7)
        # #     plt.title(f"Histogram of {key}")
        # #     plt.xlabel(key)
        # #     plt.ylabel("Frequency")
        # #     plt.show()
        #
        # for key in covers_truth_dict.keys():
        #     # 计算均值和置信区间
        #     mean_estimate = np.mean(bootstrap_results[key])
        #     std_dev = np.sqrt(np.var(bootstrap_results[key], ddof=1))
        #     print("std:",std_dev)
        #     ci_lower = mean_estimate - 1.96 * std_dev
        #     ci_upper = mean_estimate + 1.96 * std_dev
        #
        #     # 判断置信区间是否覆盖了 beta1 并记录结果
        #     print("CI:",ci_lower,ci_upper,beta1)
        #     covers_truth = ci_lower <= beta1 <= ci_upper
        #     covers_truth_dict[key].append(covers_truth)
        #     print(covers_truth_dict)

    # 假设 results 和 all_results 是已经定义好的变量
    # mean
    print(IV_str)
    mean_results = {}
    for key in results.keys():
        mean_results[key] = np.mean([result[key] for result in all_results])

    # variance
    var_results = {}
    for key in results.keys():
        var_results[key] = np.var([result[key] for result in all_results])

    # 将结果写入文件
    with open("IVD2.txt", "w") as file:
        file.write("mean：\n")
        for key, value in mean_results.items():
            file.write(f"{key}: {value}\n")

        file.write("\nvar：\n")
        for key, value in var_results.items():
            file.write(f"{key}: {value}\n")

    # 将 IV_str 写入另一个文件
    with open("IVDstr2.txt", "w") as file:
        for iv_value in IV_str:
            file.write(f"{iv_value}\n")

    all_results = []
    IV_str = []
    covers_truth_dict = {
        "beta1_direct_ols": [],
        "beta1_direct_nn_main": [],
        "beta1_direct_nn": [],
        "beta1_direct_lassoo_oollss": [],
        "beta1_direct_model_target": []
    }
    for _ in range(num_iterations):
        Z, X, ZX_combined, Z_training, X_training, ZX_training_combined, Z_test, X_test, ZX_test_combine, D, D_training, D_test, Y, Y_training, Y_test, IVStrength = simulate_and_calculate_variances3()
        model_ols, model_nn_initial, model_lasso_initial = train_models(ZX_training_combined, D_training)
        model_nn_initial, model_residual, model_oollss, model_lassoo_oollss = compute_residuals(model_nn_initial,
                                                                                                model_lasso_initial,
                                                                                                ZX_test_combine, D_test)
        model_nn_target = target_direct(ZX_test_combine, D_test)
        results = evaluate_models(model_ols, model_nn_initial, model_lasso_initial, model_residual, model_oollss,
                                  model_lassoo_oollss, ZX_combined, D, Y, model_nn_target)
        all_results.append(results)

        # print(all_results)
        IV_str.append(IVStrength)
        #
        # bootstrap_results = bootstrap_evaluate(model_ols, model_nn_initial, model_lasso_initial, model_residual, model_oollss,model_lassoo_oollss, ZX_combined, D, Y,model_nn_target)
        # # 新添加的绘制直方图的代码
        # import matplotlib.pyplot as plt
        #
        # # 绘制直方图
        # beta1_keys = ["beta1_direct_ols", "beta1_direct_nn_main", "beta1_direct_nn",
        #               "beta1_direct_lassoo_oollss", "beta1_direct_model_target"]
        #
        # # for key in beta1_keys:
        # #     plt.figure()
        # #     plt.hist(bootstrap_results[key], bins=100, alpha=0.7)
        # #     plt.title(f"Histogram of {key}")
        # #     plt.xlabel(key)
        # #     plt.ylabel("Frequency")
        # #     plt.show()
        #
        # for key in covers_truth_dict.keys():
        #     # 计算均值和置信区间
        #     mean_estimate = np.mean(bootstrap_results[key])
        #     std_dev = np.sqrt(np.var(bootstrap_results[key], ddof=1))
        #     print("std:",std_dev)
        #     ci_lower = mean_estimate - 1.96 * std_dev
        #     ci_upper = mean_estimate + 1.96 * std_dev
        #
        #     # 判断置信区间是否覆盖了 beta1 并记录结果
        #     print("CI:",ci_lower,ci_upper,beta1)
        #     covers_truth = ci_lower <= beta1 <= ci_upper
        #     covers_truth_dict[key].append(covers_truth)
        #     print(covers_truth_dict)

    # 假设 results 和 all_results 是已经定义好的变量
    # mean
    print(IV_str)
    mean_results = {}
    for key in results.keys():
        mean_results[key] = np.mean([result[key] for result in all_results])

    # variance
    var_results = {}
    for key in results.keys():
        var_results[key] = np.var([result[key] for result in all_results])

    # 将结果写入文件
    with open("IVD3.txt", "w") as file:
        file.write("mean：\n")
        for key, value in mean_results.items():
            file.write(f"{key}: {value}\n")

        file.write("\nvar：\n")
        for key, value in var_results.items():
            file.write(f"{key}: {value}\n")

    # 将 IV_str 写入另一个文件
    with open("IVDstr3.txt", "w") as file:
        for iv_value in IV_str:
            file.write(f"{iv_value}\n")

    all_results = []
    IV_str = []
    covers_truth_dict = {
        "beta1_direct_ols": [],
        "beta1_direct_nn_main": [],
        "beta1_direct_nn": [],
        "beta1_direct_lassoo_oollss": [],
        "beta1_direct_model_target": []
    }
    for _ in range(num_iterations):
        Z, X, ZX_combined, Z_training, X_training, ZX_training_combined, Z_test, X_test, ZX_test_combine, D, D_training, D_test, Y, Y_training, Y_test, IVStrength = simulate_and_calculate_variances4()
        model_ols, model_nn_initial, model_lasso_initial = train_models(ZX_training_combined, D_training)
        model_nn_initial, model_residual, model_oollss, model_lassoo_oollss = compute_residuals(model_nn_initial,
                                                                                                model_lasso_initial,
                                                                                                ZX_test_combine, D_test)
        model_nn_target = target_direct(ZX_test_combine, D_test)
        results = evaluate_models(model_ols, model_nn_initial, model_lasso_initial, model_residual, model_oollss,
                                  model_lassoo_oollss, ZX_combined, D, Y, model_nn_target)
        all_results.append(results)

        # print(all_results)
        IV_str.append(IVStrength)
        #
        # bootstrap_results = bootstrap_evaluate(model_ols, model_nn_initial, model_lasso_initial, model_residual, model_oollss,model_lassoo_oollss, ZX_combined, D, Y,model_nn_target)
        # # 新添加的绘制直方图的代码
        # import matplotlib.pyplot as plt
        #
        # # 绘制直方图
        # beta1_keys = ["beta1_direct_ols", "beta1_direct_nn_main", "beta1_direct_nn",
        #               "beta1_direct_lassoo_oollss", "beta1_direct_model_target"]
        #
        # # for key in beta1_keys:
        # #     plt.figure()
        # #     plt.hist(bootstrap_results[key], bins=100, alpha=0.7)
        # #     plt.title(f"Histogram of {key}")
        # #     plt.xlabel(key)
        # #     plt.ylabel("Frequency")
        # #     plt.show()
        #
        # for key in covers_truth_dict.keys():
        #     # 计算均值和置信区间
        #     mean_estimate = np.mean(bootstrap_results[key])
        #     std_dev = np.sqrt(np.var(bootstrap_results[key], ddof=1))
        #     print("std:",std_dev)
        #     ci_lower = mean_estimate - 1.96 * std_dev
        #     ci_upper = mean_estimate + 1.96 * std_dev
        #
        #     # 判断置信区间是否覆盖了 beta1 并记录结果
        #     print("CI:",ci_lower,ci_upper,beta1)
        #     covers_truth = ci_lower <= beta1 <= ci_upper
        #     covers_truth_dict[key].append(covers_truth)
        #     print(covers_truth_dict)

    # 假设 results 和 all_results 是已经定义好的变量
    # mean
    print(IV_str)
    mean_results = {}
    for key in results.keys():
        mean_results[key] = np.mean([result[key] for result in all_results])

    # variance
    var_results = {}
    for key in results.keys():
        var_results[key] = np.var([result[key] for result in all_results])

    # 将结果写入文件
    with open("IVD4.txt", "w") as file:
        file.write("mean：\n")
        for key, value in mean_results.items():
            file.write(f"{key}: {value}\n")

        file.write("\nvar：\n")
        for key, value in var_results.items():
            file.write(f"{key}: {value}\n")

    # 将 IV_str 写入另一个文件
    with open("IVDstr4.txt", "w") as file:
        for iv_value in IV_str:
            file.write(f"{iv_value}\n")

    all_results = []
    IV_str = []
    covers_truth_dict = {
        "beta1_direct_ols": [],
        "beta1_direct_nn_main": [],
        "beta1_direct_nn": [],
        "beta1_direct_lassoo_oollss": [],
        "beta1_direct_model_target": []
    }
    for _ in range(num_iterations):
        Z, X, ZX_combined, Z_training, X_training, ZX_training_combined, Z_test, X_test, ZX_test_combine, D, D_training, D_test, Y, Y_training, Y_test, IVStrength = simulate_and_calculate_variances5()
        model_ols, model_nn_initial, model_lasso_initial = train_models(ZX_training_combined, D_training)
        model_nn_initial, model_residual, model_oollss, model_lassoo_oollss = compute_residuals(model_nn_initial,
                                                                                                model_lasso_initial,
                                                                                                ZX_test_combine, D_test)
        model_nn_target = target_direct(ZX_test_combine, D_test)
        results = evaluate_models(model_ols, model_nn_initial, model_lasso_initial, model_residual, model_oollss,
                                  model_lassoo_oollss, ZX_combined, D, Y, model_nn_target)
        all_results.append(results)

        # print(all_results)
        IV_str.append(IVStrength)
        #
        # bootstrap_results = bootstrap_evaluate(model_ols, model_nn_initial, model_lasso_initial, model_residual, model_oollss,model_lassoo_oollss, ZX_combined, D, Y,model_nn_target)
        # # 新添加的绘制直方图的代码
        # import matplotlib.pyplot as plt
        #
        # # 绘制直方图
        # beta1_keys = ["beta1_direct_ols", "beta1_direct_nn_main", "beta1_direct_nn",
        #               "beta1_direct_lassoo_oollss", "beta1_direct_model_target"]
        #
        # # for key in beta1_keys:
        # #     plt.figure()
        # #     plt.hist(bootstrap_results[key], bins=100, alpha=0.7)
        # #     plt.title(f"Histogram of {key}")
        # #     plt.xlabel(key)
        # #     plt.ylabel("Frequency")
        # #     plt.show()
        #
        # for key in covers_truth_dict.keys():
        #     # 计算均值和置信区间
        #     mean_estimate = np.mean(bootstrap_results[key])
        #     std_dev = np.sqrt(np.var(bootstrap_results[key], ddof=1))
        #     print("std:",std_dev)
        #     ci_lower = mean_estimate - 1.96 * std_dev
        #     ci_upper = mean_estimate + 1.96 * std_dev
        #
        #     # 判断置信区间是否覆盖了 beta1 并记录结果
        #     print("CI:",ci_lower,ci_upper,beta1)
        #     covers_truth = ci_lower <= beta1 <= ci_upper
        #     covers_truth_dict[key].append(covers_truth)
        #     print(covers_truth_dict)

    # 假设 results 和 all_results 是已经定义好的变量
    # mean
    print(IV_str)
    mean_results = {}
    for key in results.keys():
        mean_results[key] = np.mean([result[key] for result in all_results])

    # variance
    var_results = {}
    for key in results.keys():
        var_results[key] = np.var([result[key] for result in all_results])

    # 将结果写入文件
    with open("IVD5.txt", "w") as file:
        file.write("mean：\n")
        for key, value in mean_results.items():
            file.write(f"{key}: {value}\n")

        file.write("\nvar：\n")
        for key, value in var_results.items():
            file.write(f"{key}: {value}\n")

    # 将 IV_str 写入另一个文件
    with open("IVDstr5.txt", "w") as file:
        for iv_value in IV_str:
            file.write(f"{iv_value}\n")

