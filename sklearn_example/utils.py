from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, train_test_split

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]


def get_model_parameters(model) -> LogRegParams:
    """Returns the paramters of a sklearn LogisticRegression model."""
    params = [
        model.coefs_,
    ]
    return params


def set_model_params(
    model, params
):
    """Sets the parameters of a sklean LogisticRegression model."""
    print(f"These are params: {params}")
    model.coefs_ = params[0]

    return model


def set_initial_params(model, n_classes, n_features):
    """Sets initial parameters as zeros Required since model params are
    uninitialized until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer
    to sklearn.linear_model.LogisticRegression documentation for more
    information.
    """
    # Number of classes in dataset.
    # Number of features in dataset
    model.classes_ = np.array([i for i in range(n_classes)])

    model.coefs_ = np.zeros(99)
    model.n_layers_ = 100
    model.intercepts_ = np.zeros(n_classes)
    

def load_diabetes_data(random_seed, test_size:float = 0.2) -> Dataset:
    df = pd.read_csv("../data/diabetes_data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
    y = df['Diabetes_binary']
    X = df.drop(['Diabetes_binary'],axis=1)
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=random_seed)   
    print('Dimensions: \n x_train:{} \n x_test{} \n y_train{} \n y_test{}'.format(x_train.shape, x_test.shape, y_train.shape, y_test.shape))

    return (x_train, y_train), (x_test, y_test)
    
    
def shuffle(X: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle X and y."""
    rng = np.random.default_rng()
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    )
    

