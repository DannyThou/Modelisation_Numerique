import numpy as np
import torch
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import TensorDataset

def handle_outliers(X, num_iterations):
    X_handled = X.copy()

    for _ in range(num_iterations):
        for i in range(X_handled.shape[0]):
            row_values = X_handled[i]
            row_maxidx = row_values.argmax()
            row_mean = row_values.mean()

            X_handled[i, row_maxidx] = row_mean

    return X_handled

def oversampling(X, y):
    ros = RandomOverSampler(random_state=3075)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    return X_resampled, y_resampled

def Processing(dataset, num_iterations, do_oversampling=True):
    X_list = []
    y_list = []

    for i in range(len(dataset)):
        x, y = dataset[i]

        if torch.is_tensor(x):
            x = x.cpu().numpy()
        if torch.is_tensor(y):
            y = y.cpu().numpy()

        x = np.asarray(x).squeeze()
        y = np.asarray(y).item()

        X_list.append(x)
        y_list.append(y)

    X = np.array(X_list)
    y = np.array(y_list)

    X = handle_outliers(X, num_iterations)

    if do_oversampling:
        X, y = oversampling(X, y)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    return TensorDataset(X, y)