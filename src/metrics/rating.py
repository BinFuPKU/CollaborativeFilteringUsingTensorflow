import numpy as np

# MAEs
def mean_absolute_error(ys_true, ys_pred):
    """Computes Mean Absolute Error between true and predicted arrays of ratings."""
    return 1 / ys_true.shape[0] * np.sum(np.fabs(ys_true - ys_pred))

# MSEs
def mean_squared_error(ys_true, ys_pred):
    """Computes Mean Squared Error between true and predicted arrays of ratings."""
    return 1 / ys_true.shape[0] * np.sum(np.power(ys_true - ys_pred, 2))

# RMSEs
def root_mean_squared_error(ys_true, ys_pred):
    """Computes Root Mean Squared Error between true and predicted arrays of ratings."""
    return np.sqrt(1 / ys_true.shape[0] * np.sum(np.power(ys_true - ys_pred, 2)))

def evaluate(ys_true, ys_pred, eval_metrics):
    scores = []
    for eval_metric in eval_metrics:
        if eval_metric=='mae':
            scores.append(mean_absolute_error(ys_true, ys_pred))
        elif eval_metric=='mse':
            scores.append(mean_squared_error(ys_true, ys_pred))
        elif eval_metric=='rmse':
            scores.append(root_mean_squared_error(ys_true, ys_pred))
        else:
            scores.append(None)
    return scores


if __name__=='__main__':
    ys_true, ys_pred = np.array([2.5, 1.5, 0]), np.array([1, 2, 1])
    print(mean_absolute_error(ys_true, ys_pred))
    print(mean_squared_error(ys_true, ys_pred))
    print(root_mean_squared_error(ys_true, ys_pred))

    print(eval(ys_true, ys_pred, ['mae','mse','rmse']))