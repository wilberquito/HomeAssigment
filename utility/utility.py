import numpy as np


def get_error_indexes(y_test, y_pred):
    errors = abs(y_test - y_pred)
    errors = np.where(errors > 0)
    return errors
