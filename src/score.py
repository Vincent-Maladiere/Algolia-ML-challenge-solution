import numpy as np


def precision_k(Y_pred, Y):
    """
    Average precision @20
    For a given question, what share of our predicted users have answered it? 
    """
    assert len(Y_pred) == len(Y)
    N = len(Y_pred)
    precision_k = []
    for y_pred, y in zip(Y_pred, Y):
        p_k = len(set(y_pred) & set(y)) / len(y_pred)
        precision_k.append(p_k)
    return np.mean(precision_k)
    
    
def recall_k(Y_pred, Y):
    """
    Average recall @20
    For a given question, what share of answers belongs to the predicted users?
    """
    assert len(Y_pred) == len(Y)
    N = len(Y_pred)
    recall_k = []
    for y_pred, y in zip(Y_pred, Y):
        r_k = len(set(y_pred) & set(y)) / len(y)
        recall_k.append(r_k)
    return np.mean(recall_k)