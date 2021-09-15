from sklearn.metrics import confusion_matrix


def get_metrics(y, y_pred):
    """Calculate and return PPV, NPV, Sensitivity, and Specificity
    Args:
        y (list): (n_samples,)
        y_pred (list): (n_samples,)
    Returns:
        float: ppv
        float: npv
        float: specificity
        float: sensitivity
    """
    cm = confusion_matrix(y, y_pred)
    tn = cm[0][0]
    fn = cm[1][0]
    tp = cm[1][1]
    fp = cm[0][1]
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    specificity = tp / (tp + fn)
    sensitivity = tn / (tn + fp)

    return {'ppv': ppv,
            'npv': npv,
            'specificity': specificity,
            'sensitivity': sensitivity}
