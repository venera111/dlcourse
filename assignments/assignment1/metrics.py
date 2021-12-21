import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification
    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0
    
    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    true = (prediction == ground_truth).astype(int)
    false = (prediction != ground_truth).astype(int)
    positives = prediction
    negatives = 1 - prediction
    tp = (true * positives).sum()
    tn = (true * negatives).sum()
    fp = (false * positives).sum()
    fn = (false * negatives).sum()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return precision, recall, f1, accuracy

def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification
    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy

    true_positives = 0
    for p in range(len(prediction)):
        if prediction[p] == ground_truth[p]:
            true_positives += 1
    accuracy = (true_positives) / len(prediction)

    return accuracy