import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores
    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops

    predictions = predictions.copy()
    if len(predictions.shape) == 1:
        predictions -= np.max(predictions) # https://cs231n.github.io/linear-classify/#softmax
        probs = np.exp(predictions) / np.sum(np.exp(predictions))
    else:
        predictions = list(map(lambda x: x - np.max(x), predictions))
        probs = np.array(list(map(lambda x: x / np.sum(x), np.exp(predictions))))

    return probs
    raise Exception("Not implemented!")


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss
    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops

    if len(probs.shape) == 1:
        target_probs = probs[target_index]
        size = 1
    else:
        target_probs = probs[np.arange(target_index.shape[0]), target_index.flatten()]
        size = target_index.shape[0]
    return np.sum(-np.log(target_probs)) / size
    raise Exception("Not implemented!")


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient
    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    
    predictions_copy = predictions.copy()
    probs = softmax(predictions_copy)
    loss = cross_entropy_loss(probs, target_index)
    dprediction = probs
    if (len(predictions.shape) == 1):
        dprediction[target_index] -= 1
    else:
        batch_size = np.arange(target_index.shape[0])
        dprediction[batch_size, target_index.flatten()] -= 1
        dprediction = dprediction / target_index.shape[0]
    return loss, dprediction
    raise Exception("Not implemented!")


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient
    Arguments:
      W, np array - weights
      reg_strength - float value
    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    
    loss = reg_strength * np.sum(W*W)
    # считаем градиент регуляризатора https://thefaq.ru/gradient-reguljarizatora-otnositelno-vesov-v-reguljarizacii-l2/ 
    grad = reg_strength * W * 2
    
    return loss, grad
    raise Exception("Not implemented!")
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W
    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes
    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss
    '''
    predictions = np.dot(X, W)

    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    
    loss, dprediction = softmax_with_cross_entropy(predictions, target_index)
    dW = X.transpose().dot(dprediction)

    return loss, dW
    raise Exception("Not implemented!")


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''
        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)
        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            '''sections [ 300  600  900 1200 1500 1800 2100 2400 2700 3000 3300 3600 3900 4200
                          4500 4800 5100 5400 5700 6000 6300 6600 6900 7200 7500 7800 8100 8400 8700]'''
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)
            
            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            
            loss = 0
            for batch_index in batches_indices:
                X_batch = X[batch_index]
                y_batch = y[batch_index]
                loss_softmax, dW_softmax = linear_softmax(X_batch, self.W, y_batch)
                loss_l2, dW_l2 = l2_regularization(self.W, reg)
                loss += (loss_softmax + loss_l2)
                loss /= X_batch.shape[0]
                loss_history.append(loss)
                self.W -= learning_rate * (dW_softmax + dW_l2)

            # end
            print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history
        raise Exception("Not implemented!")

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)
        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.argmax(np.dot(X, self.W), axis = 1)
        return y_pred
        raise Exception("Not implemented!")



                
                                                          

            

                
