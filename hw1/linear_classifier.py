import torch
from torch import Tensor
from collections import namedtuple
from torch.utils.data import DataLoader

from .losses import ClassifierLoss
from .transforms import BiasTrick

class LinearClassifier(object):
    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # TODO:
        #  Create weights tensor of appropriate dimensions
        #  Initialize it from a normal dist with zero mean and the given std.

        self.weights = None
        # ====== YOUR CODE: ======
        weight_shape : tuple = (n_features + 1, n_classes)
        self.weights : torch.Tensor = torch.randn(weight_shape) * weight_std

        # ========================

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # TODO:
        #  Implement linear prediction.
        #  Calculate the score for each class using the weights and
        #  return the class y_pred with the highest score.

        y_pred, class_scores = None, None
        # ====== YOUR CODE: ======
        x = BiasTrick()(x)
        class_scores =  torch.matmul(x, self.weights)
        y_pred = torch.argmax(class_scores, dim=1)
        # ========================

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # TODO:
        #  calculate accuracy of prediction.
        #  Do not use an explicit loop.

        acc = None
        # ====== YOUR CODE: ======
        result : torch.Tensor = y - y_pred
        acc = (len(y) - torch.count_nonzero(result)) / float(len(y))
        # ========================

        return acc * 100

    def train(
        self,
        dl_train: DataLoader,
        dl_valid: DataLoader,
        loss_fn: ClassifierLoss,
        learn_rate=0.1,
        weight_decay=0.001,
        max_epochs=100,
    ):

        Result = namedtuple("Result", "accuracy loss")
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print("Training", end="")
        for epoch_idx in range(max_epochs):
            total_correct = 0
            average_loss = 0

            # TODO:
            #  Implement model training loop.
            #  1. At each epoch, evaluate the model on the entire training set
            #     (batch by batch) and update the weights.
            #  2. Each epoch, also evaluate on the validation set.
            #  3. Accumulate average loss and total accuracy for both sets.
            #     The train/valid_res variables should hold the average loss
            #     and accuracy per epoch.
            #  4. Don't forget to add a regularization term to the loss,
            #     using the weight_decay parameter.

            # ====== YOUR CODE: ======
            cum_loss = 0
            num_samples = 0
            
            for x_batch, y_batch in dl_train:
                # 1. Calculate Loss and Gradient
                # Assuming loss_fn returns (loss, grad) based on current weights
                batch_loss, grad = loss_fn(x_batch, y_batch, self.weights)
                
                # 2. Add Regularization (L2)
                # Loss term: (lambda / 2) * ||w||^2
                # Gradient term: lambda * w
                # Note: We assume weights are tensors/arrays compatible with the math operations
                batch_loss += 0.5 * weight_decay * (self.weights ** 2).sum()
                grad += weight_decay * self.weights
                
                # 3. Update Weights (SGD Step)
                self.weights -= learn_rate * grad
                
                # 4. Metrics Calculation
                # Predict to check accuracy
                y_pred, _ = self.predict(x_batch)
                num_correct = (y_pred == y_batch).sum()
                
                total_correct += num_correct
                cum_loss += batch_loss * len(x_batch) # Weighted by batch size
                num_samples += len(x_batch)
            
            # Store epoch results for Training
            train_res.loss.append(cum_loss / num_samples)
            train_res.accuracy.append(total_correct / num_samples)
            
            # --- Validation Loop ---
            # Evaluate on validation set (No weight updates!)
            valid_cum_loss = 0
            valid_correct = 0
            valid_samples = 0
            
            for x_valid, y_valid in dl_valid:
                # Calculate loss (Forward only)
                v_loss, _ = loss_fn(x_valid, y_valid, self.weights)
                
                # Add Regularization to validation loss for fair comparison
                v_loss += 0.5 * weight_decay * (self.weights ** 2).sum()
                
                # Calculate accuracy
                y_pred_valid, _ = self.predict(x_valid)
                valid_correct += (y_pred_valid == y_valid).sum()
                
                valid_cum_loss += v_loss * len(x_valid)
                valid_samples += len(x_valid)
                
            # Store epoch results for Validation
            valid_res.loss.append(valid_cum_loss / valid_samples)
            valid_res.accuracy.append(valid_correct / valid_samples)
            # # ========================
            print(".", end="")

        print("")
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be the first feature).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO:
        #  Convert the weights matrix into a tensor of images.
        #  The output shape should be (n_classes, C, H, W).

        # ====== YOUR CODE: ======
        w = self.weights
        if has_bias:
            w = w[1:] # Strip bias row -> (D, C)
        w_images = w.t().view(self.n_classes, *img_shape)        
        # ========================

        return w_images


def hyperparams():
    hp = dict(weight_std=0.0, learn_rate=0.0, weight_decay=0.0)

    # TODO:
    #  Manually tune the hyperparameters to get the training accuracy test
    #  to pass.
    # ====== YOUR CODE: ======
    hp['weight_std'] = 0.01
    hp['learn_rate'] = 0.2
    hp['weight_decay'] = 0.1
    # ========================

    return hp
