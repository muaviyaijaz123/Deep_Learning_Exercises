import numpy as np
class CrossEntropyLoss:
    def __init__(self):
        self.loss = None
        self.predictions = None


    def forward(self,prediction_tensor, label_tensor):

        # label tensor
        # [1 0 0
        #  0 0 1
        #  0 1 0]  get indices and flatten to create 1 D array

        # prediction_tensor = np.array([[0.1, 0.6, 0.3],
        #                               [0.3, 0.4, 0.3],
        #                               [0.8, 0.1, 0.1]])

        true_label_indices = np.argmax(label_tensor, axis=1, keepdims=True)   # [3,1]
        #print (" LABEL TENSOR", label_tensor)
        # print("TRUE LABEL INDICES", true_label_indices)
        # print("\n")
        # print("PREDICTION TENSOR BEFORE", prediction_tensor)

        # selected predictions based on true label tensor values # output 1d array
        selected_predictions = prediction_tensor[np.arange(prediction_tensor.shape[0]), true_label_indices.flatten()]

        self.predictions = prediction_tensor
        #print("PREDICTION TENSOR AFTER", selected_predictions)
        # smallest representable number
        eeta = np.finfo(np.float64).eps

        self.loss = np.sum(-np.log(selected_predictions.flatten() + eeta))
        return self.loss

    def backward(self, label_tensor):
        eeta = np.finfo(np.float64).eps
        return - (label_tensor / (self.predictions + eeta))




