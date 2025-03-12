import math

import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 scheduler = None,
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda
        self._scheduler = scheduler

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})

    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        # -propagate through the network
        # -calculate the loss
        # -compute gradient by backward propagation
        # -update weights
        # -return the loss
        self._optim.zero_grad()
        y_pred = self._model(x)
        loss= self._crit(y_pred, y)
        loss.backward()
        self._optim.step()
        return loss




    def val_test_step(self, x, y):

        # predict
        #self._model.eval()
        # propagate through the network and calculate the loss and predictions
        #with t.no_grad():
        y_pred = self._model(x)
        loss = self._crit(y_pred, y)
        return loss, y_pred
        # return the loss and the predictions

    def train_epoch(self):
        # set training mode
        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it
        # set training mode
        self._model.train()
        total_loss = 0
        for x, y in self._train_dl:
            if self._cuda:
                x, y = x.cuda(), y.cuda()
            loss = self.train_step(x, y)
            total_loss += loss.item()
        avg_loss = total_loss / len(self._train_dl)
        return avg_loss

    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore.
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics
        self._model.eval()
        with t.no_grad():
            total_loss = 0
            y_true = []
            y_pred = []
            for x, y in self._val_test_dl:
                if self._cuda:
                    x, y = x.cuda(), y.cuda()
                loss, pred = self.val_test_step(x, y)
                total_loss += loss.item()
                y_true.append(y)
                y_pred.append(pred)
            avg_loss = total_loss / len(self._val_test_dl)
            y_true = t.cat(y_true)
            y_pred = t.cat(y_pred)

            return avg_loss, y_true, y_pred

    import math
    from sklearn.metrics import f1_score

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0

        train_losses = []
        val_losses = []
        counter = 0

        best_f1 = -math.inf  # Track best F1 score across epochs
        best_val_loss = float('inf')  # Track best validation loss
        patience_counter = 0  # Track epochs without improvement

        while True:
            # Stop training if max epochs reached
            if epochs > 0 and counter >= epochs:
                break

            # Train for one epoch
            train_loss = self.train_epoch()
            val_loss, y_true, y_pred = self.val_test()

            # Compute F1 Score
            F1 = f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy().round(), average='micro')

            # Store losses
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f'Epoch {counter}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, F1: {F1:.4f}')

            # Save the best model based on F1 score
            if F1 > best_f1:
                best_f1 = F1
                self.save_checkpoint(counter)

            # Early stopping based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0  # Reset patience if validation improves
            else:
                patience_counter += 1  # Increase patience counter

            # Stop training if patience exceeds limit
            if self._early_stopping_patience > 0 and patience_counter >= self._early_stopping_patience:
                print("Early stopping triggered.")
                break

            # Update learning rate scheduler (ReduceLROnPlateau requires validation loss)
            if self._scheduler is not None:
                self._scheduler.step(val_loss)

            counter += 1

        return train_losses, val_losses






