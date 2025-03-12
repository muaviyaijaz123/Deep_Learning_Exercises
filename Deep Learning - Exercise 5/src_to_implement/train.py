import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split


# load the data from the csv file and perform a train-test-split
data=pd.read_csv('data.csv',sep=';')
train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
# TODO

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
train_dl=t.utils.data.DataLoader(ChallengeDataset(train_data, 'train'), batch_size=32, shuffle=True)
val_dl=t.utils.data.DataLoader(ChallengeDataset(val_data, 'val'), batch_size=32, shuffle=False)
# TODO

# create an instance of our ResNet model
# TODO
res= model.ResNet()
criterion = t.nn.BCELoss()
optimizer = t.optim.Adam(res.parameters(), lr=0.0001, weight_decay= 1e-4, amsgrad= True)
scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=3)
trainer = Trainer(res,criterion,optimizer, scheduler= scheduler, cuda=True,early_stopping_patience=-1,train_dl=train_dl,val_test_dl=val_dl)
# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion
# TODO

# go, go, go... call fit on trainer
res = trainer.fit(epochs=100)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')