import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy

import numpy as np


class RubikPolicy(nn.Module):
    def __init__(self, state_shape, action_size):
        super(RubikPolicy, self).__init__()

        # First 2D convolutional layer, taking in 1 input channel (image),
        # outputting 32 convolutional features, with a square kernel size of 3
        # self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # Second 2D convolutional layer, taking in the 32 input layers,
        # outputting 64 convolutional features, with a square kernel size of 3
        # self.conv2 = nn.Conv2d(32, 64, 3, 1)

        # Designed to ensure that adjacent pixels are either all 0s or all active
        # with an input probability
        # self.dropout1 = nn.Dropout2d(0.25)
        # self.dropout2 = nn.Dropout2d(0.5)
        self.linears = []
        self.flatten = nn.Flatten(start_dim=0, end_dim=-1)
        self.l1 = nn.Linear(state_shape[0] * state_shape[1], 128)
        self.l2 = nn.Linear(128, 256)
        self.l3 = nn.Linear(256, 128)
        # self.l4 = nn.Linear(128, 128)
        # self.l5 = nn.Linear(128, 64)
        # self.l6 = nn.Linear(64, 128)
        # self.l7 = nn.Linear(128, 512)
        # self.l8 = nn.Linear(512, 256)
        # self.l9 = nn.Linear(256, 128)
        self.l10 = nn.Linear(128, action_size)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        # x = F.relu(self.l4(x))
        # x = F.relu(self.l5(x))
        # x = F.relu(self.l6(x))
        # x = F.relu(self.l7(x))
        # x = F.relu(self.l8(x))
        # x = F.relu(self.l9(x))
        x = self.l10(x)
        return x

    def fit(self, tr_x, tr_y, batch_size, epochs=1, learning_rate=0.001, verbose=0):
        train_loader = DataLoader(RubikDataset(tr_x, tr_y), batch_size=batch_size, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in train_loader:
                inputs, targets = batch
                for inp, target in zip(inputs, targets):
                    epoch_loss += self._fit_step(inp, target, optimizer, criterion)
            if verbose > 0:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss}')
            losses.append(epoch_loss)
        return np.array(losses)

    def single_fit(self, x, y, epochs=1, learning_rate=0.001, verbose=0):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        losses = list()
        for e in range(epochs):
            f_x = from_numpy(x.astype(np.single))
            f_y = from_numpy(y.astype(np.single))
            l = self._fit_step(f_x, f_y, optimizer, criterion)
            losses.append(l)
        return np.mean(np.array(losses))

    def _fit_step(self, x, y, optimizer, criterion):
        optimizer.zero_grad()
        output = self(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    def predict(self, d):
        return self(from_numpy(d.astype(np.single))).detach().numpy()

    def mass_predict(self, multirow):
        return np.array([self.predict(single_datarow) for single_datarow in multirow])

    def summary(self):
        print(self)


# dataset definition
class RubikDataset(Dataset):
    # load the dataset
    def __init__(self, states, rewards):
        self.X = from_numpy(states.astype(np.single))
        self.y = from_numpy(rewards.astype(np.single))

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]