import matplotlib.pyplot as plt
import torch
import numpy as np
from loader import CardsDataset
import pickle
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')


class Conv2dReluPool(torch.nn.Module):
    def __init__(self, channel_in: int, filters: int, kernel_shape: int | tuple[int, int],
                 pool_shape: int | tuple[int, int]):
        super().__init__()
        self.conv = torch.nn.Conv2d(channel_in, filters, kernel_shape)
        self.pool = torch.nn.MaxPool2d(pool_shape)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.pool(self.relu(self.conv(x)))


# HOG + CNN model

model = torch.nn.Sequential(Conv2dReluPool(9, 16, 3, 3),
                            Conv2dReluPool(16, 8, 3, 3),
                            torch.nn.Flatten(),
                            torch.nn.Linear(200, 150),
                            torch.nn.ReLU(),
                            # torch.nn.Dropout(0.25),
                            torch.nn.Linear(150, 100),
                            torch.nn.ReLU(),
                            # torch.nn.Dropout(0.25),
                            torch.nn.Linear(100, 14),
                            torch.nn.ReLU(),
                            torch.nn.Linear(14, 5),
                            torch.nn.Softmax(1))
model.to(device)


def train(model_, epochs, decay, learning_r, dataloader):
    optimizer = torch.optim.SGD(model_.parameters(),
                                lr=0.1)  # Weight decay regulates amount of L2 reg
    # optimizer = torch.optim.Adam(model_.parameters())
    loss_history = list()
    accuracy_history = list()
    for _ in tqdm(range(epochs), total=epochs):
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model_(x_batch)
            loss = torch.nn.functional.cross_entropy(y_pred, y_batch)
            loss_history.append(loss.item())
            accuracy_history.append((torch.max(y_pred.data, 1)[0] == y_batch).float().mean())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    return model, loss_history, accuracy_history


if __name__ == '__main__':
    with open(os.path.join('dataset_cards', 'label_encoders.pickle'), 'rb') as file:
        label_encoder_specific, label_encoder_category = pickle.load(file)

    training_hog = CardsDataset(csv_path=os.path.join('dataset_cards', 'cards.csv'), root_dir='dataset_cards',
                                transform='hog', card_category=False,
                                label_encoder=label_encoder_specific, subset='train')
    train_dataloader = DataLoader(training_hog, batch_size=32,
                                  shuffle=True, num_workers=5, persistent_workers=True)

    model, history, acc_history = train(model, 30, 0.1, 0.01, train_dataloader)

    plt.plot(np.arange(1, len(history) + 1), history)
    plt.show()
