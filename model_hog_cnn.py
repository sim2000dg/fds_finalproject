import matplotlib.pyplot as plt
import torch
import numpy as np
from loader import CardsDataset
import pickle
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

# HOG - CNN model

model_hog_cnn = torch.nn.Sequential(torch.nn.Conv2d(9, 16, 3, padding='valid'),
                                    torch.nn.ReLU(),
                                    torch.nn.Conv2d(16, 32, 3, padding='valid'),
                                    torch.nn.ReLU(),
                                    torch.nn.Conv2d(32, 32, 3, padding='valid'),
                                    torch.nn.ReLU(),
                                    torch.nn.Flatten(),
                                    torch.nn.Linear(80000, 1000),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(1000, 500),
                                    torch.nn.ReLU(),
                                    # torch.nn.Dropout(0.25),
                                    torch.nn.Linear(500, 100),
                                    torch.nn.ReLU(),
                                    # torch.nn.Dropout(0.25),
                                    torch.nn.Linear(100, 53),
                                    torch.nn.Softmax(1))


def train(model_, epochs, decay, learning_r, dataloaders, torch_device):
    model_.to(torch_device)
    # Weight decay regulates amount of L2 reg
    # optimizer = torch.optim.SGD(model_.parameters(), weight_decay=decay, lr=learning_r)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: 0.01 if x == 20 else 1)

    optimizer = torch.optim.Adam(model_.parameters(), lr=3e-4)
    loss_history = list()
    accuracy_history = list()
    val_loss = list()
    for _ in tqdm(range(epochs), total=epochs):
        for stage in ['train', 'valid']:
            if stage == 'train':
                for x_batch, y_batch in dataloaders[stage]:
                    x_batch, y_batch = x_batch.to(torch_device), y_batch.to(torch_device)
                    y_pred = model_(x_batch)
                    loss = torch.nn.functional.cross_entropy(y_pred, y_batch)
                    loss_history.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                with torch.no_grad():
                    for x_batch, y_batch in dataloaders[stage]:
                        x_batch, y_batch = x_batch.to(torch_device), y_batch.to(torch_device)
                        y_pred = model_(x_batch)
                        val_loss.append(torch.nn.functional.cross_entropy(y_pred, y_batch).item())

        print(f'Validation loss: {val_loss[-1]}')
        # scheduler.step()
    return model_, loss_history, accuracy_history


if __name__ == '__main__':
    with open(os.path.join('dataset_cards', 'label_encoders.pickle'), 'rb') as file:
        label_encoder_specific, label_encoder_category = pickle.load(file)

    training_hog = CardsDataset(csv_path=os.path.join('dataset_cards', 'cards.csv'), root_dir='dataset_cards',
                                transform='hog', card_category=False,
                                label_encoder=label_encoder_specific, subset='train')
    train_dataloader = DataLoader(training_hog, batch_size=128,
                                  shuffle=True, num_workers=5, persistent_workers=True)

    valid_hog = CardsDataset(csv_path=os.path.join('dataset_cards', 'cards.csv'), root_dir='dataset_cards',
                             transform='hog', card_category=False,
                             label_encoder=label_encoder_specific, subset='valid')

    valid_dataloader = DataLoader(valid_hog, batch_size=64,
                                  shuffle=False, num_workers=1)

    dataloaders = {'train': train_dataloader,
                   'valid': valid_dataloader}

    model, history, acc_history = train(model_hog_cnn, 100, 0, 1, dataloaders, device)

    history = np.convolve(np.array(history), np.ones(50) / 50, mode='valid')
    plt.plot(np.arange(1, len(history) + 1), history)
    plt.show()

# valid_value = 0
#         for x_batch, y_batch in valid_dataloader:
#             x_batch, y_batch = x_batch.to(device), y_batch.to(device)
#             y_pred = model_(x_batch)
#             valid_value += torch.nn.functional.cross_entropy(y_pred, y_batch).item()
#         valid_loss.append(valid_value/len(valid_dataloader))
