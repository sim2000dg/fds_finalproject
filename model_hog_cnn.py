import matplotlib.pyplot as plt
import torch
import numpy as np
from loader import CardsDataset
import pickle
import os
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(model_, epochs: int, learning_r: float, dataloaders: list[DataLoader, DataLoader],
          torch_device: torch.device):
    model_.to(torch_device)
    # Weight decay regulates amount of L2 reg
    optimizer = torch.optim.SGD(model_.parameters(), lr=learning_r)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: 0.01 if x == 20 else 1)

    # optimizer = torch.optim.Adam(model_.parameters(), lr=learning_r)  # Adam optimizer initialization
    loss_history = list()  # Init loss history
    val_loss = list()  # List of validation losses
    for _ in tqdm(range(epochs), total=epochs):
        for stage in ['train', 'valid']:
            if stage == 'train':
                model_.train()  # Set train mode for model (For BatchNorm and Dropout)
                for x_batch, y_batch in dataloaders[stage]:  # get dataloader for specific stage (train or validation)
                    x_batch, y_batch = x_batch.to(torch_device), y_batch.to(torch_device)  # Move to device tensors
                    y_pred = model_(x_batch)  # get pred from model
                    loss = torch.nn.functional.cross_entropy(y_pred, y_batch)  # compute categorical cross-entropy
                    loss_history.append(loss.item())  # append to loss_history
                    loss.backward()  # Call backward propagation on the loss
                    optimizer.step()  # Move into parameter space
                    optimizer.zero_grad()  # set to zero gradients
            else:
                with torch.no_grad():  # we do not need gradients when calculating validation loss and accuracy
                    loss_singleval = 0  # Initialize to 0 the loss for the single iteration on the validation set
                    accuracy = 0
                    model_.eval()
                    for x_batch, y_batch in dataloaders[stage]:  # Access the dataloader for validation
                        # Move the tensors to right device
                        x_batch, y_batch = x_batch.to(torch_device), y_batch.to(torch_device)
                        y_pred = model_(x_batch)  # Get prediction from validation
                        # add loss for single batch from validation
                        loss_singleval += torch.nn.functional.cross_entropy(y_pred, y_batch).item()
                        # Add mean accuracy from batch of validation
                        accuracy += (torch.max(y_pred, 1)[1] == y_batch).float().sum()/len(y_batch)
                    # append mean validation loss (mean over the number of batches)
                    val_loss.append(loss_singleval/len(dataloaders[stage]))

        print(f'Validation loss: {val_loss[-1]}')
        print(f'Accuracy on validation: {accuracy/len(dataloaders[stage])}')

    return model_, loss_history, val_loss


if __name__ == '__main__':
    device = torch.device(
        'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

    model_hog_cnn = torch.nn.Sequential(torch.nn.Conv2d(9, 16, 3, padding='valid'),
                                        torch.nn.ReLU(),
                                        torch.nn.Conv2d(16, 32, 3, padding='valid'),
                                        torch.nn.ReLU(),
                                        torch.nn.Conv2d(32, 32, 3, padding='valid'),
                                        torch.nn.ReLU(),
                                        torch.nn.Conv2d(32, 32, 3, padding='valid'),
                                        torch.nn.ReLU(),
                                        torch.nn.Flatten(),
                                        torch.nn.Linear(73728, 1000),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(1000, 1000),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(1000, 1000),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(1000, 500),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(500, 100),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(100, 53),
                                        torch.nn.Softmax(1))

    with open(os.path.join('dataset_cards', 'label_encoders.pickle'), 'rb') as file:
        label_encoder_specific, label_encoder_category = pickle.load(file)

    training_hog = CardsDataset(csv_path=os.path.join('dataset_cards', 'cards.csv'), root_dir='dataset_cards',
                                transform='hog', vector_hog=False, card_category=False,
                                label_encoder=label_encoder_specific, subset='train')
    train_dataloader = DataLoader(training_hog, batch_size=32,
                                  shuffle=True, num_workers=5, persistent_workers=True)

    valid_hog = CardsDataset(csv_path=os.path.join('dataset_cards', 'cards.csv'), root_dir='dataset_cards',
                             transform='hog', vector_hog=False, card_category=False,
                             label_encoder=label_encoder_specific, subset='valid')

    valid_dataloader = DataLoader(valid_hog, batch_size=256,
                                  shuffle=False, num_workers=5)

    dataloaders = {'train': train_dataloader,
                   'valid': valid_dataloader}

    model, history, acc_history = train(model_hog_cnn, 100, 1e-4, dataloaders, device)

    # Running average of the loss to isolate trend
    history = np.convolve(np.array(history), np.ones(50) / 50, mode='valid')
    plt.plot(np.arange(1, len(history) + 1), history)
    plt.show()

# valid_value = 0
#         for x_batch, y_batch in valid_dataloader:
#             x_batch, y_batch = x_batch.to(device), y_batch.to(device)
#             y_pred = model_(x_batch)
#             valid_value += torch.nn.functional.cross_entropy(y_pred, y_batch).item()
#         valid_loss.append(valid_value/len(valid_dataloader))
