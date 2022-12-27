import torch
from torch.utils.data import Dataset, DataLoader
import skimage.io as io
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
import warnings
import time
import torchvision
from tqdm import tqdm

# Fit encoders in order to map string labels to integers
# The fact that we are storing the scikit objects makes the encoding reproducible and consistent
# We do not need to fit each time, we store the fitted object and use its transform method when needed

# label_encoder_specific = LabelEncoder().fit(pd.read_csv('dataset_cards/cards.csv')['labels'])  # Specific
# label_encoder_category = LabelEncoder().fit(pd.read_csv('dataset_cards/cards.csv')['card type'])  # card type
# with open(os.path.join('dataset_cards', 'label_encoders.pickle'), 'wb') as file:
#     pickle.dump((label_encoder_specific, label_encoder_category), file=file)

# Deserialize the two encoders objects
with open(os.path.join('dataset_cards', 'label_encoders.pickle'), 'rb') as file:
    label_encoder_specific, label_encoder_category = pickle.load(file)


class CardsDataset(Dataset):
    """
    A PyTorch Dataset class specifically aimed at digesting and retrieving the images of the cards, together with their
    labels, which are also reliably encoded.
    """

    def __init__(self, csv_path: str, root_dir: str, transform: str,
                 card_category: bool, label_encoder: LabelEncoder,
                 subset: str = 'train') -> None:
        """
        Initialization of the CardDataset class
        :param csv_path: The path of the csv where to retrieve the necessary information about the images of the cards.
        :param root_dir: The root directory for the dataset.
        :param transform: Option between Histogram of Gradients ('hog') or Histogram of colors ('rgb_hist').
        Pass None in order to avoid transformations (normalization at pixel level is the only operation performed).
        A third option is the vector representation of the Histogram of Gradients ('vector-hog').
        :param card_category: Whether the label refers to the card category or to the specific card itself.
        :param label_encoder: The scikit-learn LabelEncoder object necessary to map the string labels to integers.
        :param subset: Choose between train ('train'), test ('test') and validation ('valid') dataset.
        """
        total_cards = pd.read_csv(csv_path)
        try:
            self.cards_table = total_cards[total_cards['data set'] == subset]
        except KeyError:
            raise KeyError('There is no subset in the dataset with the queried label')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.cards_table['labels' if not card_category else 'card type'] = \
                label_encoder.transform(self.cards_table['labels']) if not card_category else \
                label_encoder.transform(self.cards_table['card type'])
        self.root_dir = root_dir
        self.transform = transform
        self.card_category = card_category
        if self.transform == 'random-augmentation':
            self.random_aug = torch.nn.Sequential(
                torchvision.transforms.RandomPerspective(p=0.05),
                torchvision.transforms.RandomInvert(p=0.2)
            )

    def __len__(self):
        return len(self.cards_table)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        card_class = self.cards_table.iat[idx, 2] if not self.card_category else self.cards_table.iat[idx, 3]

        if self.transform == 'hog':
            representation = np.load(os.path.join(self.root_dir, 'hog',
                                                  self.cards_table.iat[idx, 1].split('.')[0] + '.npy'))
            representation = np.moveaxis(representation, 2, 0).astype(np.float32)
        elif self.transform == 'vector-hog':
            representation = np.load(os.path.join(self.root_dir, 'hog_vector',
                                                  self.cards_table.iat[idx, 1].split('.')[0] + '.npy')).\
                astype(np.float32)

        elif self.transform == 'random-augmentation' or self.transform == 'rgb_hist' or not self.transform:
            img_path = os.path.join(self.root_dir, self.cards_table.iat[idx, 1])
            image = io.imread(img_path)
            if self.transform == 'rgb_hist':
                representation, _ = np.histogramdd((np.reshape(image, (-1, 3))), bins=[np.linspace(0, 256, 6)] * 3)
                representation = (representation / representation.sum()).flatten().astype(np.float32)
            else:
                representation = torch.tensor(np.moveaxis(image, 2, 0))
                if self.transform == 'random-augmentation':
                    representation = self.random_aug(representation)
                representation = representation/255

        else:
            raise ValueError(f'There is no transformation function for \'{self.transform}\'')

        return representation, card_class


# Training function def
def train(model_, epochs: int, learning_r: float, dataloaders: list[DataLoader, DataLoader],
          torch_device: torch.device):
    model_.to(torch_device)

    optimizer = torch.optim.Adam(model_.parameters(), lr=learning_r)  # Adam optimizer initialization
    loss_history = list()  # Init loss history
    val_loss = list()  # List of validation losses
    for _ in tqdm(range(epochs), total=epochs, position=0, leave=True):
        for stage in ['train', 'valid']:
            if stage == 'train':
                model_.train()  # Set train mode for model (For Dropout)
                for x_batch, y_batch in dataloaders[stage]:  # get dataloader for specific stage (train or validation)
                    x_batch, y_batch = x_batch.to(torch_device), y_batch.to(torch_device)  # Move to device tensors
                    y_pred = model_(x_batch)  # get pred from model
                    loss = torch.nn.functional.cross_entropy(y_pred, y_batch)  # compute categorical cross-entropy
                    loss_history.append(loss.item())  # append to loss_history
                    loss.backward()  # Call backward propagation on the loss
                    optimizer.step()  # Move in the parameter space
                    optimizer.zero_grad()  # set to zero gradients
            else:
                with torch.no_grad():  # we do not need gradients when calculating validation loss and accuracy
                    loss_singleval = 0  # Initialize to 0 the loss for the single iteration on the validation set
                    accuracy = 0
                    model_.eval()  # Evaluation mode (for dropout)
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
        print(f'Training loss: {sum(loss_history[-120:])/120}')

    return model_, loss_history, val_loss
