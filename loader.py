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

    def __init__(self, csv_path: str, root_dir: str, transform: str, vector_hog: bool,
                 card_category: bool, label_encoder: LabelEncoder,
                 subset: str = 'train') -> None:
        """
        Initialization of the CardDataset class
        :param csv_path: The path of the csv where to retrieve the necessary information about the images of the cards.
        :param root_dir: The root directory for the dataset.
        :param transform: Option between Histogram of Gradients ('hog') or Histogram of colors ('rgb_hist').
        Pass None in order to avoid transformations (normalization at pixel level is the only operation performed).
        :param vector_hog: if hog, whether to use vector representation or tensor representation.
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
        self.vector_hog = vector_hog
        self.transform = transform
        self.card_category = card_category

    def __len__(self):
        return len(self.cards_table)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        card_class = self.cards_table.iat[idx, 2] if not self.card_category else self.cards_table.iat[idx, 3]

        if self.transform == 'hog' and not self.vector_hog:
            representation = np.load(os.path.join(self.root_dir, 'hog',
                                                  self.cards_table.iat[idx, 1].split('.')[0] + '.npy'))
            representation = np.moveaxis(representation, 2, 0)
        elif self.transform == 'hog' and not self.vector_hog:
            representation = np.load(os.path.join(self.root_dir, 'hog_vector',
                                                  self.cards_table.iat[idx, 1].split('.')[0] + '.npy'))

        elif self.transform is None or self.transform == 'rgb_hist':
            img_path = os.path.join(self.root_dir, self.cards_table.iat[idx, 1])
            image = io.imread(img_path)
            if self.transform == 'rgb_hist':
                representation, _ = np.histogramdd((np.reshape(image, (-1, 3))), bins=[np.linspace(0, 256, 6)] * 3)
                representation = (representation / representation.sum()).flatten()
            else:
                representation = np.moveaxis(image/255.0, 2, 0)

        else:
            raise ValueError(f'There is no transformation function for \'{self.transform}\'')

        return representation.astype(np.float32), card_class


if __name__ == '__main__':
    training_hog = CardsDataset(csv_path=os.path.join('dataset_cards', 'cards.csv'), root_dir='dataset_cards',
                                transform=None, card_category=False,
                                label_encoder=label_encoder_specific, subset='train')
    train_dataloader = DataLoader(training_hog, batch_size=16, shuffle=True)
    start = time.time()
    for x in train_dataloader:
        pass
    print(time.time() - start)
