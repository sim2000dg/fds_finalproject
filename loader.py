import torch
from torch.utils.data import Dataset, DataLoader
from skimage.feature import hog
import skimage.io as io
import pandas as pd
import os
import matplotlib.pyplot as plt


class CardsDataset(Dataset):
    def __init__(self, csv_path, root_dir, hog_transform: bool, card_category: bool, subset: str = 'train'):
        total_cards = pd.read_csv(csv_path)
        try:
            self.cards_table = total_cards[total_cards['data set'] == subset]
        except KeyError:
            raise KeyError('There is no subset in the dataset with the queried label')
        self.root_dir = root_dir
        self.transform = hog_transform
        self.card_category = card_category

    def __len__(self):
        return len(self.cards_table)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.cards_table.iat[idx, 1])
        image = io.imread(img_path)
        card_class = self.cards_table.iat[idx, 2] if self.card_category else self.cards_table.iat[idx, 3]

        if self.transform:
            image = hog(image, channel_axis=2, block_norm='L2', cells_per_block=(1, 1), pixels_per_cell=(7, 7),
                        visualize=True)[1]

        return image, card_class


if __name__ == '__main__':
    training_hog = CardsDataset('dataset_cards/cards.csv', 'dataset_cards', True, False, 'train')
    train_dataloader = DataLoader(training_hog, batch_size=16, shuffle=True)
    image_batch, _ = next(iter(train_dataloader))
    plt.imshow(image_batch[0])
    plt.show()

