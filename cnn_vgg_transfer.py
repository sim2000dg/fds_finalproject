import torch
import torchvision
from torch.utils.data import DataLoader
from loader import CardsDataset, train
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    device = torch.device(
        'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

    model = torchvision.models.vgg16(weights='DEFAULT')

    model.classifier[-2] = torch.nn.Linear(4096, 53)  # Change linear (number and type of classes is different)
    model.classifier[-1] = torch.nn.Softmax(1)  # Add Softmax normalization
    model.classifier[2] = torch.nn.Dropout(p=0.2)  # Moderate Dropout

    # model.load_state_dict(torch.load('transfer_vgg_cards.pt'))

    # Get scikit object to encode labels
    with open(os.path.join('dataset_cards', 'label_encoders.pickle'), 'rb') as file:
        label_encoder_specific, label_encoder_category = pickle.load(file)

    # Instantiate Dataset and Dataloader objects to use in order to get observations during training
    training_data = CardsDataset(csv_path=os.path.join('dataset_cards', 'cards.csv'), root_dir='dataset_cards',
                                 transform='random-augmentation', card_category=False,
                                 label_encoder=label_encoder_specific, subset='train')
    train_dataloader = DataLoader(training_data, batch_size=64,
                                  shuffle=True, num_workers=5, persistent_workers=True)

    # Do the same for validation set
    valid_data = CardsDataset(csv_path=os.path.join('dataset_cards', 'cards.csv'),
                              root_dir='dataset_cards',
                              transform=None, card_category=False,
                              label_encoder=label_encoder_specific, subset='valid')

    valid_dataloader = DataLoader(valid_data, batch_size=256,
                                  shuffle=False, num_workers=2,
                                  persistent_workers=True)

    # Build a dictionary containing the two dataloaders (this needs to be passed to the train function)
    dataloaders = {'train': train_dataloader,
                   'valid': valid_dataloader}

    # Train the model
    model, history, val_loss = train(model, 100, 1e-5, dataloaders, device)
    # Save the weights
    torch.save(model.state_dict(), 'transfer_vgg_cards_conv_train_100.pt')

    # Running average of the loss to isolate trend
    history = np.convolve(np.array(history), np.ones(50) / 50, mode='valid')
    plt.plot(np.arange(1, len(history) + 1), history)
    plt.title('Smoothed categorical cross-entropy loss w.r.t. training iterations\n'
              'Unfrozen convolutional layers')
    plt.xlabel('Batches')
    plt.ylabel('Smoothed cross-entropy')
    plt.show()

    # Save training data for analysis and visualization
    with open('train_data', 'wb') as file:
        pickle.dump([history, val_loss], file)
