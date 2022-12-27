import torch
import torchvision
from torch.utils.data import DataLoader
from utilities import CardsDataset
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

if __name__ == '__main__':

    device = torch.device(
        'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

    model = torchvision.models.vgg16(weights='DEFAULT')

    model.classifier[-2] = torch.nn.Linear(4096, 53)  # Change linear (number and type of classes is different)
    model.classifier[-1] = torch.nn.Softmax(1)  # Add Softmax normalization
    model.classifier[2] = torch.nn.Dropout(p=0.2)  # Moderate Dropout

    # Load state of the model
    model.load_state_dict(torch.load('transfer_vgg_cards_conv_train_100.pt'))

    # Get scikit object to decode labels
    with open(os.path.join('dataset_cards', 'label_encoders.pickle'), 'rb') as file:
        label_encoder_specific, label_encoder_category = pickle.load(file)

    # Dataset and Dataloader for test set
    test_set = CardsDataset(csv_path=os.path.join('dataset_cards', 'cards.csv'), root_dir='dataset_cards',
                            transform=None, card_category=False,
                            label_encoder=label_encoder_specific, subset='test')

    test_dataloader = DataLoader(test_set, batch_size=64,
                                 shuffle=False, num_workers=1)

    check = list()  # Element inside True if correct pred, False otherwise
    pred_indexes = list()
    truth = list()
    model.eval()  # Evaluation mode
    with torch.no_grad():  # We do not need backpropagation for inference
        for x_test, y_test in test_dataloader:
            y_pred = model(x_test)
            check.extend((torch.max(y_pred, 1)[1] == y_test).tolist())
            pred_indexes.extend(torch.max(y_pred, 1)[1].tolist())
            truth.extend(y_test.tolist())

    # Read test data
    table_data = pd.read_csv('dataset_cards/cards.csv')
    test_data = table_data[table_data['data set'] == 'test']
    test_data['pred_label'] = label_encoder_specific.classes_[pred_indexes]  # add column for pred labels
    misclassified = test_data[test_data.pred_label != test_data.labels]  # Filter misclassified
    print(f'Accuracy:{sum(check)/len(check)}')

    # Plot confusion matrix
    plt.rcParams.update({'font.size': 13,
                         'figure.figsize': (15, 15)})
    ConfusionMatrixDisplay.from_predictions(label_encoder_specific.classes_[truth],
                                            label_encoder_specific.classes_[pred_indexes],
                                            cmap='Reds', include_values=False, xticks_rotation='vertical')
    plt.title('Confusion matrix on test data (transfer learning with VGG16)',
              fontdict={'size': 30}, pad=110)
    plt.show()
