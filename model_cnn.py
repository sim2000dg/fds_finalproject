import torch
import numpy as np

device = torch.device('mps' if input('Metal Acceleration?  Yes/No: ').lower().strip() == 'yes' else 'cpu')


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

model = torch.nn.Sequential(Conv2dReluPool(9, 16, 3, 3),  # Number of learned parameters = 16*3*3*9 = 1'296
                            Conv2dReluPool(16, 8, 3, 3),  # Number of learned parameters = 8*9*16 = 1'152
                            torch.nn.Flatten(),
                            torch.nn.Linear(200, 150),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(0.25),
                            torch.nn.Linear(150, 100),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(0.25),
                            torch.nn.Linear(100, 53),
                            torch.nn.Softmax(1))
model.to(device)
optimizer = torch.optim.SGD(model.parameters, weight_decay=0.01, lr=0.01)  # Weight decay regulates amount of L2 reg

test = torch.tensor(np.ones((1, 9, 56, 56), dtype=np.float32), device=device)
print(model(test))

