import torch.nn as nn

class PLS_DNN(nn.Module):
    def __init__(self):
        super(PLS_DNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 128),  
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)