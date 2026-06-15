from torch import nn


class DQNNetwork(nn.Module):
    def __init__(self, height=16, width=30, input_channels=10):
        super().__init__()
        self.height = height
        self.width = width
        self.input_channels = input_channels
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, states):
        q_grid = self.layers(states)
        return q_grid.view(states.shape[0], self.height * self.width)
