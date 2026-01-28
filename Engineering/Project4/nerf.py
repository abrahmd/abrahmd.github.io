import torch
import torch.nn as nn
import torch.nn.functional as F

class NeRF(nn.Module):
    def __init__(self, in_features=63, width=256):
        super().__init__()
        self.in_features = in_features
        self.x_features = in_features // 2 ## Assume x and r have same input length

        self.fc1 = nn.Linear(in_features, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, width)
        self.fc4 = nn.Linear(width, width)
        self.fc5 = nn.Linear(width + self.x_features, width)
        self.fc6 = nn.Linear(width, width)
        self.fc7 = nn.Linear(width, width)
        self.fc8 = nn.Linear(width, width)

        self.fc_density_output = nn.Linear(width, 1)

        self.fc9 = nn.Linear(width, width)
        self.fc10 = nn.Linear(width + self.in_features - self.x_features, width // 2)
        self.fc11 = nn.Linear(width//2, 3)

    def forward(self, x):
        x_input = x[..., :self.x_features]
        r_input = x[..., self.x_features:]

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)

        x = torch.cat([x, x_input], dim=-1)

        x = self.fc5(x)
        x = F.relu(x)
        x = self.fc6(x)
        x = F.relu(x)
        x = self.fc7(x)
        x = F.relu(x)
        x = self.fc8(x)

        d = self.fc_density_output(x)
        density = F.relu(d)

        x_rgb = self.fc9(x)

        x_rgb = torch.cat([x_rgb, r_input], dim=-1)

        x_rgb = self.fc10(x_rgb)
        x_rgb = F.relu(x_rgb)
        x_rgb = self.fc11(x_rgb)
        rgb = torch.sigmoid(x_rgb)

        return density, rgb