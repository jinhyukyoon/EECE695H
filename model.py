import torch.nn as nn


""" Optional conv block """
def conv_block(in_channels, out_channels):

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


""" Define your own model """
class FewShotModel(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super(FewShotModel, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 20 * 20, 1024),
        )

    def forward(self, x):
        # embedding_vector = self.encoder(x)
        out = self.encoder(x)
        dim = 1
        for d in out.size() [1:]:
            dim = dim*d
        out = out.view(-1, dim)
        embedding_vector = self.fc(out)
        return embedding_vector
