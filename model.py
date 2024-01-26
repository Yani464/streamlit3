import torch.nn as nn

class Muffler(nn.Module):
        def __init__(self):
            super(Muffler, self).__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

            self.encoder = nn.Sequential(
                # nn.Flatten(),
                nn.Conv2d(
                    in_channels=1,
                    out_channels=128, 
                    kernel_size=4, 
                    stride=1, 
                    padding=2),
                nn.BatchNorm2d(128),
                nn.Dropout(),
                nn.SELU(),
                nn.Conv2d(128, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.Dropout(),
                nn.SELU(),
                nn.Conv2d(64, 16, 2, 1),
                nn.Dropout(),
                nn.SELU()
                # nn.AdaptiveAvgPool2d(2)
            )

            self.decoder = nn.Sequential(
                nn.Conv2d(16, 64, 2, 1),
                nn.BatchNorm2d(64),
                nn.Dropout(),
                nn.SELU(),
                nn.Conv2d(64, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.Dropout(),
                nn.SELU(),
                nn.Conv2d(128, 1, 4, 1, 2),
                nn.Sigmoid()
            )

        def forward(self, x):
            latent = self.encoder(x)
            reconstructed = self.decoder(latent)
            return reconstructed
        


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x        