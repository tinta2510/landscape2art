import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, act_fn=None):

        super().__init__()
        layers = [
            # When Conv is followed by BatchNorm, SET Conv's bias = False
            nn.Conv2d(in_channels, out_channels, stride=stride, 
                      padding=kernel_size // 2, kernel_size=kernel_size, bias=False),
            nn.InstanceNorm2d(out_channels)
        ]
        if act_fn:
            layers.append(act_fn)
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.conv(x)
    
class ConvTransposeBlock(nn.Module):
    """
    Convolutional Transpose Block: ConvTranspose2d + BatchNorm + ReLU
    - H_out = (H_in - 1) * stride - 2 * padding + kernel_size + output_padding
    - W_out = (W_in - 1) * stride - 2 * padding + kernel_size + output_padding
    """
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, act_fn=None):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, stride=stride, 
                               padding=kernel_size // 2, output_padding=stride - 1, 
                               kernel_size=kernel_size, bias=False),
            nn.InstanceNorm2d(out_channels)
        ]
        if act_fn:
            layers.append(act_fn)
        self.conv_transpose = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.conv_transpose(x)
    
class ConvAutoencoder(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 hidden_channels=[],
                 kernel_size=3,
                 stride=2):
        super().__init__()

        encoder_layers = [
            ConvBlock(in_channel, hidden_channels[0], kernel_size=kernel_size,
                      stride=stride, act_fn=nn.ReLU(True)),
        ]
        for i in range(len(hidden_channels)-1):
            encoder_layers.append(
                ConvBlock(hidden_channels[i], hidden_channels[i+1],
                          kernel_size=kernel_size, stride=stride, act_fn=nn.ReLU(True))
            )
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(len(hidden_channels)-1, 0, -1):
            decoder_layers.append(
                ConvTransposeBlock(hidden_channels[i], hidden_channels[i-1],
                                   kernel_size=kernel_size, stride=stride, act_fn=nn.ReLU(True))
            )
        decoder_layers.append(
            ConvTransposeBlock(hidden_channels[0], out_channel,
                               kernel_size=kernel_size, stride=stride, act_fn=nn.Sigmoid())
        )
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
