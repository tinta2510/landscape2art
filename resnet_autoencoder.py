import torch.nn as nn
from conv_autoencoder import ConvBlock, ConvTransposeBlock    
    
class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, act_fn=None, shortcut=None):
        """
        Basic residual block for ResNet with two 3x3 conv layers and a skip connection.
        """
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, stride=stride, 
                              kernel_size=kernel_size, act_fn=nn.ReLU(True))
        self.conv2 = ConvBlock(out_channels, out_channels, stride=1, 
                               kernel_size=kernel_size, act_fn=None)
        
        # Create shortcut to match dimensions when needed
        if shortcut is not None:
            self.shortcut = shortcut
        elif stride != 1 or in_channels != out_channels:
            self.shortcut = ConvBlock(in_channels, out_channels, stride=stride, 
                                      kernel_size=1, act_fn=None)
        else:
            self.shortcut = nn.Identity()
        
        if act_fn:
            self.act_fn = act_fn    
    
    def forward(self, x):
        out = self.conv(x)
        out = self.conv2(out) + self.shortcut(x)
        out = self.act_fn(out)
        return out
    
class ResnetTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, act_fn=None, shortcut=None):
        """
        Residual block with ConvTranspose2d for upsampling.
        """
        super().__init__()
        self.conv_transpose = ConvTransposeBlock(in_channels, out_channels, stride=stride, 
                                                 kernel_size=kernel_size, act_fn=nn.ReLU(True))
        self.conv2 = ConvBlock(out_channels, out_channels, stride=1, 
                               kernel_size=kernel_size, act_fn=None)
        
        if shortcut is not None:
            self.shortcut = shortcut
        elif stride != 1 or in_channels != out_channels:
            # Need to adjust dimensions for skip connection (upsampling case)
            self.shortcut = ConvTransposeBlock(in_channels, out_channels, stride=stride, 
                                               kernel_size=1, act_fn=None)
        else:
            self.shortcut = nn.Identity()
            
        
        if act_fn:
            self.act_fn = act_fn
    
    def forward(self, x):
        out = self.conv_transpose(x)
        out = self.conv2(out) + self.shortcut(x)
        out = self.act_fn(out)
        return out
    
class ResnetAutoencoder(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 hidden_channels=[],
                 kernel_size=3,
                 stride=2):
        super().__init__()

        encoder_layers = [
            ResnetBlock(in_channel, hidden_channels[0], kernel_size=kernel_size,
                        stride=stride, act_fn=nn.ReLU(True)),
        ]
        for i in range(len(hidden_channels)-1):
            encoder_layers.append(
                ResnetBlock(hidden_channels[i], hidden_channels[i+1],
                            kernel_size=kernel_size, stride=stride, act_fn=nn.ReLU(True))
            )
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(len(hidden_channels)-1, 0, -1):
            decoder_layers.append(
                ResnetTransposeBlock(hidden_channels[i], hidden_channels[i-1],
                                     kernel_size=kernel_size, stride=stride, act_fn=nn.ReLU(True))
            )
        decoder_layers.append(
            ResnetTransposeBlock(hidden_channels[0], out_channel, kernel_size=kernel_size, 
                                 stride=stride, act_fn=nn.Sigmoid())
        )
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
