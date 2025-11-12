import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    
    def __init__(self, n_channels, n_classes, bilinear):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256, bilinear)
        self.up2 = up(512, 128, bilinear)
        self.up3 = up(256, 64, bilinear)
        self.up4 = up(128, 64, bilinear)
        self.outc = outconv(64, n_classes)
    
    def forward(self, x):
        try:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            x = self.outc(x)
            return torch.sigmoid(x)
        except Exception as e:
            print(f"UNet forward error: {e}", flush=True)
            raise
    

class double_conv(nn.Module):
    ''' 2 * (conv -> BN -> ReLU) '''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x
    

class inconv(nn.Module):
    ''' double_conv '''
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)
    
    def forward(self, x):
        x = self.conv(x)
        return x
    

class down(nn.Module):
    ''' maxpool -> double_conv '''
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )
    
    def forward(self, x):
        x = self.mpconv(x)
        return x
    

class up(nn.Module):
    ''' upsample -> conv '''
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)
    
    def forward(self, x1, x2):
        x1 = self.up(x1) # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2, diffY // 2, diffY - diffY//2))
        
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
    

class outconv(nn.Module):
    ''' conv '''
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
    
    def forward(self, x):
        x = self.conv(x)
        return x

class MelanomaClassifier(nn.Module):
    """
    Model architecture TinyVGG from:
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1), # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                          stride=2) # default stride value is same as kernel_size

        )

        hidden_units_B = hidden_units * 2 #increase out-feautures hidden units (32 to 64)
        print(hidden_units_B) # debug

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units_B, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units_B, hidden_units_B, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        hidden_units = hidden_units_B * 2 #increase out-feautures hidden units_B (64 to 128)
        print(hidden_units) # debug

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(hidden_units_B, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        hidden_units_B = hidden_units * 2 #ncrease out-feautures hidden units (128 to 256)
        print(hidden_units_B) # debug
        
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units_B, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units_B, hidden_units_B, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self._to_linear = None
        self._compute_linear_input_size(input_shape) #dynamically compute in_features

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from?
            # It's because each layer of our network compresses and changes the shape of our input data.
            nn.Linear(in_features=self._to_linear,
                      out_features=output_shape)

        )

    # Function to automate the calculation of in_features for the final nn.linear layer by passing a dummy tensor into the model.
    def _compute_linear_input_size(self, input_shape):
        """Pass a dummy tensor through conv layers by making an inference to determine in_features size."""
        with torch.no_grad():
            dummy_input = torch.randn(1, input_shape, 224, 224)  # Dummy tensor input to simulate the changes of the in_features as a tensor passes the conv blocks.
            output = self.conv_block_1(dummy_input)
            output = self.conv_block_2(output)
            output = self.conv_block_3(output)
            output = self.conv_block_4(output)
            self._to_linear = output.view(1, -1).shape[1]  # Flatten and get feature size
            print("Computed in_features:", self._to_linear)  # Debugging

    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        # print(x.shape)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        print(f"Feature map shape before flattening: {x.shape}")  # Debugging line
        x = self.classifier(x)
        # print(x.shape)
        return x
        # return self.classifier(self.conv_block_2(self.conv_block_1(x))) # <- leverage the benefits of operator fusion
