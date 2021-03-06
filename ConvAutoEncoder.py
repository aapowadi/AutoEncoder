import torch.nn as nn
import torch.nn.functional as tf


class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()

        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 784 ==> 9
        self.conv1 = nn.Conv2d(3, 32, (3, 3), padding='same')
        self.max1 = nn.MaxPool2d(2, 2)  # 224 x 224 -> 112 x 112
        self.conv2 = nn.Conv2d(32, 64, (3, 3), padding='same')
        self.max2 = nn.MaxPool2d(2, 2)  # 106 x 106 -> 53 x 53
        self.conv3 = nn.Conv2d(64, 128, (3, 3), padding='same')
        self.max3 = nn.MaxPool2d(2, 2)  # 53 x 53 -> 23 x 23
        self.conv4 = nn.Conv2d(128, 256, (3, 3), padding='same')
        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 9 ==> 784
        self.conv5 = nn.Conv2d(256, 256, (3, 3), padding='same')
        self.up5 = nn.ConvTranspose2d(256, 256, (2, 2), 2)  # 23 x 23 -> 57 x 57 -> 114 x 114
        self.conv6 = nn.Conv2d(256, 128, (3, 3), padding='same')
        self.up6 = nn.ConvTranspose2d(128, 128, (2, 2), 2)  # 57 x 57 -> 114 x 114
        self.conv7 = nn.Conv2d(128, 64, (3, 3), padding='same')
        self.up7 = nn.ConvTranspose2d(64, 64, (2, 2), 2)  # 114 x 114 -> 224 x 224
        self.conv_out = nn.Conv2d(64, 3, (3, 3), padding='same')

    def forward(self, x, latents=False):
        x = tf.relu(self.conv1(x))
        x = self.max1(x)
        x = tf.relu(self.conv2(x))
        x = self.max2(x)
        x = tf.relu(self.conv3(x))
        x = self.max3(x)
        x = tf.relu(self.conv4(x))
        if latents:
            return x
        else:
            x = tf.relu(self.conv5(x))
            x = self.up5(x)
            x = tf.relu(self.conv6(x))
            x = self.up6(x)
            x = tf.relu(self.conv7(x))
            x = self.up7(x)
            x = tf.sigmoid(self.conv_out(x))
            return x
