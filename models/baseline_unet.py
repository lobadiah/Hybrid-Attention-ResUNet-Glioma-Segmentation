"""
Baseline U-Net Model for Medical Image Segmentation

A standard U-Net architecture for 2D medical image segmentation tasks.
Based on: Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, 
    Concatenate, Dropout, BatchNormalization
)
import numpy as np


class BaselineUNet:
    """
    Standard U-Net model for 2D medical image segmentation.
    
    Architecture:
    - Encoder: Contracting path with conv blocks and max pooling
    - Decoder: Expanding path with upsampling and skip connections
    - Output: Single channel segmentation mask
    """
    
    def __init__(self, input_shape=(256, 256, 1), num_classes=1, filters_start=64):
        """
        Initialize Baseline U-Net model.
        
        Args:
            input_shape (tuple): Input image shape (height, width, channels)
            num_classes (int): Number of output classes/channels
            filters_start (int): Number of filters in first layer
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.filters_start = filters_start
        self.model = None
        
    def _conv_block(self, inputs, filters, name_prefix):
        """
        Convolutional block with two Conv2D layers followed by BatchNormalization.
        
        Args:
            inputs: Input tensor
            filters (int): Number of filters
            name_prefix (str): Prefix for layer names
            
        Returns:
            Output tensor after convolutions
        """
        x = Conv2D(filters, (3, 3), activation='relu', padding='same', 
                   name=f'{name_prefix}_conv1')(inputs)
        x = BatchNormalization(name=f'{name_prefix}_bn1')(x)
        x = Conv2D(filters, (3, 3), activation='relu', padding='same', 
                   name=f'{name_prefix}_conv2')(x)
        x = BatchNormalization(name=f'{name_prefix}_bn2')(x)
        return x
    
    def build(self):
        """
        Build the Baseline U-Net model.
        
        Returns:
            keras.Model: Compiled U-Net model
        """
        inputs = Input(shape=self.input_shape, name='input')
        
        # Encoder (Contracting Path)
        # Level 1
        conv1 = self._conv_block(inputs, self.filters_start, 'enc1')
        pool1 = MaxPooling2D((2, 2), name='pool1')(conv1)
        
        # Level 2
        conv2 = self._conv_block(pool1, self.filters_start * 2, 'enc2')
        pool2 = MaxPooling2D((2, 2), name='pool2')(conv2)
        
        # Level 3
        conv3 = self._conv_block(pool2, self.filters_start * 4, 'enc3')
        pool3 = MaxPooling2D((2, 2), name='pool3')(conv3)
        
        # Level 4
        conv4 = self._conv_block(pool3, self.filters_start * 8, 'enc4')
        pool4 = MaxPooling2D((2, 2), name='pool4')(conv4)
        
        # Bottleneck
        bottleneck = self._conv_block(pool4, self.filters_start * 16, 'bottleneck')
        
        # Decoder (Expanding Path)
        # Level 4 Up
        up4 = UpSampling2D((2, 2), name='up4')(bottleneck)
        up4 = Concatenate(name='concat4')([up4, conv4])
        dec4 = self._conv_block(up4, self.filters_start * 8, 'dec4')
        
        # Level 3 Up
        up3 = UpSampling2D((2, 2), name='up3')(dec4)
        up3 = Concatenate(name='concat3')([up3, conv3])
        dec3 = self._conv_block(up3, self.filters_start * 4, 'dec3')
        
        # Level 2 Up
        up2 = UpSampling2D((2, 2), name='up2')(dec3)
        up2 = Concatenate(name='concat2')([up2, conv2])
        dec2 = self._conv_block(up2, self.filters_start * 2, 'dec2')
        
        # Level 1 Up
        up1 = UpSampling2D((2, 2), name='up1')(dec2)
        up1 = Concatenate(name='concat1')([up1, conv1])
        dec1 = self._conv_block(up1, self.filters_start, 'dec1')
        
        # Output Layer
        outputs = Conv2D(self.num_classes, (1, 1), activation='sigmoid', 
                        name='output')(dec1)
        
        self.model = Model(inputs=inputs, outputs=outputs, name='BaselineUNet')
        return self.model
    
    def compile(self, learning_rate=1e-4):
        """
        Compile the model with optimizer and loss function.
        
        Args:
            learning_rate (float): Learning rate for Adam optimizer
        """
        if self.model is None:
            self.build()
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
    
    def summary(self):
        """Print model architecture summary."""
        if self.model is None:
            self.build()
        return self.model.summary()
    
    def get_model(self):
        """Get the compiled model."""
        if self.model is None:
            self.build()
            self.compile()
        return self.model


# Example usage
if __name__ == '__main__':
    # Initialize and build model
    unet = BaselineUNet(input_shape=(256, 256, 1), num_classes=1)
    model = unet.build()
    unet.compile()
    unet.summary()
    
    # Print model info
    print(f"\nTotal parameters: {model.count_params():,}")