# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 12:38:24 2022

@author: markmbailey
"""

"""
Contains classes for training and for querying arrays, and for saving model weights using Pickle.
"""

import helper_monkey as mojo
import sys

"""
For training, input should be a 3D array (# of spectra, y-axis, x-axis), or if images (# of images, height, width).  Images should be in grayscale.  Labels can be input as a list of text labels (which will be converted to integer categorical labels).
                                         
"""

# wrapper class for CNN and ANN
class ModelBuilder:
    def __init__(self):
        self.data = None
        self.reshaped_data = None
        self.model = None
        self.labels = None
        self.imgpaths = None
    
    def load_data(self, dataset, **options):
        '''
        load_data(dataset, **options)
        
        Loads dataset for training, which should be a 3D array (# of spectra, y-axis, x-axis), or if images (# of images, height, width).  Images should be in grayscale.  Labels can be input as a list of text labels (which will be converted to integer categorical labels).
        
        Parameters (required):
            'dataset : ndarray' - Stacked spectra or images as Numpy arrays.                                                        
        
        Parameters (optional):
            'dims : tuple' - Dimensions for reshaped arrays (rows, columns). Default value is (227, 227).
            'labels : list' - list of output labels.  Default will result in 10 model output parameters.
        '''
        
        dims = options.pop('dims', (227, 227))
        self.data = dataset
        self.labels = options.pop("labels", [1 for i in range(10)])
        self.reshaped_data = mojo.reshape_imgs(self.data, h=dims[0], w=dims[1])
        
    def load_images(self, filepath, **options):
        '''
        load_images(filepath, **options)
        
        Loads and processes a set of image files from a specified file path.  Replaces the reshaped_data attribute.  Note all files in directory must be images as this method does not currently sort out non-image files.

        Parameters (required):
            'filepath : str' - Path to folder where image files are stored.                                                        
        
        Parameters (optional):
            'dims : tuple' - Dimensions for reshaped arrays (rows, columns). Default value is (227, 227).
            'labels : list' - list of output labels.  Default will result in 10 model output parameters.
        '''
        dims = options.pop('dims', (227, 227))
        self.labels = options.pop("labels", [1 for i in range(10)])
        self.reshaped_data, self.imgpaths = mojo.load_reshape_images(filepath, h=dims[0], w=dims[1])
        
    
    def train(self, **options):
        '''
        train(**options)
        
        Trains the model on the loaded dataset.
        
        Parameters (optional):
            'train : float' - Training fraction. Default is 0.4.
            'test : float' - Testing fraction. Default is 0.6.
            'epochs : int' - Total training epochs.  Default is 35.
            'learning_rate : int' - Learning rate. Default is 0.1.
        '''
        train = options.pop('train', 0.4)
        test = options.pop('test', 0.6)
        epochs = options.pop('epochs', 35)
        learning_rate = options.pop('learning_rate', 0.1)
        if self.reshaped_data == None:
            sys.exit('Data must be loaded first.')
        c_stack = mojo.convolutional_phase(self.reshaped_data)
        self.model = mojo.build_ANN(self, c_stack)
        self.confusion_matrix = mojo.net_test(self.model, c_stack, self.labels, train=train, test=test, epochs=epochs, learning_rate=learning_rate)
        
        
    def save_model(self, **options):
        '''
        save_model(**options)
        
        Saves a copy of the model to the specified (optional) path.
        
        Parameters (optional):
            'path : str' - Absolute path to saved file.  Default is 'net.model' in the current directory.
        '''
        
        filename = options.pop('path', 'net.model')
        mojo.save_model(self.model, filename=filename)
    
    
    def load_model(self, path):
        '''
        load_model(path)
        
        Loads model from specified path.
        
        Parameters (required):
            'path : str' - location of model file.
        '''
        
        self.model = mojo.load_model(path)