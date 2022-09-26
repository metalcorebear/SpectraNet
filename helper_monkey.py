# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 12:35:19 2022

@author: metalcorebear
"""

"""
Helper Monkey contains functions and base classes.
"""

import numpy as np
import skimage
import pickle
import os
import cv2
import math
# resize(bottle, (140, 54))

# use zero padding for smaller images than the target size.

"""
General Functions
"""

# Load trained model object from file
def load_model(file_location):
    with open(file_location) as f:
        model = pickle.load(f)
    return model

# saves or loads the model as a Pickle object
def save_model(model_object, filename='net.model'):
    with open(filename) as f:
        pickle.dump(model_object, f)
    
# Process Images
def image_to_grayscale(img):
    img = skimage.color.rgb2gray(img)
    #img = img.reshape(img.shape[0], img.shape[1], 1)
    return img

# Get test data
# Test data are a subset of data from the LFW dataset (faces and non-faces)
# Reference Huang, G., Mattar, M., Lee, H., & Learned-Miller, E. G. (2012). Learning to align from scratch. In Advances in Neural Information Processing Systems (pp. 764-772)
def get_data():
    imgs = skimage.data.lfw_subset()
    faces = imgs[:100,:,:]
    nonfaces = imgs[100:,:,:]
    return faces, nonfaces

# Reshape and normalize images: imgs array of float64 (#images, height, width)
# Each image in imgs should have bit depth of 1.
def scale(im, nR, nC):
  nR0 = im.shape[0]  # source number of rows 
  nC0 = im.shape[1]  # source number of columns
  return np.array([[im[int(nR0 * r / nR)][int(nC0 * c / nC)]  
             for c in range(nC)] for r in range(nR)])

def reshape_imgs(imgs, h=227, w=227):
    out_list = []
    for i in range(imgs.shape[0]):
        out = scale(imgs[i,:,:],h,w)
        m = np.amax(out)
        if m > 1.0:
            out /= m
        out_list.append(out)
    out_tuple = tuple(out_list)
    out_imgs = np.stack(out_tuple)
    return out_imgs

def load_reshape_images(filepath, h=227, w=227):
    f = [os.path.join(filepath, f) for f in os.listdir(filepath) if 
                 os.path.isfile(os.path.join(filepath, f))]
    imgs_reshaped = []
    for i in f:
        im = cv2.imread(i)
        im = image_to_grayscale(im)
        im = scale(im, h, w)
        m = np.amax(im)
        if m > 1.0:
            im /= m
        imgs_reshaped.append(im)
    imgs_tuple = tuple(imgs_reshaped)
    output_image_vector = np.stack(imgs_tuple)
    return output_image_vector, f


# Reshape feature map into 1D array
def reshape_feature_map(pool_out):
    reshaped = pool_out.reshape(1,pool_out.shape[0]*pool_out.shape[1])
    return reshaped


# Turn label list into categorical array
def name_labels(category_list):
    category_set = set(category_list)
    reduced_list = list(category_set)
    numerical = [i for i in range(len(reduced_list))]
    d = dict(zip(reduced_list,numerical))
    out_list = []
    for item in category_list:
        out_list.append(d[item])
    return d, numerical, out_list
        
def to_categorical(category_list: list):
    _, numerical, out_list = name_labels(category_list)
    start_array = np.zeros((len(out_list), len(numerical)))
    for i in range(len(out_list)):
        for j in range(len(numerical)):
            if out_list[i] == numerical[j]:
                start_array[i][j] = 1
    return start_array

def categorical_to_int(category_list):
    d, _, _ = name_labels(category_list)
    int_list = []
    for c in category_list:
        int_list.append(d[c])
    return int_list

# Base class
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # computes the output Y of a layer for a given input X
    def forward_propagation(self, input):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError


"""
Convolutional Functions

Functions for convolving spectra as image data.
"""

# Initiate convolutional filters
def start_filters():
    l1_filter = np.zeros((7,3,3))
    l1_filter[0, :, :] = np.array([[[-1,0,1],
                                   [-1,0,1],
                                   [-1,0,1]]])
    l1_filter[1, :, :] = np.array([[[1,1,1],
                                   [0,0,0],
                                   [-1,-1,-1]]])
    # Bottom Sobel
    l1_filter[2, :, :] = np.array([[[-1,-2,-1],
                                   [0,0,0],
                                   [1,2,1]]])
    # Left Sobel
    l1_filter[3, :, :] = np.array([[[1,0,-1],
                                   [2,0,-2],
                                   [1, 0, -1]]])
    # Right Sobel
    l1_filter[4, :, :] = np.array([[[-1,0,1],
                                   [-2,0,2],
                                   [-1,0,1]]])
    # Top Sobel
    l1_filter[5, :, :] = np.array([[[1,2,1],
                                   [0,0,0],
                                   [-1,2,1]]])
    # Outline
    l1_filter[6, :, :] = np.array([[[-1,-1,-1],
                                   [-1,8,-1],
                                   [-1,-1,1]]])
    return l1_filter


# Convolve Layer functions
def conv_(img, conv_filter):
    filter_size = conv_filter.shape[1]
    result = np.zeros((img.shape))
    #Looping through the image to apply the convolution operation.
    for r in np.uint16(np.arange(filter_size/2.0, 
                          img.shape[0]-filter_size/2.0+1)):
        for c in np.uint16(np.arange(filter_size/2.0, 
                                           img.shape[1]-filter_size/2.0+1)):
            #Getting the current region to get multiplied with the filter.
            curr_region = img[r-np.uint16(np.floor(filter_size/2.0)):r+np.uint16(np.ceil(filter_size/2.0)), 
                              c-np.uint16(np.floor(filter_size/2.0)):c+np.uint16(np.ceil(filter_size/2.0))]
            #Element-wise multipliplication between the current region and the filter.
            curr_result = curr_region * conv_filter
            conv_sum = np.sum(curr_result) #Summing the result of multiplication.
            result[r, c] = conv_sum #Saving the summation in the convolution layer feature map.
            
    #Clipping the outliers of the result matrix.
    final_result = result[np.uint16(filter_size/2.0):result.shape[0]-np.uint16(filter_size/2.0), 
                          np.uint16(filter_size/2.0):result.shape[1]-np.uint16(filter_size/2.0)]
    return final_result

def conv(img, conv_filter):
    # An empty feature map to hold the output of convolving the filter(s) with the image.
    feature_maps = np.zeros((img.shape[0]-conv_filter.shape[1]+1, 
                                img.shape[1]-conv_filter.shape[1]+1, 
                                conv_filter.shape[0]))

    # Convolving the image by the filter(s).
    for filter_num in range(conv_filter.shape[0]):
        #print("Filter ", filter_num + 1)
        curr_filter = conv_filter[filter_num, :] # getting a filter from the bank.
        #Checking if there are mutliple channels for the single filter.
        #If so, then each channel will convolve the image.
        #The result of all convolutions are summed to return a single feature map.
        if len(curr_filter.shape) > 2:
            conv_map = conv_(img[:, :, 0], curr_filter[:, :, 0]) # Array holding the sum of all feature maps.
            for ch_num in range(1, curr_filter.shape[-1]): # Convolving each channel with the image and summing the results.
                conv_map = conv_map + conv_(img[:, :, ch_num], 
                                  curr_filter[:, :, ch_num])
        else: # There is just a single channel in the filter.
            conv_map = conv_(img, curr_filter)
        feature_maps[:, :, filter_num] = conv_map # Holding feature map with the current filter.
    return feature_maps # Returning all feature maps.


# ReLU Layer function
def relu(feature_maps):
    #Preparing the output of the ReLU activation function.
    relu_out = np.zeros(feature_maps.shape)
    for map_num in range(feature_maps.shape[-1]):
        for r in np.arange(0,feature_maps.shape[0]):
            for c in np.arange(0, feature_maps.shape[1]):
                relu_out[r, c, map_num] = np.max([feature_maps[r, c, map_num], 0])
    return relu_out

# Max Polling Layer function
def pooling(feature_maps, size=2, stride=2):
    #Preparing the output of the pooling operation.
    pool_out = np.zeros((np.uint16((feature_maps.shape[0]-size+1)/stride),
                            np.uint16((feature_maps.shape[1]-size+1)/stride),
                            feature_maps.shape[-1]))
    for map_num in range(feature_maps.shape[-1]):
        r2 = 0
        for r in np.arange(0,feature_maps.shape[0]-size-1, stride):
            c2 = 0
            for c in np.arange(0, feature_maps.shape[1]-size-1, stride):
                pool_out[r2, c2, map_num] = np.max([feature_maps[r:r+size,  c:c+size, map_num]])
                c2 = c2 + 1
            r2 = r2 +1
    return pool_out


# Convolutional layers
def convolve(img):
    l1_filter = start_filters()
    # First Conv layer
    #print("\n**Working with conv layer 1**")
    l1_feature_map = conv(img, l1_filter)
    #print("\n**ReLU**")
    l1_feature_map_relu = relu(l1_feature_map)
    #print("\n**Pooling**")
    l1_feature_map_relu_pool = pooling(l1_feature_map_relu, 2, 2)
    #print("**End of conv layer 1**\n")
    
    # Second Conv layer
    l2_filter = np.random.rand(5, 5, 5, l1_feature_map_relu_pool.shape[-1])
    #print("\n**Working with conv layer 2**")
    l2_feature_map = conv(l1_feature_map_relu_pool, l2_filter)
    #print("\n**ReLU**")
    l2_feature_map_relu = relu(l2_feature_map)
    #print("\n**Pooling**")
    l2_feature_map_relu_pool = pooling(l2_feature_map_relu, 2, 2)
    #print("**End of conv layer 2**\n")
    
    # Third conv layer
    l3_filter = np.random.rand(3, 7, 7, l2_feature_map_relu_pool.shape[-1])
    #print("\n**Working with conv layer 3**")
    l3_feature_map = conv(l2_feature_map_relu_pool, l3_filter)
    #print("\n**ReLU**")
    l3_feature_map_relu = relu(l3_feature_map)
    #print("\n**Pooling**")
    l3_feature_map_relu_pool = pooling(l3_feature_map_relu, 2, 2)
    #print("**End of conv layer 3**\n")
    
    # Fourth conv layer
    l4_filter = np.random.rand(1, 9, 9, l3_feature_map_relu_pool.shape[-1])
    #print("\n**Working with conv layer 4**")
    l4_feature_map = conv(l3_feature_map_relu_pool, l4_filter)
    # print("\n**ReLU**")
    l4_feature_map_relu = relu(l4_feature_map)
    #print("\n**Pooling**")
    l4_feature_map_relu_pool = pooling(l4_feature_map_relu, 2, 2)
    #print("**End of conv layer 4**\n")
    
    return l4_feature_map_relu_pool

# Cycle convolutional layers and save as stacked 1D arrays in a 2D array.
def convolutional_phase(imgs):
    output_arrays = []
    for i in range(imgs.shape[0]):
        a = convolve(imgs[i,:,:])
        a_reshaped = reshape_feature_map(a)
        output_arrays.append(a_reshaped)
    out_tuple = tuple(output_arrays)
    c_stack = np.stack(out_tuple)
    return c_stack

"""
ANN Functions

Functions for building the artifical neural network for classification.
"""

# inherit from base class Layer
class FCLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error


# inherit from base class Layer
class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    # returns the activated input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error


# activation function and its derivative
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2


# loss function and its derivative
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size


# Network class
class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))
        


def build_ANN(c_stack, category_list):
    input_size = c_stack.shape[2]
    category_size = len(category_list)
    # Network
    net = Network()
    net.add(FCLayer(input_size, 100))                # input_shape=(1, input_size)    ;   output_shape=(1, 100)
    net.add(ActivationLayer(tanh, tanh_prime))
    net.add(FCLayer(100, 50))                        # input_shape=(1, 100)      ;   output_shape=(1, 50)
    net.add(ActivationLayer(tanh, tanh_prime))
    net.add(FCLayer(50, category_size))              # input_shape=(1, 50)       ;   output_shape=(1, category_size)
    net.add(ActivationLayer(tanh, tanh_prime))
    net.use(mse, mse_prime)
    return net
        
"""
Testing Functions
"""

# Split data
def data_split(c_stack, int_list, train=0.4, test = 0.6):
    total_len = c_stack.shape[2]
    test_len = math.ceil(total_len*test)
    train_len = math.floor(total_len*train)
    x_train = c_stack[:,:,train_len:]
    x_test = c_stack[:,:,:test_len:]
    y_train = int_list[train_len:]
    y_test = int_list[test_len:]
    return x_train, x_test, y_train, y_test

# find max positions in ndarrays arrays and return list
def max_positions(l):
    positions = []
    for i in l:
        m = np.amax(i)
        # if equal values at multiple positions, function will only take the first position.
        where = np.where(i == m)
        position = where[0][0]
        positions.append(position)
    return positions

# Test network
def net_test(net, c_stack, category_list, train=0.4, test = 0.6, epochs=35, learning_rate=0.1):
    _, numerical, _ = name_labels(category_list)
    int_list = to_categorical(category_list)
    x_train, x_test, y_train, y_test = data_split(c_stack, int_list, train=train, test = test)
    net.fit(x_train, y_train, epochs=epochs, learning_rate=learning_rate)
    out = net.predict(x_test)
    y_test_list = list(y_test)
    predicted = max_positions(out)
    actual = max_positions(y_test_list)
    confusion_matrix = np.zeroes((len(numerical), len(numerical)))
    for i in range(len(predicted)):
        confusion_matrix[predicted[i]][actual[i]] += 1
    return confusion_matrix