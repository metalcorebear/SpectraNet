# SpectraNet

(C) 2022 Mark M. Bailey, PhD

## About
This project is an ongoing effort to develop a method to classify 2D spectal data using convolutional neural networks.  The hypothesis is that 2D spectral data, when represented as an array, have the same data structure as an image of bit depth 1, and thus can be classified analogously.  While I'm testing this script against images to ensure the propoer functioning of the CNN, I ultimately see this being used to classify vibrational spectra that dampen over time, or similar data.

Much of this code was adapted from Gad, Aflak, and Huang, **et al.** (see references below).

Be advised that this is a beta version and its functionality is currently being tested. 

## References
* Gad, A. (2018). Building Convolutional Neural Network using NumPy from Scratch. **Towards Data Science.** https://towardsdatascience.com/building-convolutional-neural-network-using-numpy-from-scratch-b30aac50e50a
* Aflak, O. (2018). Neural Network from Scratch in Python. **Towards Data Science.** https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65
* Huang, G., Mattar, M., Lee, H., & Learned-Miller, E. G. (2012). Learning to align from scratch. In Advances in Neural Information Processing Systems (pp. 764-772). https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.lfw_subset

## Updates
* 2022-09-26: Initial commit.

## Contents
* `helper_monkey.py` - Base helper functions and classes.
* `spectrum.py` - Contains ModelBuilder class.

## ModelBuilder Class
`ModelBuilder`<br><br>

        &emsp;`load_data(dataset, **options)`<br>
        
        &emsp;Loads dataset for training, which should be a 3D array (# of spectra, y-axis, x-axis), or if images (# of images, height, width).  Images should be in grayscale.  Labels can be input as a list of text labels (which will be converted to integer categorical labels).<br><br>
        
        &emsp;Parameters (required):<br>
            &emsp;&emsp;'dataset : `ndarray`' - Stacked spectra or images as Numpy arrays.<br><br>                                                
        
        &emsp;Parameters (optional):<br>
            &emsp;&emsp;'dims : `tuple`' - Dimensions for reshaped arrays (rows, columns). Default value is (227, 227).<br>
            &emsp;&emsp;'labels : `list`' - list of output labels.  Default will result in 10 model output parameters.<br><br>
            
        &emsp;`load_images(filepath, **options)`<br>
        
        &emsp;Loads and processes a set of image files from a specified file path.  Replaces the reshaped_data attribute.  Note all files in directory must be images as this method does not currently sort out non-image files.<br><br>
    
        &emsp;Parameters (required):<br>
            &emsp;&emsp;'filepath : `str`' - Path to folder where image files are stored.<br><br>                                             
        
       &emsp;Parameters (optional):<br>
            &emsp;&emsp;'dims : `tuple`' - Dimensions for reshaped arrays (rows, columns). Default value is (227, 227).<br>
            &emsp;&emsp;'labels : `list`' - list of output labels.  Default will result in 10 model output parameters.<br><br>
            
        &emsp;`train(**options)`<br><br>
        
        &emsp;Trains the model on the loaded dataset.<br><br>
        
        &emsp;Parameters (optional):<br>
            &emsp;&emsp;'train : `float`' - Training fraction. Default is 0.4.<br>
            &emsp;&emsp;'test : `float`' - Testing fraction. Default is 0.6.<br>
            &emsp;&emsp;'epochs : `int`' - Total training epochs.  Default is 35.<br>
            &emsp;&emsp;'learning_rate : `int`' - Learning rate. Default is 0.1.<br><br>
            
        &emsp;`save_model(**options)`<br><br>
        
        &emsp;Saves a copy of the model to the specified (optional) path.<br><br>
        
        &emsp;Parameters (optional):<br>
            &emsp;&emsp;'path : `str`' - Absolute path to saved file.  Default is 'net.model' in the current directory.<br><br>
            
        &emsp;`load_model(path)`<br><br>
        
        &emsp;Loads model from specified path.<br><br>
        
        &emsp;Parameters (required):<br>
            &emsp;&emsp;'path : `str`' - location of model file.

