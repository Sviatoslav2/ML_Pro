import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def max_poling(layer):
    layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
    return layer


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))    
    
def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights
    
    
def conv2d_transpose_strided(x, W, b,batch_size, output_shape=None):
    try:
        if output_shape is None:    
            output_shape = x.get_shape().as_list()
            output_shape[1] *= 2
            output_shape[2] *= 2
            output_shape[3] = W.get_shape().as_list()[2]
            output_shape[0] = batch_size
            
    except Exception:
        print("Hear!")
    print("x = ",x)
    print("W = ", W)
    
    print("output_shape = ", tf.convert_to_tensor(np.asarray(output_shape)))
    print("strides = ", [1, 2, 2, 1])
    
    conv = tf.nn.conv2d_transpose(x, W, tf.convert_to_tensor(np.asarray(output_shape)), strides=[1, 2, 2, 1], padding='SAME')
    try:
        res = tf.nn.bias_add(conv, b)    
    except Exception:
        print("Hear!")
    return res 
    
def new_conv2d_transpose_layer(input,
                               num_input_channels,
                               filter_output_size,
                               strides_for_transpose,
                               num_filters,
                               activation=tf.nn.relu):
                               
    '''
    batch_size = tf.shape(something_or_other)[0]
    deconv_shape = tf.pack([batch_size, 40, 40, 32])
    conv2d_transpose(..., output_shape=deconv_shape, ...)
    '''                           
                               
                               
                                
    shape = [filter_output_size, filter_output_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)
    batch_size = tf.shape(input)[0]
    if batch_size == None:
        batch_size = -1
    
    #try:
    layer = conv2d_transpose_strided(input, weights, biases, batch_size, output_shape=None)
    #except Exception as erorr:    
    layer = activation(layer)
    
    return layer 
######################################################################################################################
######################################################################################################################
######################################################################################################################    


