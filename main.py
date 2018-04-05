import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import numpy as np


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    layer_names = []
    layer_names.append('image_input:0')
    layer_names.append('keep_prob:0')
    layer_names.append('layer3_out:0')
    layer_names.append('layer4_out:0')
    layer_names.append('layer7_out:0')
    
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    
    graph = tf.get_default_graph()
    tensors = [graph.get_tensor_by_name(name) for name in layer_names]
    
    return tuple(tensors)

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    tensors = [vgg_layer3_out, vgg_layer4_out, vgg_layer7_out]
    upscale_params = [(8, 8), (2, 2), (2, 2)]
    
    # Add 1x1 convolution layers on top of each input
    tensors = [tf.layers.conv2d(tensor, num_classes, 1, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3)) for tensor in tensors]
    
    # Combine layers
    while True:
        params = upscale_params[len(tensors) - 1]
        # Upscale the last tensor in our list and remove it
        result = tf.layers.conv2d_transpose(tensors.pop(), num_classes, sum(params), strides=params, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        
        # Stop if it was the last tensor
        if len(tensors) == 0:
            break
            
        # If not -- add it to the one before it (now the last) and continue
        tensors[-1] = tf.add(result, tensors[-1])
    
    return result

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    
    # Flatten last layer and training labels
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    
    # Create cross entropy loss function and optimizer
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    
    # Include regularizers to the loss function
    l2_loss = tf.losses.get_regularization_losses()
    cross_entropy_loss += tf.reduce_sum(l2_loss)
    
    adam_op = tf.train.AdamOptimizer(learning_rate)
    
    # Create training operation with defined optimizer and loss function
    train_op = adam_op.minimize(cross_entropy_loss)
    
    return logits, train_op, cross_entropy_loss
    
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    
    sess.run(tf.global_variables_initializer())
    
    # Training parameters
    keep_prob_value = .25
    learning_rate_value = .001
    
    # Initialize list of loss function results for each epoch
    train_results = []

    print("Training start")
    for epoch in range(epochs):
        
        print("Epoch {0} start".format(epoch))
        
        # List of results for each batch in this epoch
        epoch_results = []
        
        for (images, labels) in get_batches_fn(batch_size):
            # Run training on one batch
            feed_dict = {input_image: images, correct_label: labels, keep_prob: keep_prob_value, learning_rate: learning_rate_value}
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict=feed_dict)
            
            # Print and save the results
            print('Loss: {0}'.format(loss))
            epoch_results.append(loss);
            
        train_results.append(epoch_results);
    
    # Save results for evaluating later
    np.save('train_results_dot25.dot001.batch8.epochs50', np.array(train_results));
    
            
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    
    data_dir = './data'
    runs_dir = './runs'
    
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)
    
    # Batch size and number of training epochs params
    epochs = 34
    batch_size = 8

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        
        #declare vars
        correct_label = tf.placeholder(tf.float32,shape=(None,image_shape[0],image_shape[1],num_classes))
        learning_rate = tf.placeholder(tf.float32,())        
    
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        output_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)
        
        logits, train_op, cross_entropy_loss = optimize(output_layer, correct_label, learning_rate, num_classes)
        
        # Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate)
        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)


if __name__ == '__main__':
    run()
