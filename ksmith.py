#https://github.com/ksmith6/sdc-semantic-segmentation
#This does not do well despite having similar structure.  Good case for analysis.

import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Define global variables for hyperparameters
NUM_EPOCHS = 10        # Number of training epochs
LEARNING_RATE = 1e-4    # Learning rate for ADAM optimizer
BATCH_SIZE = 16         # Batch size


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
    
    # Specify the layers to fetch from the VGG16 network.
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    # Load the serialized model and weights.
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path);

    # Print a helper message to indicate that the model was loaded.
    print('Loaded VGG!')

    # Get VGG graph.
    graph = tf.get_default_graph()

    # Fetch the layers from the VGG16 graph.
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name) 
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    # Implement 1x1 convolution from layer 7 from VGG
    kernel_size = 1
    conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, kernel_size, padding='same', strides=1, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    

    # Upsample the layer
    kernel = 4
    output = tf.layers.conv2d_transpose(conv_1x1, num_classes, kernel, padding='same', strides=2, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # Add skip connection
    kernel = 1
    pool_4 = tf.layers.conv2d(vgg_layer4_out, num_classes, kernel, padding='same', strides=1, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    output = tf.add(output, pool_4)


    # Upsample again by 2
    kernel = 4
    output = tf.layers.conv2d_transpose(output, num_classes, kernel, padding='same', strides=2, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # Make pooling layer
    kernel = 1
    pool_3 = tf.layers.conv2d(vgg_layer3_out, num_classes, kernel, padding='same', strides=1, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    # Skip connection
    output = tf.add(output, pool_3)

    # Upsample by 8
    kernel = 16
    output = tf.layers.conv2d_transpose(output, num_classes, kernel, padding='same', strides=8, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    return output
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

    # Reshape logits and labels
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    truth = tf.reshape(correct_label, (-1, num_classes))

    # Form the cross-entropy loss operation
    cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(labels=truth, logits=logits)
    loss_op = tf.reduce_mean(cross_entropy_loss)

    # Create an ADAM Optimizer for training
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    return logits, train_op, loss_op

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

    # Initialize tf variables
    sess.run(tf.global_variables_initializer())

    # Initialize list to store history of loss from training steps
    loss_history = []

    # Train the network for NUM_EPOCHS epochs.
    for epoch in range(NUM_EPOCHS):

        # Reset Batch Counter 
        batch_ctr = 0

        # Generate batches of images and labels from the generator.
        for images, labels in get_batches_fn(batch_size):
            #x, loss = sess.run([train_op, cross_entropy_loss], feed_dict={learning_rate:1e-4, keep_prob:1.0, input_image:images, correct_label:correct_label})
            batch_ctr += 1
            
            # Handle irregularly shaped batches (not fully-sized)
            if images.shape[0] != batch_size:
                # Skip to next epoch
                continue
             
             # Train the network for a single batch.
            x, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={learning_rate: LEARNING_RATE, keep_prob: 1.0, 
                               input_image: images, correct_label: labels})
             
             # Print out the training progress and loss.
            print("Epoch #{}, Batch #{}, Loss {:.8f}".format((epoch+1), batch_ctr, loss))
             
             # Append the loss to the list of losses for future plotting.
            loss_history.append(loss)
        

    # Return the list of losses during training.
    return loss_history
    
    
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Create placeholders for the labels and the learning rate
    label = tf.placeholder(tf.float32, shape=[BATCH_SIZE, image_shape[0], image_shape[1], num_classes])
    learning_rate = tf.placeholder(tf.float32, shape=[])

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # TODO: OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg and layers
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)

        # Generate the training and loss operations.
        logits, train_op, loss_op = optimize(layer_output, label, learning_rate, num_classes)

        # Train the neural network using ADAM Optimizer
        loss_history = train_nn(sess, NUM_EPOCHS, BATCH_SIZE, get_batches_fn, train_op, loss_op, input_image, label, keep_prob, learning_rate)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # TODO: Apply the trained model to a video
        # TODO: Plot the loss_history variable.


if __name__ == '__main__':
    run()
