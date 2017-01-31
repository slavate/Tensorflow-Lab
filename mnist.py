# Load the modules
import pickle
import math

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

# Reload the data
pickle_file = 'notMNIST.pickle'
with open(pickle_file, 'rb') as f:
  pickle_data = pickle.load(f)
  train_features = pickle_data['train_dataset']
  train_labels = pickle_data['train_labels']
  valid_features = pickle_data['valid_dataset']
  valid_labels = pickle_data['valid_labels']
  test_features = pickle_data['test_dataset']
  test_labels = pickle_data['test_labels']
  del pickle_data  # Free up memory


print('Data and modules loaded.')

features_count = 784
labels_count = 10

# Set the features and labels tensors
features = tf.placeholder(tf.float32)
labels = tf.placeholder(tf.float32)

# Set the weights and biases tensors
weights = tf.Variable(tf.random_normal((features_count, labels_count)))
biases = tf.Variable(tf.zeros(labels_count))
##############
# relus_count = 1
# hidden_layer_weights = tf.random_normal((features_count, relus_count))
# out_weights = tf.random_normal((relus_count, labels_count))
#
# weights = [
#     tf.Variable(hidden_layer_weights),
#     tf.Variable(out_weights)]
# biases = [
#     tf.Variable(tf.zeros(relus_count)),
#     tf.Variable(tf.zeros(labels_count))]

# Feed dicts for training, validation, and test session
train_feed_dict = {features: train_features, labels: train_labels}
valid_feed_dict = {features: valid_features, labels: valid_labels}
test_feed_dict = {features: test_features, labels: test_labels}

# NN Model
# Linear Function WX + b
logits = tf.matmul(features, weights) + biases
######
# ReLUs
# hidden_layer = tf.add(tf.matmul(features, weights[0]), biases[0])
# hidden_layer = tf.nn.relu(hidden_layer)
# logits = tf.add(tf.matmul(hidden_layer, weights[1]), biases[1])
#
#
prediction = tf.nn.softmax(logits)

# Cross entropy
# cross_entropy = -tf.reduce_sum(labels * tf.log(prediction), reduction_indices=1)
cross_entropy = -tf.reduce_sum(labels*tf.log(tf.clip_by_value(prediction,1e-10,1.0)))

# Training loss
loss = tf.reduce_mean(cross_entropy)

# Determine if the predictions are correct
is_correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))

# Calculate the accuracy of the predictions
accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))

# Find the best parameters for each configuration
epochs = 100
batch_size = 1000
learning_rate = 1e-3
# learning_rate = tf.placeholder(tf.float32, shape=[])

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
# optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate).minimize(loss)
# optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)
# optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

# The accuracy measured against the validation set
validation_accuracy = 0.0
# The accuracy measured against the test set
test_accuracy = 0.0

# Measurements use for graphing loss and accuracy
log_batch_step = 500
batches = []
loss_batch = []
train_acc_batch = []
valid_acc_batch = []

# Create an operation that initializes all variables
init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    batch_count = int(math.ceil(len(train_features) / batch_size))

    for epoch_i in range(epochs):
        # Progress bar
        batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i + 1, epochs), unit='batches')

        # The training cycle
        for batch_i in batches_pbar:
        # for batch_i in range(0, batch_count):
            # Get a batch of training features and labels
            batch_start = batch_i * batch_size
            batch_features = train_features[batch_start:batch_start + batch_size]
            batch_labels = train_labels[batch_start:batch_start + batch_size]

            # Run optimizer and get loss
            _, l = session.run(
                [optimizer, loss],
                feed_dict={features: batch_features, labels: batch_labels})

            # Log every 50 batches
            if not batch_i % log_batch_step:
                # Calculate Training and Validation accuracy
                training_accuracy = session.run(accuracy, feed_dict=train_feed_dict)
                validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)

                # Log batches
                previous_batch = batches[-1] if batches else 0
                batches.append(log_batch_step + previous_batch)
                loss_batch.append(l)
                train_acc_batch.append(training_accuracy)
                valid_acc_batch.append(validation_accuracy)

        # Check accuracy against Training data
        training_accuracy = session.run(accuracy, feed_dict=train_feed_dict)

        # Check accuracy against Validation data
        validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)

        # Check accuracy against Test data
        test_accuracy = session.run(accuracy, feed_dict=test_feed_dict)

loss_plot = plt.subplot(211)
loss_plot.set_title('Loss')
loss_plot.plot(batches, loss_batch, 'g')
loss_plot.set_xlim([batches[0], batches[-1]])
acc_plot = plt.subplot(212)
acc_plot.set_title('Accuracy')
acc_plot.plot(batches, train_acc_batch, 'r', label='Training Accuracy')
acc_plot.plot(batches, valid_acc_batch, 'x', label='Validation Accuracy')
acc_plot.set_ylim([0, 1.0])
acc_plot.set_xlim([batches[0], batches[-1]])
acc_plot.legend(loc=4)
plt.tight_layout()

print('Training accuracy at {}'.format(training_accuracy))
print('Validation accuracy at {}'.format(validation_accuracy))
print('Test Accuracy is {}'.format(test_accuracy))

plt.show()