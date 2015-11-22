def convert(x):
    return (2 * x).astype(int)

import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# Create the model --> y = softmax(Wx+b)
y = tf.nn.softmax(tf.matmul(x,W) + b)

#Placeholder for correct (true) answers
y_ = tf.placeholder("float", [None,10])
# Cross-entropy calculation
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# Train the model using backpropagation algorithm with a Gradient Descent optimizer
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})