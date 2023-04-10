import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class MNISTLoader():
    #Constructor
    def __init__(self):
        (self.train_x, self.train_y), (self.test_x, self.test_y) = tf.keras.datasets.mnist.load_data()
        self.train_x = np.expand_dims((self.train_x.astype(np.float32)/255.0), axis=-1)
        self.test_x = np.expand_dims(self.test_x.astype(np.float32)/255.0, axis=-1)
        self.train_y = self.train_y.astype(np.int32)
        self.test_y = self.test_y.astype(np.int32)
        self.num_train = len(self.train_x)
        self.num_test = len(self.test_x)

    def random_group(self):
        ary = np.arange(0,self.num_train,dtype=np.int32)
        random.shuffle(ary)
        return ary

    def print_data(self):
        print("train data x " + str(np.shape(self.train_x)))
        print("train data y " + str(np.shape(self.train_y)))
        print("test data x " + str(np.shape(self.test_x)))
        print("test data y " + str(np.shape(self.test_y)))
        print("Number of train " + str(self.num_train))


class CNNmodel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.input_data = None
        self.conv1 = tf.keras.layers.Conv2D(filters=6, kernel_size=[5,5], activation=tf.nn.relu, use_bias=True) #output: [batch_size,24,24,6]
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2,2], strides=2)                                      #output: [batch_size,12,12,6]
        self.conv2 = tf.keras.layers.Conv2D(filters=12, kernel_size=[5,5], activation=tf.nn.relu, use_bias=True)#A size of filter is 5x5x6. output: [batch_size,8,8,12]
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2,2], strides=2)                                      #output: [batch_size,4,4,12]
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=10, use_bias=True)

    def call(self,input):
        self.input_data = input
        x = self.conv1(self.input_data)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        output = tf.nn.softmax(x)
        return output

    def summary_model(self):
        inputs = tf.keras.Input(shape=self.input_data[0].shape)
        outputs = self.call(inputs)
        tf.keras.Model(inputs=inputs, outputs=outputs, name="CNN").summary()

data = MNISTLoader()

learning_rate = 0.01
epoch = 5
batch_size = 50
batch_per_epoch = int(data.num_train // batch_size)


#Model traning
model = CNNmodel()
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate) #Define the optimizer SGD
for i in range(epoch):
    group_ary = data.random_group()
    for j in range(batch_per_epoch):
        x = data.train_x[group_ary[j*batch_size:(j+1)*batch_size-1]] #group
        y = data.train_y[group_ary[j*batch_size:(j+1)*batch_size-1]]
        #Forward Propagation
        with tf.GradientTape() as tape:
            pred_y = model(x)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=pred_y)
            loss = tf.reduce_mean(loss)
            print("epoch " + str(i+1) + " batch %d: loss %f" % (j+1, loss.numpy()))
        #Backward Propagation
        grads = tape.gradient(loss, model.variables)
        #Update the parameters
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

# 宣告一個評量器 tf.keras.metrics
sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
num_batches = int(data.num_test // batch_size)
for batch_index in range(num_batches):
    start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
    y_pred_test = model.predict(data.test_x[start_index: end_index])
    sparse_categorical_accuracy.update_state(y_true=data.test_y[start_index: end_index], y_pred=y_pred_test)
print("test accuracy: %f" % sparse_categorical_accuracy.result())