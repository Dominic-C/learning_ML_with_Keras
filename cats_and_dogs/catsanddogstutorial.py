import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

NAME="Cats-vs-dogs-cnn-64x2-{}".format(int(time.time())) # if models have the same name, they are appended to each other

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33) # specify fraction of GPU the model can take
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

X = X/255.0

model = Sequential()



dense_layers = [0,1,2]
layer_sizes = [32,64,128]
conv_layers = [1,2,3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-filters-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)
            # print(X.shape)
            model.add(Conv2D(layer_size, (3,3), input_shape = X.shape[1:]))
            model.add(Activation("relu")) # in general, conv and max pooling go together.
            model.add(MaxPooling2D(pool_size=(2,2)))

            # for loop to generate number of conv layers
            for i in range(layer_size-1):
                model.add(Conv2D(layer_size, (3,3))) # where 3,3 is the sliding window size (known as kernel)
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2,2)))
          
            model.add(Flatten()) # flatten to 1D array to connect to dense layer
            
            # for loop to generate number of dense layers
            for i in range(dense_layer):
                model.add(Dense(i))
          
            model.add(Dense(1))
            model.add(Activation("sigmoid"))
                    
            model.compile(loss="binary_crossentropy", optimizer = "adam",
                        metrics = ["accuracy"])
                    
            model.fit(X, y, epochs=10, batch_size=32, validation_split=0.1, callbacks=[tensorboard]) # batch size is how many we want to fit