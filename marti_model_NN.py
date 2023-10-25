from keras import backend as K
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.optimizers import Adam
import numpy as np


# Load the Data
X_train = np.load("./Data/Dataset/X_train.npy")
X_test = np.load("./Data/Dataset/X_test.npy")
X_validate = np.load("./Data/Dataset/X_validate.npy")

Y_train = np.load("./Data/Dataset/Y_train.npy")
Y_test = np.load("./Data/Dataset/Y_test.npy")
Y_validate = np.load("./Data/Dataset/Y_validate.npy")

print(X_train.shape, X_test.shape, X_validate.shape)

# Dimensions of the input vectors
input_dim = 10*256  

# Define the Siamese network architecture
input_a = Input(shape=(input_dim,))
input_b = Input(shape=(input_dim,))

# Shared weights between the two networks
shared_dense_layer_1 = Dense(64, activation='relu')
shared_dense_layer_2 = Dense(32, activation='relu')  # Additional layer
shared_dense_layer_3 = Dense(16, activation='relu')  # Additional layer

# Stacking the layers
encoded_a = shared_dense_layer_3(shared_dense_layer_2(shared_dense_layer_1(input_a)))
encoded_b = shared_dense_layer_3(shared_dense_layer_2(shared_dense_layer_1(input_b)))


# Define the Euclidean distance between the encoded vectors
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

distance = Lambda(euclidean_distance)([encoded_a, encoded_b])

# Output layer with a sigmoid activation function
prediction = Dense(1, activation='sigmoid')(distance)

# Define the Siamese model
model = Model(inputs=[input_a, input_b], outputs=prediction)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# Display the model summary
model.summary()



