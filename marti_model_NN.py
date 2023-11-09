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
input_dim =   X_train.shape[2]

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
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# Display the model summary
model.summary()



# Train the model
history = model.fit([X_train[:, 0], X_train[:, 1]], Y_train, 
          validation_data=([X_validate[:, 0], X_validate[:, 1]], Y_validate), 
          batch_size=64, epochs=3)


import matplotlib.pyplot as plt
# Plot the training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Evaluation

# prediction
y_pred = model.predict([X_test[:, 0], X_test[:, 1]])

# Define a custom function to calculate accuracy within a margin of error
def calculate_accuracy(y_true, y_pred, margin):
    correct_predictions = 0
    for true, pred in zip(y_true, y_pred):
        if abs(true - pred) < margin:
            correct_predictions += 1
    return correct_predictions / len(y_true)

# Set a margin of error
margin_of_error = 0.001  # Adjust as needed based on the tolerance level

# Calculate accuracy within the margin of error
accuracy_within_margin = calculate_accuracy(Y_test, y_pred.flatten(), margin_of_error)

# Display the calculated accuracy within the margin of error
print(f"Accuracy within a margin of {margin_of_error}: {accuracy_within_margin}")

