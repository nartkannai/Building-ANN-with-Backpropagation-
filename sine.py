import numpy as np
import matplotlib.pyplot as plt

# Generate training data (1000 points equally distributed)
x_train = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
y_train = np.sin(x_train)

# Generate validation data (300 points randomly)
x_valid = np.random.uniform(-2 * np.pi, 2 * np.pi, 300)

# Define the activation function (tanh)
def activate(x):
    return np.tanh(x)

# Initialize the neural network parameters
input_size = 1
hidden_size = 50  # You can adjust the number of hidden units
output_size = 1

# Initialize weights and biases
w1 = np.random.randn(hidden_size, input_size)
b1 = np.zeros((hidden_size, 1))
w2 = np.random.randn(output_size, hidden_size)
b2 = np.zeros((output_size, 1))

# Training parameters
epochs = 1000
learning_rate = 0.01

# Training loop
for epoch in range(epochs):
    # Forward propagation
    z1 = np.dot(w1, x_train.reshape(1, -1)) + b1
    a1 = activate(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = z2

    # Compute loss (mean squared error)
    loss = np.mean((a2 - y_train.reshape(1, -1)) ** 2)

    # Backpropagation
    dz2 = a2 - y_train.reshape(1, -1)
    dw2 = np.dot(dz2, a1.T) / len(x_train)
    db2 = np.sum(dz2, axis=1, keepdims=True) / len(x_train)
    dz1 = np.dot(w2.T, dz2) * (1 - np.power(a1, 2))  # Derivative of tanh
    dw1 = np.dot(dz1, x_train.reshape(1, -1).T) / len(x_train)
    db1 = np.sum(dz1, axis=1, keepdims=True) / len(x_train)

    # Update weights and biases
    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Validation
z1_valid = np.dot(w1, x_valid.reshape(1, -1)) + b1
a1_valid = activate(z1_valid)
z2_valid = np.dot(w2, a1_valid) + b2
a2_valid = z2_valid

# Plot the training and validation data
plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, label='Training Data', s=5)
plt.scatter(x_valid, a2_valid, label='Validation Output', s=5, c='r', marker='x')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sine Function Approximation')
plt.legend()
plt.show()
