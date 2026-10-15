import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

FILE_NAME = "maintenance.txt"

def generate_weight(n_input, n_output, num_hidden_layer, hidden_layer_sizes):
    #generate weights randomly for each network layer

    weights = []
    weights.append(np.random.randn(n_input, hidden_layer_sizes[0]) * 0.01) #input layer-first hidden
    for i in range(1, num_hidden_layer): #hidden layers
        weights.append(np.random.randn(hidden_layer_sizes[i - 1], hidden_layer_sizes[i]) * 0.01)
    weights.append(np.random.randn(hidden_layer_sizes[num_hidden_layer - 1], n_output) * 0.01) #last hidden layer - output
    return weights


def generate_biases(num_hidden_layer, hidden_layer_sizes, n_output):
    # generate w0
    biases = []
    for i in range(num_hidden_layer):
        biases.append(np.zeros(hidden_layer_sizes[i]))
    biases.append(np.zeros(n_output))
    return biases


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward_pass(X, weights, biases): # calculates the output of each node in the network, given X as input
    activations = [X]  # in this I'm going to save al the output of each layers
    current_activation = X  # input
    for i in range(len(weights) - 1):
        W = weights[i]
        b = biases[i]
        net = np.dot(current_activation, W) + b  # calculate net
        current_activation = sigmoid(net)    # calculate sigmoid on net
        activations.append(current_activation)  # save output
    W_output = weights[-1]
    b_output = biases[-1]
    # for the output layer nodes, don't use sigmoid, just return the net value
    net_output = np.dot(current_activation, W_output) + b_output
    final_output = net_output
    activations.append(final_output)
    return final_output, activations


def sigmoid_derivative(o):
    return o * (1 - o)

def backward_pass(Y_true, weights, biases, activations):

    # For each weight matrix, create a corresponding zero matrix for the gradients
    grad_weights = []
    for w in weights:
        grad_weights.append(np.zeros_like(w))

    #same for bias gradients
    grad_biases = []
    for b in biases:
        grad_biases.append(np.zeros_like(b))


    #Start of Backpropagation

    #output layer error
    final_output = activations[-1]
    error = final_output - Y_true  # t_j - O_j

    # delta for the last layer
    delta = error

    # calculate gradients for the last layer (hidden -> output)
    prev_activation = activations[-2] # values calculated from last hidden layer

    # Use .reshape(-1, 1) to make the multiplication work
    grad_weights[-1] = np.dot(prev_activation.reshape(-1, 1), delta.reshape(1, -1)) # ^W = delta * X
    grad_biases[-1] = delta # ^W0

    # propagate the error backward for the other layers
    # start from the second-to-last layer (-2) and go backward to the first (index 0)
    for i in range(len(weights) - 2, -1, -1):
        w_next = weights[i + 1]
        error_hidden = np.dot(delta, w_next.T) # summation( delta * W)

        current_activation = activations[i + 1] # O_j
        delta = error_hidden * sigmoid_derivative(current_activation) # O_j (1 - O_j) summation( delta * W)

        prev_activation = activations[i] # X_ji
        grad_weights[i] = np.dot(prev_activation.reshape(-1, 1), delta.reshape(1, -1)) # ^W = delta * X
        grad_biases[i] = delta # ^W0

    return grad_weights, grad_biases

def calculate_mse(Y_pred, Y_true):
    squared_errors = (Y_pred - Y_true) ** 2
    mse_compressor = np.mean(squared_errors[:, 0])
    mse_turbine = np.mean(squared_errors[:, 1])
    return mse_compressor, mse_turbine


def update_plot(fig, train_hist_comp, val_hist_comp, train_hist_turb, val_hist_turb):
    ax1, ax2 = fig.get_axes()
    ax1.cla()
    ax2.cla()

    epochs_range = range(1, len(train_hist_comp) + 1)

    # Plot for Compressor on the first axis
    ax1.plot(epochs_range, train_hist_comp, 'b-', label='Training MSE')
    ax1.plot(epochs_range, val_hist_comp, 'r-', label='Validation MSE')
    ax1.set_title('Compressor MSE Evolution')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('MSE (log scale)')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True)

    # Plot for Turbine on the second axis
    ax2.plot(epochs_range, train_hist_turb, 'b-', label='Training MSE')
    ax2.plot(epochs_range, val_hist_turb, 'r-', label='Validation MSE')
    ax2.set_title('Turbine MSE Evolution')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('MSE (log scale)')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    plt.draw()
    plt.pause(1)  # Pause to allow the plot to update


if __name__ == "__main__":
    try:
        data = np.loadtxt(FILE_NAME)
        print(f"Data loaded from {FILE_NAME}. Total shape: {data.shape}")
    except FileNotFoundError:
        print("ERROR")
        exit()

    #split input and output
    X_full = data[:, :16]
    Y_full = data[:, 16:]

    #randomly split into training, validation, and test set
    RANDOM_STATE = 45 #this allows us to always have the same split
    X_train, X_rest, Y_train, Y_rest = train_test_split(X_full, Y_full, test_size=0.5, random_state=RANDOM_STATE)
    X_val, X_test, Y_val, Y_test = train_test_split(X_rest, Y_rest, test_size=0.5, random_state=RANDOM_STATE)

    #normalize input
    mu_X_train = np.mean(X_train, axis=0)
    sigma_X_train = np.std(X_train, axis=0)
    sigma_X_train_safe = np.where(sigma_X_train == 0, 1, sigma_X_train)
    X_train_norm = (X_train - mu_X_train) / sigma_X_train_safe
    X_val_norm = (X_val - mu_X_train) / sigma_X_train_safe
    X_test_norm = (X_test - mu_X_train) / sigma_X_train_safe

    #normalize output
    mu_Y_train = np.mean(Y_train, axis=0)
    sigma_Y_train = np.std(Y_train, axis=0)
    sigma_Y_train_safe = np.where(sigma_Y_train == 0, 1, sigma_Y_train)
    Y_train_norm = (Y_train - mu_Y_train) / sigma_Y_train_safe
    Y_val_norm = (Y_val - mu_Y_train) / sigma_Y_train_safe
    Y_test_norm = (Y_test - mu_Y_train) / sigma_Y_train_safe

    #network parameters
    hidden_layer_sizes = [30,30]
    num_hidden_layer = len(hidden_layer_sizes)
    n_input = 16
    n_output = 2
    learning_rate = 0.01
    epochs = 15000 #maximum epochs

    # History lists for plotting
    training_mse_history_compressor = []
    training_mse_history_turbine = []
    validation_mse_history_compressor = []
    validation_mse_history_turbine = []

    #generate initial weights and biases
    weights = generate_weight(n_input, n_output, num_hidden_layer, hidden_layer_sizes)
    biases = generate_biases(num_hidden_layer, hidden_layer_sizes, n_output)

    # for plot creation
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    #start training
    for epoch in range(epochs):
        for i in range(len(X_train_norm)): #for each training input
            X = X_train_norm[i]
            Y = Y_train_norm[i]

            final_output, activations = forward_pass(X, weights, biases)
            grad_weights, grad_biases = backward_pass(Y, weights, biases, activations)

            #update the weights
            for j in range(len(weights)):
                weights[j] -= learning_rate * grad_weights[j]
                biases[j] -= learning_rate * grad_biases[j]

        # MSE CALCULATION AT THE END OF EACH EPOCH
        # calculate mse for both training and validation set, and save data for the final plots


        predictions_train_norm, _ = forward_pass(X_train_norm, weights, biases) #apply the function to the entire input set
        predictions_train = (predictions_train_norm * sigma_Y_train_safe) + mu_Y_train
        mse_train_comp, mse_train_turb = calculate_mse(predictions_train, Y_train)
        training_mse_history_compressor.append(mse_train_comp) # saving values for plot
        training_mse_history_turbine.append(mse_train_turb)


        predictions_val_norm, _ = forward_pass(X_val_norm, weights, biases)
        predictions_val = (predictions_val_norm * sigma_Y_train_safe) + mu_Y_train
        mse_val_comp, mse_val_turb = calculate_mse(predictions_val, Y_val)
        validation_mse_history_compressor.append(mse_val_comp)
        validation_mse_history_turbine.append(mse_val_turb)
        print(f"Epoch {epoch + 1:5d} | MSE Compressor (Train/Val): {mse_train_comp:.2e} / {mse_val_comp:.2e} | MSE Turbine (Train/Val): {mse_train_turb:.2e} / {mse_val_turb:.2e}")


    #update plot
    update_plot(fig, training_mse_history_compressor, validation_mse_history_compressor,training_mse_history_turbine, validation_mse_history_turbine)

    print("Training complete")

    print("FINAL EVALUATION ON TEST SET")
    predictions_test_norm, _ = forward_pass(X_test_norm, weights, biases)
    predictions_test = (predictions_test_norm * sigma_Y_train_safe) + mu_Y_train
    mse_test_compressor, mse_test_turbine = calculate_mse(predictions_test, Y_test)

    print(f"Final MSE on Test Set (Compressor): {mse_test_compressor:.4e}")
    print(f"Final MSE on Test Set (Turbine):    {mse_test_turbine:.4e}")

    print("Verifying Task Requirements")
    target_mse_compressor = 5.0e-8
    target_mse_turbine = 1.5e-8
    compressor_ok = mse_test_compressor <= target_mse_compressor
    turbine_ok = mse_test_turbine <= target_mse_turbine

    if compressor_ok and turbine_ok:
        print("Both MSE objectives have been reached.")
    else:
        print("Objective NOT reached")
        if not compressor_ok:
            print(f"  Compressor: MSE obtained {mse_test_compressor:.4e} > Threshold {target_mse_compressor:.4e}")
        if not turbine_ok:
            print(f"  Turbine: MSE obtained {mse_test_turbine:.4e} > Threshold {target_mse_turbine:.4e}")

    # Keep the final plot visible at the end
    plt.ioff()
    plt.show()
