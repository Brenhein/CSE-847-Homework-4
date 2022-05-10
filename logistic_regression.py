import matplotlib.pyplot as plt
from operator import itemgetter
import numpy as np
import math


def predict(x, w):
    """
    This function uses a sample and its weights to predict its values
    """
    x, w = np.array(x), np.array(w)
    return np.dot(x, w)


def sigmoid(x, w):
    """
    Calculates the output of the sigmoid function
    """
    s = predict(x, w)
    f = math.e ** (-1 * s)
    return 1 / (1 + f)


def error_gradient(w, data, labels):
    """
    This function calculates the error of the gradient with respect to w
    """
    error = 0
    for n in range(len(data)):
        xn, yn = data[n], labels[n]
        y_hat = sigmoid(xn, w)
        error += (y_hat - yn) * np.array(xn)
    return error


def gradient_descent(data, labels, epsilon, maxiter, n):
    """
    This function performs gradient descent to find the optimal vector of 
    of weights to use for logistic regression
    """
    wt = len(data[0]) * [0]
    diff, i = 1, 0
    
    # Update the weights for each step of the first order gradient
    while diff > epsilon and i <= maxiter:
        wt_old = wt
        i += 1
        
        # Move the gradient down
        e_grad = error_gradient(wt, data, labels)
        wt = wt - n * e_grad
        
        # Are we close enough to say we converged
        wt_diff = wt - wt_old
        diff = np.linalg.norm(wt_diff)

    return wt
        

def plotting(N, error):
    i = min(enumerate(error), key=itemgetter(1))[0]
    min_n = N[i]

    plt.ylabel("% Error")
    plt.xlabel("n")
    plt.title("Error for First Order Gradient Descent Logistic Regression")
    plt.plot(N, error)
    plt.axvline(x=min_n, color='g', linestyle='--',
                label="Minimum % Error (n={})".format(min_n))
    plt.legend(loc="upper left")
    plt.savefig("q1_err.png")
    plt.show()


def logistic_regression(data, labels, epsilon=1e-5, maxiter=1000):
    N = [200, 500, 800, 1000, 1500, 2000]
    error = []

    # Split the data sets
    for n in N:
        # Train the data
        train_x = data[:n]
        train_y = labels[:n]
        ws = gradient_descent(train_x, train_y, epsilon, maxiter, .0005)

        # Test the data ;)
        test_x = data[n:]
        test_y = labels[n:]
        incorrect = 0
        for i in range(len(test_x)):
            x = np.array(test_x[i])
            y = test_y[i]
            res = round(sigmoid(ws, x))
            if res != y:
                incorrect += 1

        # Get the error stuff
        e = round(incorrect / len(test_x), 3)
        error.append(e)
        print("n =", n, "error =", e)

    # Plot everything so it looks pretty
    plotting(N, error)


def main():
    # Opens the data file
    fpd = open("data.txt")
    data = []
    for line in fpd:
        line = line.replace("  ", " ")
        data.append([int(l[0]) for l in line.strip().split(" ")])
        data[-1].append(1)

    # Opens the label file
    fpl = open("labels.txt")
    labels = []
    for line in fpl:
        labels.append(float(line.strip()))

    logistic_regression(data, labels)


if __name__ == "__main__":
    main()

