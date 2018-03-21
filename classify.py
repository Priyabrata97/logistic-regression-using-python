from sklearn.model_selection import train_test_split
import numpy as np
import pickle

def sigmoid(z):
    return 1/(1+np.exp(-z))


# cost function with regularisation
def cost(w,b,X,y,lmd=10):

    m = X.shape[0]
    z = np.matmul(X,w) + b
    hx = sigmoid(z)

    J = (-1/m)*( np.sum( y * np.log(hx) + (1. - y) * np.log(1.- hx) ) )

    J += (lmd/(2*m))* np.matmul(w,w)
    
    return J

# function to perform the gradient descent with regularisation
def gradient_descent(w,b,X,y,learning_rate=0.001,lmd=10,no_of_iteration=10000):

    m = X.shape[0]

    print("Initial cost: {}".format( cost(w,b,X,y) ))

    for i in range(no_of_iteration):

        z = np.matmul(X,w) + b
        hx = sigmoid(z)

        dw = (1/m)*np.matmul(X.T,hx-y)
        db = (1/m)*np.sum(hx-y)

        factor = 1-( (learning_rate * lmd)/m)
        
        w = w*factor - learning_rate*dw
        b = b - learning_rate*db

        if i % 100 == 0:
            print("Iteration {} cost :{}".format(i,cost(w,b,X,y)))
            
    print("Final cost: {}".format( cost(w,b,X,y) ))

    return w,b

def accuracy(w,b,X,y):

    m = X.shape[0]

    z = np.matmul(X,w) + b
    hx = sigmoid(z)

    pred = np.round(hx)

    correct_pred = (pred==y)

    total = np.sum(correct_pred)

    return (total*100)/m



def main():

    # The images where preprocessed beforehand and stored in pickle format
    with open("images.pickle","rb") as f:
        X_original = pickle.load(f)

    X_original = X_original.astype(np.float64)

    # The original shape: (200,64,64,3)
    # After reshaping : (200,64x64x3) = (200,12288)
    X = X_original.reshape(X_original.shape[0],-1)


    # Performing scaling so that convergence during gradient descent
    # happens faster than usual
    X = X/255

    # Number of samples(cats+dogs)
    m = X.shape[0]

    # Labels of the samples
    y = np.array([0]*(m//2) + [1]*(m//2))

    # Dividing the samples into train and test sets
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    # Initializing parameters for the model
    w = np.zeros(X.shape[1],dtype=np.float64)
    b = 0.0

    # Performing gradient descent
    w,b = gradient_descent(w,b,X_train,y_train)

    # Final output
    print("Train accuracy: {}".format( accuracy(w,b,X_train,y_train) ))
    print("Test accuracy: {}".format( accuracy(w,b,X_test,y_test) ))
    
if __name__ == "__main__":
    main()
