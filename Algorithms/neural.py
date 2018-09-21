import numpy as np

# Initialize w and b vector for every layer.
def initialize_weights(dimOfLayers):
    params = {}
    np.random.seed()
    L = len(dimOfLayers)
    for l in range(1,L):
        params['W' + str(l)] = np.random.randn(dimOfLayers[l],dimOfLayers[l-1]) * 0.01
        params['b' + str(l)] = np.zeros((dimOfLayers[l],1))
    return params

# Calculates W.A + b and returns that value with a cache.
def linear_forward(A, W, b):
    Z = np.dot(W,A) + b
    cache = (A, W, b)
    return Z,cache

def sigmoid( Z ):
    return 1 / (1 + np.exp(-1  * Z)) , Z

def sigmoid_backward(dA,cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = np.asarray(np.multiply(dA ,np.multiply(s , (1-s))))
    return dZ


def relu( Z ):
    return np.maximum(Z,0), Z

def relu_backward(dA,cache):
    Z = cache
    dZ = np.array(dA,copy = True)
    dZ[Z <= 0] = 0
    return dZ

#Activates neuron. First it propagates forward and activates with given function.
def activate_neuron(prevA, W, b, activationStr):
    Z,linear_cache = linear_forward(prevA, W ,b)
    if activationStr == "sigmoid":
        A,activation_cache = sigmoid(Z)
    elif activationStr == "relu":
        A,activation_cache = relu(Z)
    cache = (linear_cache,activation_cache)
    return A,cache

#Propagates all layers.
def forward_propagate_layers(X,parameters):
    cacheList = []
    A = X
    L = len(parameters) // 2
    for l in range(1,L):
        A_prev = A
        A, cache = activate_neuron(A_prev,parameters["W" + str(l)],parameters["b"+str(l)],"relu")
        cacheList.append(cache)
    AL, cache = activate_neuron(A,parameters["W" + str(L)],parameters["b"+str(L)],"sigmoid")
    cacheList.append(cache)
    return AL,cacheList

def calculate_cost(AL, Y):
    m = Y.shape[1]
    cost = np.sum( np.multiply(Y, np.log(AL) ) + np.multiply( 1 - Y, np.log(1 - AL) ) )
    cost = cost * -1/m
    cost = np.squeeze(cost) # to make the cost scalar.
    return cost

def linear_backward(dZ, cache):
    A_prev, W , b = cache
    m = A_prev.shape[1]
    dW = 1 / m * np.dot(dZ,A_prev.T)
    db = 1 / m * np.sum(dZ, axis = 1,keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "sigmoid":
        dZ = sigmoid_backward(dA,activation_cache)
    elif activation == "relu":
        dZ = relu_backward(dA,activation_cache)
    dA_prev, dW, db =  linear_backward(dZ,linear_cache)
    return dA_prev,dW,db

def backward_propagate_layers(AL,Y,caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y,AL) - np.divide(1 - Y , 1 - AL))
    current_cache = caches[L-1]
    grads["dA" + str(L)],grads["dW" + str(L)],grads["db" + str(L)] = linear_activation_backward(dAL,current_cache,"sigmoid")
    for l in reversed(range(L-1)):
        print(l)
        current_cache = caches[l]
        dA_prev_temp,dW_temp,db_temp = linear_activation_backward(grads["dA" + str(l+2)],current_cache,"relu")
        grads["dA" + str(l+1)] = dA_prev_temp
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp
    return grads

def update_neurons(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    return parameters


def train_model(X,Y,layer_dims,learning_rate,num_iterations):
    np.random.seed()
    X = np.asarray(X)
    costs = []
    parameters = initialize_weights(layer_dims)
    for i in range(0,num_iterations):
        #Forward propagation
        AL,caches = forward_propagate_layers(X,parameters)
        #Cost calculation
        cost = calculate_cost(AL,Y)
        #Back propagation
        grads = backward_propagate_layers(AL,Y,caches)
        #Updating parameters
        parameters = update_neurons(parameters,grads,learning_rate)
        if i % 100 == 0:
            costs.append(cost)
    return parameters,costs

def predict(test_data,params,confidence):
    m = test_data.shape[1]
    predictions = np.zeros((1,m))
    res,cachesList = forward_propagate_layers(test_data,params)
    for i in range(m):
        if res[0,i] >= confidence:
            predictions[0,i] = 1
        else:
            predictions[0,i] = 0
    return predictions

def visualize_cost(costs):
    fig,ax = plt.subplots()
    ax.plot(costs)
    ax.set(xlabel = 'iteration num',ylabel = 'cost',title = 'Cost values over iterations')
    plt.show()
