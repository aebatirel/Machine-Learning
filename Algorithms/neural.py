import numpy as np

# Initialize w and b vector for every layer.
def initialize_weights(dimOfLayers):
    params = {}
    np.random.seed(3)
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
    dZ = np.multiply(dA ,np.multiply(s , (1-s)))

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
    db = 1 / m * np.sum(dZ, axis = 1)
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
p = {'b1': np.array([[ 0.04153939],
       [-1.11792545],
       [ 0.53905832]]), 'b2': np.array([[-0.74787095]]), 'W2':  np.array([[-0.5961597 , -0.0191305 ,  1.17500122]]), 'W1':  np.array([[-0.41675785, -0.05626683, -2.1361961 ,  1.64027081],
       [-1.79343559, -0.84174737,  0.50288142, -1.24528809],
       [-1.05795222, -0.90900761,  0.55145404,  2.29220801]])}
g = {'dW1':  np.array([[ 1.78862847,  0.43650985,  0.09649747, -1.8634927 ],
       [-0.2773882 , -0.35475898, -0.08274148, -0.62700068],
       [-0.04381817, -0.47721803, -1.31386475,  0.88462238]]), 'db2':  np.array([[ 0.98236743]]), 'dW2':  np.array([[-0.40467741, -0.54535995, -1.54647732]]), 'db1':  np.array([[ 0.88131804],
       [ 1.70957306],
       [ 0.05003364]])}

print(update_neurons(p,g,0.1))
