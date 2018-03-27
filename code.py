import numpy as np
from keras.layers import LSTM, Input, Dense
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split as split
from deap import base, creator, tools, algorithms
from keras.models import Model
from scipy.stats import bernoulli
#import tflearn
#from tflearn.metrics import Accuracy
from bitstring import BitArray

np.random.seed(1120)
data = pd.read_csv('weatherHistory.csv')
data = np.reshape(np.array(data['Apparent Temperature (C)'],data['Humidity']),(len(data['Formatted Date']),1))
train_data = data[0:90000]
test_data = data[90000:]

def prepare(data, win_size):
    A, B = np.empty((0,win_size)), np.empty((0))
    for i in range(len(data)-win_size-1):
        A = np.vstack([A,data[i:(i + win_size),0]])
        B = np.append(B,data[i + win_size,0])   
    A = np.reshape(A,(len(A),win_size,1))
    B = np.reshape(B,(len(B),1))
    return A, B

def evaluate(ga_sol):       
    win_size_bits = BitArray(ga_sol[0:6])
    number_units_bits = BitArray(ga_sol[6:]) 
    win_size = win_size_bits.uint
    number_units = number_units_bits.uint
    print('\nWindow Size: ', win_size, ', Num of Units: ', number_units)
    
    
    if win_size == 0 or number_units == 0:
        return 100, 
    
    
    A,B = prepare(train_data,win_size)
    A_train, A_val, B_train, B_val = split(A, B, test_size = 0.20, random_state = 1120)
    
    
    inputs = Input(shape=(win_size,1))
    A = LSTM(number_units, input_shape=(win_size,1))(inputs)
    predictions = Dense(1, activation='relu')(A)
    model = Model(inputs=inputs, outputs=predictions)
    #model = tflearn.DNN(predictions, tensorboard_verbose=3, tensorboard_dir="logs")
    model.compile(optimizer='adam',loss='mean_squared_error')
    model.fit(A_train, B_train, epochs=5, batch_size=10,shuffle=True)
    B_pred = model.predict(A_val)
    #model.save('./ZtrainedNet/final-model.tfl')      
    rmse = np.sqrt(mean_squared_error(B_val, B_pred))
    print('Validation RMSE: ', rmse,'\n')    
    return rmse,
population_size = 10
num_generations = 10
gene_length = 12

creator.create('FitnessMax', base.Fitness, weights = (-1.0,))
creator.create('Individual', list , fitness = creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register('binary', bernoulli.rvs, 0.5)
toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.binary, 
n = gene_length)
toolbox.register('population', tools.initRepeat, list , toolbox.individual)

toolbox.register('mate', tools.cxOrderedOrdered)
toolbox.register('mutate', tools.mutShuffleIndexes, indpbb = 0.6)
toolbox.register('select', tools.selRoulette)
toolbox.register('evaluate', evaluate)


population = toolbox.population(n = population_size)
r = algorithms.eaSimple(population, toolbox, cxpb = 0.4, mutpb = 0.1, 
ngen = num_generations, verbose = False)


best_individuals = tools.selBest(population,k = 1)
best_win_size = None
best_number_units = None

for bi in best_individuals:
    win_size_bits = BitArray(bi[0:6])
    number_units_bits = BitArray(bi[6:]) 
    best_win_size = win_size_bits.uint
    best_number_units = number_units_bits.uint
    print('\nWindow Size: ', best_win_size, ', Num of Units: ', best_number_units)


A_train,B_train = prepare(train_data,best_win_size)
A_test, B_test = prepare(test_data,best_win_size)

inputs = Input(shape=(best_win_size,1))
A = LSTM(best_number_units, input_shape=(best_win_size,1))(inputs)
predictions = Dense(1, activation='relu')(A)
model = Model(inputs = inputs, outputs = predictions)
model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(A_train, B_train, epochs=5, batch_size=10,shuffle=True)
B_pred = model.predict(A_test)

rmse = np.sqrt(mean_squared_error(B_test, B_pred))
print('Test RMSE: ', rmse)
