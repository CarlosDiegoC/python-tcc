# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 07:38:23 2023

@author: Diego
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base = pd.read_csv("C:\\Users\\Diego\\projetos\\spyder_predicao_pjme\\base_dados_treinamento.csv")
base = base.dropna()
base_treinamento = base.iloc[:, 1:2].values

normalizador = MinMaxScaler(feature_range=(0,1))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)

previsores = []
consumo_real = []
for i in range(90, 6025):
    previsores.append(base_treinamento_normalizada[i-90:i, 0])
    consumo_real.append(base_treinamento_normalizada[i, 0])
previsores, consumo_real = np.array(previsores), np.array(consumo_real)
previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1], 1))

regressor = Sequential()
regressor.add(LSTM(units = 90, return_sequences = True, input_shape = (previsores.shape[1], 1)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 80, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 80, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 60))
regressor.add(Dropout(0.3))

regressor.add(Dense(units = 1, activation = 'sigmoid'))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error',
                  metrics = ['mean_absolute_error'])

es = EarlyStopping(monitor = 'loss', min_delta = 1e-10, patience = 10, verbose = 1)
rlr = ReduceLROnPlateau(monitor = 'loss', factor = 0.2, patience = 5, verbose = 1)
mcp = ModelCheckpoint(filepath = 'pesos.h5', monitor = 'loss', 
                      save_best_only = True, verbose = 1)
history = regressor.fit(previsores, consumo_real, epochs = 100, batch_size = 32,
              callbacks = [es, rlr, mcp])

base_teste = pd.read_csv('C:\\Users\\Diego\\projetos\\spyder_predicao_pjme\\base_dados_teste.csv')
consumo_real_teste = base_teste.iloc[:, 1:2].values
base_completa = pd.concat((base['PJME_MW'], base_teste['PJME_MW']), axis = 0)
entradas = base_completa[len(base_completa) - len(base_teste) - 90:].values
entradas = entradas.reshape(-1, 1)
entradas = normalizador.transform(entradas)

X_teste = []
for i in range(90, 121):
    X_teste.append(entradas[i-90:i, 0])
X_teste = np.array(X_teste)
X_teste = np.reshape(X_teste, (X_teste.shape[0], X_teste.shape[1], 1))
previsoes = regressor.predict(X_teste)
previsoes = normalizador.inverse_transform(previsoes)

consumo_real_teste.mean()
previsoes.mean()


plt.plot(consumo_real_teste, color = 'red', label = 'Consumo real')
plt.plot(previsoes, color = 'blue', label = 'Previsões')
plt.title('Previsão do consumo')
plt.xlabel('Tempo')
plt.ylabel('Consumo')
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(25, 10)) 
plt.title('MSE ao longo do treinamento') 
plt.ylabel('MSE') 
plt.xlabel('Época') 
plt.plot(history.history['mean_absolute_error']) 
plt.plot(history.history['val_mean_absolute_error']) 
plt.legend(['Treino', 'Validação'], loc='best') 
plt.savefig(r'C:\Users\Diego\projetos\spyder_predicao_pjme\fig01.png', bbox_inches='tight') 
plt.clf

mape = mean_absolute_percentage_error(consumo_real_teste, previsoes)
print(mape)
mae = mean_absolute_error(consumo_real_teste, previsoes)
print(mae)
mse = mean_squared_error(consumo_real_teste, previsoes)
print(mse)
