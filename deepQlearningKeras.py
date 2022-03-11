# Author: Ing. Juan David Díaz
# Date: 07-03-2022
# Version: 1.0
# Proyect: Culebra IA con Python, Tenserflow, Keras, Pygame
# Proyect description: Ejercicio de aprendizaje de redes neuronales con Deep Q-learning
# Description: Clases BufferDeReproduccion para el almacenamiento en memoria 
#              y Agente para la construccion, entrenamiento y prediccion del modelo

from keras.layers import Dense
from keras.models import Sequential, load_model
import numpy as np
import tensorflow as tf
  
class BufferDeReproduccion(object):
    def __init__(self, tamMax, formaEntrada, numeroAcciones, discreto=False) -> None:
        self.tamMemoria = tamMax
        self.discreto = discreto
        self.memDeEstado = np.zeros((self.tamMemoria, formaEntrada))
        self.NuevaMemDeEstado = np.zeros((self.tamMemoria, formaEntrada))
        dtype = np.int8 if self.discreto else np.float32
        self.memoriaAccion = np.zeros((self.tamMemoria, numeroAcciones), dtype=dtype)
        self.memoriaRecompensa = np.zeros(self.tamMemoria)
        self.memoriaFinal = np.zeros(self.tamMemoria, dtype=np.float32)
        self.contadorMemoria = 0

    def guardarTransicion(self, estado, accion, recompensa, estadoSig, final):
        estado = np.array(estado)
        indice = self.contadorMemoria % self.tamMemoria
        self.memDeEstado[indice] = estado
        self.NuevaMemDeEstado[indice] = estadoSig
        self.memoriaAccion[indice] = accion
        self.memoriaRecompensa[indice] = recompensa
        self.memoriaFinal[indice] = 1 - int(final)
        if self.discreto:
            acciones = np.zeros(self.memoriaAccion.shape[1])
            acciones[accion] = 1.0
            self.memoriaAccion[indice] = acciones
        else:
            self.memoriaAccion[indice] = accion
        self.contadorMemoria += 1

    def getTransicion(self, indice):
        return self.memDeEstado[indice], self.memoriaAccion[indice], self.memoriaRecompensa[indice], self.NuevaMemDeEstado[indice], self.memoriaFinal[indice]

    def getTransiciones(self, indices):
        return self.memDeEstado[indices], self.memoriaAccion[indices], self.memoriaRecompensa[indices], self.NuevaMemDeEstado[indices], self.memoriaFinal[indices]

    def bufferVacio(self):
        return self.contadorMemoria == 0

    def bufferLleno(self):
        return self.tamMemoria == self.tamMemoria

    def muestraDelBuffer(self, tamBatch):
        memMax = min(self.tamMemoria, self.contadorMemoria)
        batch = np.random.choice(memMax, tamBatch)
        estado = self.memDeEstado[batch]
        nuevosEstados = self.NuevaMemDeEstado[batch]
        acciones = self.memoriaAccion[batch]
        recompensas = self.memoriaRecompensa[batch]
        final = self.memoriaFinal[batch]
        return estado, acciones, recompensas, nuevosEstados, final

class Agente(object):
    def __init__(self, alfa, gama, numeroAcciones, epsilon, tamBatch, numeroEntradas, decEpsilon =0.996, 
                 minEpsilon=0.01, tamMemoria=100000, nomArchivo='modeloDQN.h5') -> None:
        self.espacioAccion = [i for i in range(numeroAcciones)]
        self.alfa = alfa
        self.gama = gama
        self.epsilon = epsilon
        self.decEpsilon = decEpsilon
        self.minEpsilon = minEpsilon
        self.tamBatch = tamBatch
        self.tamMemoria = tamMemoria
        self.numeroAcciones = numeroAcciones
        self.numeroEntradas = numeroEntradas
        self.modelo = self.construirModelo(self.alfa, self.numeroAcciones, self.numeroEntradas, 256, 256, 256, 256)
        self.memoria = BufferDeReproduccion(self.tamMemoria, self.numeroEntradas, self.numeroAcciones, discreto=True)
        self.nomArchivo = nomArchivo

    def construirModelo(self, tasaAprendizaje, numeroAcciones, dimensionEntrada, dimensionFc1, dimensionFc2, dimensionFc3, dimensionFc4):
        # Modelo de red neuronal
        modelo = Sequential([
            Dense(dimensionFc1, input_shape=(dimensionEntrada,), activation='relu'),
            Dense(dimensionFc2, activation='relu'),
            Dense(dimensionFc3, activation='relu'),
            Dense(dimensionFc4, activation='relu'),
            Dense(numeroAcciones, activation='linear')
        ])
        modelo.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=tasaAprendizaje))
        return modelo

    def recordar(self, estado, accion, recompensa, estadoSig, final):
        self.memoria.guardarTransicion(estado, accion, recompensa, estadoSig, final)

    def elegirAccion(self, estado:np.ndarray) -> int:
        estado = estado[np.newaxis, :]
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.espacioAccion)
        else:
            return np.argmax(self.modelo.predict(estado))

    def aprender(self) -> None:
        if self.memoria.contadorMemoria < self.tamBatch:
            return
        estado, acciones, recompensas, nuevosEstados, final = self.memoria.muestraDelBuffer(self.tamBatch)
        valoresAcciones = np.array(self.espacioAccion, dtype=np.int8)
        indicesAcciones = np.dot(acciones, valoresAcciones)
        prediccion = self.modelo.predict(estado)
        sigPrediccion = self.modelo.predict(nuevosEstados)
        objetivo = prediccion.copy()
        indiceBatch = np.arange(self.tamBatch, dtype=np.int32)
        objetivo[indiceBatch, indicesAcciones] = recompensas + self.gama * np.max(sigPrediccion, axis=1)*final
        self.modelo.fit(estado, objetivo, verbose=0)
        if self.epsilon > self.minEpsilon:
            self.epsilon *= self.decEpsilon
        else:
            self.epsilon = self.minEpsilon

    def guardarModelo(self):
        try:
            self.modelo.save(self.nomArchivo)
            return True
        except:
            return False

    def cargarPesosModelo(self):
        try:
            self.modelo.load_weights(self.nomArchivo)
            return True
        except:
            return False
        

        