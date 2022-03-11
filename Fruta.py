# Author: Ing. Juan David Díaz
# Date: 07-03-2022
# Version: 1.0
# Proyect: Culebra IA con Python, Tenserflow, Keras, Pygame
# Proyect description: Ejercicio de aprendizaje de redes neuronales con Deep Q-learning
# Description: Clase Fruta

import pygame
import random

class Fruta:
    ROJO = (255, 0, 0)
    def __init__(self,x:int,y:int, tam) -> None:
        self.x = x
        self.y = y
        self.tam = tam
 
    def dibujar(self, pantalla) -> None:
        pygame.draw.ellipse(pantalla, self.ROJO, [self.x, self.y, self.tam, self.tam], 0)

    def cambiarCoordenadas(self,culebrita, ancho, alto):
        choca = True
        while choca:
            x = int(random.randint(self.tam, ancho - (2*self.tam))/self.tam)*self.tam
            y = int(random.randint(self.tam, alto - (2*self.tam))/self.tam)*self.tam
            choca = False
            for parte in culebrita.cuerpo:
                if parte.x == x and parte.y == y:
                    choca = True
        self.x = x
        self.y = y
        return x,y

    def getCoord(self):
        return self.x, self.y
