# Author: Ing. Juan David Díaz
# Date: 07-03-2022
# Version: 1.0
# Proyect: Culebra IA con Python, Tenserflow, Keras, Pygame
# Proyect description: Ejercicio de aprendizaje de redes neuronales con Deep Q-learning
# Description: Clase Parte

import pygame

class Parte:
    AZUL = (0, 0, 255)
    VERDE = (0, 255, 0)
    VERDE2 = (0, 180, 0)
    def __init__(self, x:int, y:int, esCabeza:bool, tam) -> None:
        self.x = x
        self.y = y 
        self.tam = tam
        self.esCabeza = esCabeza

    def dibujar(self, pantalla, direccion:int) -> None:
        color = self.AZUL if self.esCabeza else self.VERDE
        pygame.draw.rect(pantalla, color, [self.x, self.y, self.tam, self.tam], 0, 3)
        if self.esCabeza:
            try:
                carita = pygame.image.load("img/carita.png")
                if self.tam != 20:
                    carita = pygame.transform.scale(carita, (self.tam, self.tam))
                if direccion == 0:
                    carita = pygame.transform.rotate(carita, 90)
                elif direccion == 1:
                    carita = pygame.transform.rotate(carita, -90)
                elif direccion == 2:
                    carita = pygame.transform.rotate(carita, 180)
                pantalla.blit(carita, [self.x, self.y])
            except:
                pass
        else:
            pygame.draw.rect(pantalla, self.VERDE2, [self.x, self.y, self.tam, self.tam], 1, 3)
