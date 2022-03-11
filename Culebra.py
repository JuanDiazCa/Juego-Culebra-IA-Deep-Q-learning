# Author: Ing. Juan David Díaz
# Date: 07-03-2022
# Version: 1.0
# Proyect: Culebra IA con Python, Tenserflow, Keras, Pygame
# Proyect description: Ejercicio de aprendizaje de redes neuronales con Deep Q-learning
# Description: Clase Culebra

from cmath import sqrt
from Parte import Parte
import numpy as np

class Culebra:    
    def __init__(self, x:int, y:int, tam:int) -> None:
        self.tam = tam
        self.comiendo=False
        self.cuerpo = []
        #direccion
        # derecha=0 ; izquierda=1 ; arriba=2 ; abajo=3
        self.direccion = 0
        self.cuerpo.append(Parte(x,y,True, tam))
        self.cuerpo.append(Parte(x-tam,y,False, tam))
        self.cuerpo.append(Parte(x-(2*tam),y,False, tam))

    def dibujar(self, pantalla) -> None:
        for parte in self.cuerpo:
            parte.dibujar(pantalla, self.direccion)
    
    def comer(self, x:int, y:int):#agregar a la ultima posicion
        self.cuerpo.append(Parte(x,y,False, self.tam))
    
    def cambiarDireccion(self, direccion:int):
        self.direccion = direccion

    def mover(self) -> None:
        pasoX = 0
        pasoY = 0
        if(self.direccion == 0):#derecha
            pasoX += self.tam
        if(self.direccion == 1):#izquierda
            pasoX -= self.tam
        if(self.direccion == 2):#arriba
            pasoY -= self.tam
        if(self.direccion == 3):#abajo
            pasoY += self.tam
        cola = self.cuerpo[-1]
        for i in reversed(range(1,len(self.cuerpo))):
            self.cuerpo[i].x = self.cuerpo[i-1].x      
            self.cuerpo[i].y = self.cuerpo[i-1].y   
        self.cuerpo[0].x = self.cuerpo[0].x + pasoX     
        self.cuerpo[0].y = self.cuerpo[0].y + pasoY
        if(self.comiendo):
            self.comer(cola.x,cola.y)

    def getCabezaCoord(self):
        return self.cuerpo[0].x, self.cuerpo[0].y

    def verificarColision(self, ancho:int, alto:int) -> bool:
        #verificar colision con bordes
        x, y = self.getCabezaCoord()
        if x == 0 or x == ancho - self.tam or y == 0 or y == alto - self.tam:
            return True
        #verificar colision cuerpo
        for i in reversed(range(1,len(self.cuerpo))):
            if self.cuerpo[i].x == x and self.cuerpo[i].y == y:
                return True
        return False

    def getDistanciaAPunto(self, x:int, y:int) -> float:
        return sqrt(abs((self.cuerpo[0].x - x)/self.tam)**2 + abs((self.cuerpo[0].y - y)/self.tam)**2).real
 
    def getDistanciaACola(self) -> float:
        return self.getDistanciaAPunto(self.cuerpo[-1].x, self.cuerpo[-1].y)

    def getVisionCompleta(self,frutaX:int, frutaY:int, ancho:int, alto:int) -> np.ndarray:
        escenario = np.zeros((ancho//self.tam, alto//self.tam), dtype=np.int8)
        #agregar las paredes como -1 al escenario
        for i in range(ancho//self.tam):
            escenario[i][0] = -1
            escenario[i][(alto//self.tam)-1] = -1
        for i in range(alto//self.tam):
            escenario[0][i] = -1
            escenario[(ancho//self.tam)-1][i] = -1
        #agregar las frutas como 1 al escenario
        escenario[int(frutaX/self.tam)][int(frutaY/self.tam)] = 1
        #agregar la culebra como -1 al escenario la cabeza como 2
        for i in range(len(self.cuerpo)):
            if i == 0:
                escenario[int(self.cuerpo[i].x/self.tam)][int(self.cuerpo[i].y/self.tam)] = 2
            else:
                escenario[int(self.cuerpo[i].x/self.tam)][int(self.cuerpo[i].y/self.tam)] = -1
        #devolver el escenario como vector junto con la direccion en la ultima posicion
        #for f in escenario: print(str(f)+"\n") if mostrar else None
        return np.append(escenario.flatten(), [self.direccion])


    def getVision(self, frutaX:int, frutaY:int, ancho:int, alto:int) -> list:
        vision = []
        #vision derecha
        vision.append(self.getVisionDerecha(frutaX, frutaY, ancho))
        #vision arriba derecha
        vision.append(self.getVisionArribaDerecha(frutaX, frutaY, ancho))
        #vision arriba
        vision.append(self.getVisionArriba(frutaX, frutaY))
        #vision arriba izquierda
        vision.append(self.getVisionArribaIzquierda(frutaX, frutaY))
        #vision izquierda
        vision.append(self.getVisionIzquierda(frutaX, frutaY))
        #vision abajo izquierda
        vision.append(self.getVisionAbajoIzquierda(frutaX, frutaY, alto))
        #vision abajo
        vision.append(self.getVisionAbajo(frutaX, frutaY, alto))
        #vision abajo derecha
        vision.append(self.getVisionAbajoDerecha(frutaX, frutaY, alto, ancho))
        return vision

    def getVisionDerecha(self, frutaX:int, frutaY:int, ancho:int) -> int:
        x, y = self.getCabezaCoord()
        #vision derecha
        #verificar si es una parte del cuerpo diferente a la cabeza
        for i in range(1,len(self.cuerpo)):
            if x + self.tam == self.cuerpo[i].x and y == self.cuerpo[i].y:
                return -1
        #verificar si es pared
        if x + self.tam == ancho:
            return -1
        #verificar si hay fruta
        if x + self.tam == frutaX and y == frutaY:
            return 1
        return 0

    def getVisionIzquierda(self, frutaX:int, frutaY:int) -> int:
        x, y = self.getCabezaCoord()
        #vision izquierda
        #verificar si es una parte del cuerpo diferente a la cabeza
        for i in range(1,len(self.cuerpo)):
            if x - self.tam == self.cuerpo[i].x and y == self.cuerpo[i].y:
                return -1
        #verificar si es pared
        if x - self.tam == 0:
            return -1
        #verificar si hay fruta
        if x - self.tam == frutaX and y == frutaY:
            return 1
        return 0

    def getVisionArriba(self, frutaX:int, frutaY:int) -> int:
        x, y = self.getCabezaCoord()
        #vision arriba
        #verificar si es una parte del cuerpo diferente a la cabeza
        for i in range(1,len(self.cuerpo)):
            if x == self.cuerpo[i].x and y - self.tam == self.cuerpo[i].y:
                return -1
        #verificar si es pared
        if y - self.tam == 0:
            return -1
        #verificar si hay fruta
        if x == frutaX and y - self.tam == frutaY:
            return 1
        return 0

    def getVisionAbajo(self, frutaX:int, frutaY:int, alto:int) -> int:
        x, y = self.getCabezaCoord()
        #vision abajo
        #verificar si es una parte del cuerpo diferente a la parte posterior a la cabeza
        for i in range(1,len(self.cuerpo)):
            if x == self.cuerpo[i].x and y + self.tam == self.cuerpo[i].y:
                return -1
        #verificar si es pared
        if y + self.tam == alto:
            return -1
        #verificar si hay fruta
        if x == frutaX and y + self.tam == frutaY:
            return 1
        return 0

    def getVisionArribaDerecha(self, frutaX:int, frutaY:int, ancho:int) -> int:
        x, y = self.getCabezaCoord()
        #vision arriba derecha
        #verificar si es una parte del cuerpo diferente a la cabeza
        for i in range(1,len(self.cuerpo)):
            if x + self.tam == self.cuerpo[i].x and y - self.tam == self.cuerpo[i].y:
                return -1
        #verificar si es pared
        if y - self.tam == 0 or x + self.tam == ancho:
            return -1
        #verificar si hay fruta
        if x + self.tam == frutaX and y - self.tam == frutaY:
            return 1
        return 0

    def getVisionArribaIzquierda(self, frutaX:int, frutaY:int) -> int:
        x, y = self.getCabezaCoord()
        #vision arriba izquierda
        #verificar si es una parte del cuerpo diferente a la cabeza
        for i in range(1,len(self.cuerpo)):
            if x - self.tam == self.cuerpo[i].x and y - self.tam == self.cuerpo[i].y:
                return -1
        #verificar si es pared
        if y - self.tam == 0 or x - self.tam == 0:
            return -1
        #verificar si hay fruta
        if x - self.tam == frutaX and y - self.tam == frutaY:
            return 1
        return 0

    def getVisionAbajoDerecha(self, frutaX:int, frutaY:int, alto:int, ancho:int) -> int:
        x, y = self.getCabezaCoord()
        #vision abajo derecha
        #verificar si es una parte del cuerpo diferente a la cabeza
        for i in range(1,len(self.cuerpo)):
            if x + self.tam == self.cuerpo[i].x and y + self.tam == self.cuerpo[i].y:
                return -1
        #verificar si es pared
        if y + self.tam == alto or x + self.tam == ancho:
            return -1
        #verificar si hay fruta
        if x + self.tam == frutaX and y + self.tam == frutaY:
            return 1
        return 0

    def getVisionAbajoIzquierda(self, frutaX:int, frutaY:int, alto:int) -> int:
        x, y = self.getCabezaCoord()
        #vision abajo izquierda
        #verificar si es una parte del cuerpo diferente a la cabeza
        for i in range(1,len(self.cuerpo)):
            if x - self.tam == self.cuerpo[i].x and y + self.tam == self.cuerpo[i].y:
                return -1
        #verificar si es pared
        if y + self.tam == alto or x - self.tam == 0:
            return -1
        #verificar si hay fruta
        if x - self.tam == frutaX and y + self.tam == frutaY:
            return 1
        return 0

    def getOrientacionAPunto(self, px:int, py:int) -> list:
        x, y = self.getCabezaCoord()
        ox, oy = ((px-self.tam)/self.tam) - (x/self.tam), ((py-self.tam)/self.tam) - (y/self.tam)
        return [ox, oy]

    def getOrientacionACola(self) -> list:
        return self.getOrientacionAPunto(self.cuerpo[-1].x, self.cuerpo[-1].y)

    def getDistanciaAParedes(self, ancho:int, alto:int) -> list:
        x, y = self.getCabezaCoord()
        return [((ancho-self.tam) - x)/self.tam, x/self.tam, ((alto-self.tam) - y)/-self.tam, y/self.tam]

    def getObservacionEstado(self, frutaX:int, frutaY:int, ancho:int, alto:int) -> list:
        #agregar lo que ve (8)
        estado:list = self.getVision(frutaX, frutaY, ancho, alto)
        #agregar su direccion actual (1)
        estado.append(self.direccion)
        #agregar la distancia a la fruta (1)
        estado.append(self.getDistanciaAPunto(frutaX, frutaY))
        #agregar la orientacion de la fruta (2)
        estado.extend(self.getOrientacionAPunto(frutaX, frutaY)) 
        #agregar la distancia a su cola (1)
        estado.append(self.getDistanciaACola())
        #agregar la orientacion de su cola (2)
        estado.extend(self.getOrientacionACola())
        #agregar la distancia a las paredes (4)
        estado.extend(self.getDistanciaAParedes(ancho, alto))
        return np.array(estado)