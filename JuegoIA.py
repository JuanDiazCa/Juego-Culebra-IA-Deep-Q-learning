# Author: Ing. Juan David Díaz
# Date: 07-03-2022
# Version: 1.0
# Proyect: Culebra IA con Python, Tenserflow, Keras, Pygame
# Proyect description: Ejercicio de aprendizaje de redes neuronales con Deep Q-learning
# Description: Clase JuegoIA para ver a la culebra jugar con IA usando el modelo de la red neuronal

import sys
import pygame
from Culebra import Culebra
from Fruta import Fruta
from deepQlearningKeras import Agente
import numpy as np

class JuegoIA:
    def __init__(self, ANCHO_VENTANA:int, ALTO_VENTANA:int, 
    ALTO_INFO:int, COLOR_BORDE:tuple, COLOR_TABLERO:tuple, 
    TAM:int, VEL:int, nJuegos:int) -> None:
        self.ANCHO_VENTANA = ANCHO_VENTANA
        self.ALTO_VENTANA = ALTO_VENTANA
        self.ALTO_INFO = ALTO_INFO
        self.COLOR_BORDE = COLOR_BORDE
        self.COLOR_TABLERO = COLOR_TABLERO
        self.TAM = TAM
        self.VEL = VEL  
        self.maxBatch = 64
        self.nJuegos = nJuegos
        self.historialEpsilon = []
        self.puntajes = []
        self.nomArchivoEpsilon = "epsilon.txt"
        nEntradas = 19
        self.agente = Agente(0.005, 0.99, 4, self.cargarEpsilon(self.nomArchivoEpsilon), self.maxBatch, nEntradas)
        print("\033[1;33m"+"Cargando agente...")
        print("\033[1;32m"+"Modelo cargado correctamente") if self.agente.cargarPesosModelo() else print("\033[1;31m"+"No se encontro el modelo")
        print("\033[1;34m"+"Ver a la IA en juego")
        
    def cargarEpsilon(self, archivo:str)->float:
        try:
            with open(archivo, "r") as archivo:
                for linea in archivo:
                    self.historialEpsilon.append(float(linea))
                archivo.close()
            return float(self.historialEpsilon[-1])
        except:
            return 1.0
        
    def loopJuego(self):
        pygame.init()
        pygame.display.set_caption("Culebrita")
        pantalla = pygame.display.set_mode((self.ANCHO_VENTANA,self.ALTO_VENTANA + self.ALTO_INFO))
        reloj = pygame.time.Clock()
        jugando = True
        gameOver = False
        completado = False
        puntaje = 0
        culebra = Culebra(int((self.ANCHO_VENTANA/2)/self.TAM)*self.TAM, int((self.ALTO_VENTANA/2)/self.TAM)*self.TAM, self.TAM)
        fruta = Fruta(culebra.cuerpo[0].x+self.TAM*2,culebra.cuerpo[0].y, self.TAM)
        fruta.cambiarCoordenadas(culebra, self.ANCHO_VENTANA, self.ALTO_VENTANA)
        nJuego = 0
        frame = 0
        # main loop
        while jugando:
            # main loop
            while not gameOver and not completado:
                pygame.display.update()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                observacion = culebra.getObservacionEstado(fruta.x, fruta.y, self.ANCHO_VENTANA, self.ALTO_VENTANA).copy()
                #observacion = culebra.getVisionCompleta(fruta.x, fruta.y, self.ANCHO_VENTANA, self.ALTO_VENTANA)
                accion = self.agente.elegirAccion(observacion)
                culebra.direccion = accion
                self.dibujarTablero(pantalla)
                self.mostrarInfo(pantalla, puntaje)
                culebra.mover()
                culebra.dibujar(pantalla)
                fruta.dibujar(pantalla)
                if fruta.getCoord() == culebra.getCabezaCoord():
                    puntaje += 10
                    celdasX = int((self.ANCHO_VENTANA-(2*self.TAM))/self.TAM)
                    celdasY = int((self.ALTO_VENTANA-(2*self.TAM))/self.TAM)
                    if len(culebra.cuerpo) == (celdasX*celdasY)-1:
                        completado = True
                        self.dibujarTablero(pantalla)
                        self.mostrarInfo(pantalla, puntaje)
                        culebra.dibujar(pantalla)
                    else:
                        culebra.comiendo = True
                        fruta.cambiarCoordenadas(culebra, self.ANCHO_VENTANA, self.ALTO_VENTANA)
                else:
                    culebra.comiendo = False
                gameOver = culebra.verificarColision(self.ANCHO_VENTANA, self.ALTO_VENTANA) 
                reloj.tick(self.VEL)
                punPromedio = np.mean(self.puntajes) if len(self.puntajes)>0  else 0
                print("\r"+"\033[1;33m"+"Juego:"+"\033[1;37m", nJuego+1, "\033[1;33m"+"Puntaje:"+"\033[1;37m", puntaje,"\033[1;33m"+"Puntaje promedio:"+"\033[1;37m", "{0:.4f}".format(punPromedio), "\033[1;33m"+"Maximo puntaje:"+"\033[1;37m", max(self.puntajes) if len(self.puntajes)>0 else 0, end="          ")
                if(nJuego==self.nJuegos):
                    jugando = False
                    print("\033[1;32m"+"\nJuegos Terminados"+"\033[0;37m")
                    pygame.quit()
                    sys.exit()
                #pygame.image.save(pantalla, "img/frames/Juego"+str(nJuego)+"f"+str(frame)+".png")
                frame += 1
            self.puntajes.append(puntaje)
            if gameOver:
                frame = 0
                nJuego += 1
                gameOver = not jugando
                puntaje = 0
                culebra = Culebra(int((self.ANCHO_VENTANA/2)/self.TAM)*self.TAM, int((self.ALTO_VENTANA/2)/self.TAM)*self.TAM, self.TAM)
                fruta = Fruta(culebra.cuerpo[0].x+self.TAM*2,culebra.cuerpo[0].y, self.TAM)
                fruta.cambiarCoordenadas(culebra, self.ANCHO_VENTANA, self.ALTO_VENTANA)    
            else:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_RETURN:
                            gameOver = False
                            completado = False
                            puntaje = 0
                            culebra = Culebra(int((self.ANCHO_VENTANA/2)/self.TAM)*self.TAM, int((self.ALTO_VENTANA/2)/self.TAM)*self.TAM, self.TAM)
                            fruta = Fruta(0,0, self.TAM)
                            fruta.cambiarCoordenadas(culebra, self.ANCHO_VENTANA, self.ALTO_VENTANA)
                    elif event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
            pygame.display.update()
        pygame.quit()
        sys.exit()

    def mostrarInfo(self, pantalla, puntaje):
        fuente = pygame.font.match_font('consolas')
        tipo_letra = pygame.font.Font(fuente,20)
        superficie = tipo_letra.render("Puntaje: "+ str(puntaje),True, (233,234,237))
        rectangulo = superficie.get_rect()
        rectangulo.center = (int(self.ANCHO_VENTANA/2), int(self.ALTO_VENTANA + self.ALTO_INFO/2))
        pantalla.blit(superficie,rectangulo)

    def dibujarTablero(self, pantalla):
        pantalla.fill(self.COLOR_BORDE)
        pygame.draw.rect(pantalla, self.COLOR_TABLERO, [self.TAM, self.TAM, self.ANCHO_VENTANA-(2*self.TAM), self.ALTO_VENTANA-(2*self.TAM)], 0)
