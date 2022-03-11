# Author: Ing. Juan David Díaz
# Date: 07-03-2022
# Version: 1.0
# Proyect: Culebra IA con Python, Tenserflow, Keras, Pygame
# Proyect description: Ejercicio de aprendizaje de redes neuronales con Deep Q-learning
# Description: Clase Entrenamiento para entrenar la red neuronal

import sys
from Culebra import Culebra
from Fruta import Fruta
from deepQlearningKeras import Agente
import numpy as np
import pygame

class Entrenamiento:
    def __init__(self, ANCHO_VENTANA:int, ALTO_VENTANA:int, 
    ALTO_INFO:int, COLOR_BORDE:tuple, COLOR_TABLERO:tuple, 
    TAM:int, VEL:int, nJuegos:int, visualizar:bool) -> None:
        self.visualizar = visualizar
        self.ALTO_INFO = ALTO_INFO
        self.COLOR_BORDE = COLOR_BORDE
        self.COLOR_TABLERO = COLOR_TABLERO
        self.VEL = VEL
        self.ANCHO_VENTANA = ANCHO_VENTANA
        self.ALTO_VENTANA = ALTO_VENTANA
        self.TAM = TAM
        self.maxBatch = 64
        self.nJuegos = nJuegos
        self.historialEpsilon = []
        self.puntajes = []
        self.nomArchivoEpsilon = "epsilon.txt"
        nEntradas = 19
        self.agente = Agente(0.005, 0.99, 4, self.cargarEpsilon(self.nomArchivoEpsilon), self.maxBatch, nEntradas)
        print("\033[1;33m"+"Cargando agente...")
        print("\033[1;32m"+"Modelo cargado correctamente") if self.agente.cargarPesosModelo() else print("\033[1;31m"+"No se encontro el modelo")
        
    def cargarEpsilon(self, archivo:str)->float:
        try:
            with open(archivo, "r") as archivo:
                for linea in archivo:
                    self.historialEpsilon.append(float(linea))
                archivo.close()
            return float(self.historialEpsilon[-1])
        except:
            return 1.0

    def guardarEpsilon(self, archivo:str, epsilon:float)->None:
        try:
            with open(archivo, "a") as archivo:
                archivo.write(str(epsilon) + "\n")
                archivo.close()
        except:
            try:
                with open(archivo, "w") as archivo:
                    archivo.write(str(epsilon) + "\n")
                    archivo.close()
            except:
                print("\033[1;31m"+"Error al guardar epsilon")
        
    def entrenar(self):
        if self.visualizar:
            pygame.init()
            pygame.display.set_caption("Culebrita")
            pantalla = pygame.display.set_mode((self.ANCHO_VENTANA,self.ALTO_VENTANA + self.ALTO_INFO))
            reloj = pygame.time.Clock()
        entrenando = True
        gameOver = False
        completado = False
        puntaje = 0
        culebra = Culebra(int((self.ANCHO_VENTANA/2)/self.TAM)*self.TAM, int((self.ALTO_VENTANA/2)/self.TAM)*self.TAM, self.TAM)
        fruta = Fruta(culebra.cuerpo[0].x+self.TAM*2,culebra.cuerpo[0].y, self.TAM)
        fruta.cambiarCoordenadas(culebra, self.ANCHO_VENTANA, self.ALTO_VENTANA)
        nJuego = 0
        recompensa = 0
        cFruta = 0
        ccFruta = 0
        accion = 0
        pasos = 0
        while entrenando:
            while not gameOver and not completado:
                if self.visualizar:
                    pygame.display.update()
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            print("\033[1;32m"+"\nModelo Guardado"+"\033[0;37m") if self.agente.guardarModelo() else print("\033[1;31m"+"\nNo se pudo guardar el modelo"+"\033[0;37m")
                            pygame.quit()
                            sys.exit()
                observacion = culebra.getObservacionEstado(fruta.x, fruta.y, self.ANCHO_VENTANA, self.ALTO_VENTANA).copy()
                #observacion = culebra.getVisionCompleta(fruta.x, fruta.y, self.ANCHO_VENTANA, self.ALTO_VENTANA)
                accion = self.agente.elegirAccion(observacion)
                culebra.cambiarDireccion(accion)
                if self.visualizar:
                    self.dibujarTablero(pantalla)
                    self.mostrarInfo(pantalla, puntaje)
                culebra.mover()
                pasos += 1
                if self.visualizar:
                    culebra.dibujar(pantalla)
                    fruta.dibujar(pantalla)
                if fruta.getCoord() == culebra.getCabezaCoord():
                    recompensa+=10000+(len(culebra.cuerpo)*100)-pasos
                    recompensa = 1000 if recompensa < 1000 else recompensa
                    cFruta+=1
                    pasos = 0
                    celdasX = int((self.ANCHO_VENTANA-(2*self.TAM))/self.TAM)
                    celdasY = int((self.ALTO_VENTANA-(2*self.TAM))/self.TAM)
                    if len(culebra.cuerpo) == (celdasX*celdasY)-1:
                        completado = True
                        recompensa += 100000000
                    else:
                        culebra.comiendo = True
                        fruta.cambiarCoordenadas(culebra, self.ANCHO_VENTANA, self.ALTO_VENTANA)
                else:
                    culebra.comiendo = False
                    if observacion[9] == 1:
                        recompensa-=10
                gameOver = culebra.verificarColision(self.ANCHO_VENTANA, self.ALTO_VENTANA)
                if self.visualizar:
                    reloj.tick(self.VEL)
                if gameOver:
                    recompensa -= 15000
                    if culebra.cuerpo[0].x == culebra.cuerpo[1].x and culebra.cuerpo[0].y == culebra.cuerpo[1].y :
                        recompensa -= 5000
                    nJuego+=1
                nuevaObservacion = culebra.getObservacionEstado(fruta.x, fruta.y, self.ANCHO_VENTANA, self.ALTO_VENTANA).copy()
                #nuevaObservacion = culebra.getVisionCompleta(fruta.x, fruta.y, self.ANCHO_VENTANA, self.ALTO_VENTANA)
                self.agente.recordar(observacion, accion, recompensa, nuevaObservacion, gameOver)
                self.agente.aprender()
                punPromedio = np.mean(self.puntajes) if len(self.puntajes)>0  else 0
                epsilonPromedio = np.mean(self.historialEpsilon) if len(self.historialEpsilon)>0  else 0
                puntaje += recompensa
                recompensa = 0
                print("\r"+"\033[1;33m"+"Juego:"+"\033[1;37m", nJuego+1, "\033[1;33m"+"Puntaje:"+"\033[1;37m", 
                puntaje,"\033[1;33m"+"Puntaje promedio:"+"\033[1;37m", "{0:.2f}".format(punPromedio), 
                "\033[1;33m"+"Epsilon promedio:"+"\033[1;37m", "{0:.2f}".format(epsilonPromedio), 
                "\033[1;33m"+"Max puntaje:"+"\033[1;37m", max(self.puntajes) if len(self.puntajes)>0 else 0, 
                "\033[1;33m"+"Max Frutas:"+"\033[1;37m", ccFruta, "\033[1;33m"+"Distancia fruta:"+"\033[34m",
                "{0:.2f}".format(nuevaObservacion[9]), end="       ")
                if(nJuego==self.nJuegos):
                    entrenando = False
                    print("\033[1;32m"+"\nModelo Guardado"+"\033[0;37m") if self.agente.guardarModelo() else print("\033[1;31m"+"\nNo se pudo guardar el modelo"+"\033[0;37m")
                    self.guardarEpsilon(self.nomArchivoEpsilon, self.agente.epsilon)
                    if self.visualizar:
                        pygame.quit()
                    sys.exit()
            self.guardarEpsilon(self.nomArchivoEpsilon, self.agente.epsilon)
            self.historialEpsilon.append(self.agente.epsilon)
            self.puntajes.append(puntaje)
            if gameOver or completado:
                gameOver = False
                completado = False
                puntaje = 0
                recompensa = 0
                accion = 0
                pasos = 0
                culebra = Culebra(int((self.ANCHO_VENTANA/2)/self.TAM)*self.TAM, int((self.ALTO_VENTANA/2)/self.TAM)*self.TAM, self.TAM)
                fruta = Fruta(culebra.cuerpo[0].x+self.TAM*2,culebra.cuerpo[0].y, self.TAM)
                fruta.cambiarCoordenadas(culebra, self.ANCHO_VENTANA, self.ALTO_VENTANA)
                if cFruta > ccFruta:
                    ccFruta = cFruta
                cFruta = 0
            if self.visualizar:
                pygame.display.update()
        if self.visualizar:
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
