# Author: Ing. Juan David Díaz
# Date: 07-03-2022
# Version: 1.0
# Proyect: Culebra IA con Python, Tenserflow, Keras, Pygame
# Proyect description: Ejercicio de aprendizaje de redes neuronales con Deep Q-learning
# Description: Ejecución inicial del programa

from JuegoIA import JuegoIA
from Entrenamiento import Entrenamiento

ANCHO_VENTANA = 500
ALTO_VENTANA = 500
ALTO_INFO = 50
COLOR_BORDE = (12, 41, 119)
COLOR_TABLERO = (0,0,0)
TAM = 20

if __name__=="__main__":
    print("\033[1;34m"+"CULEBRITA IA con tensorflow y keras")
    print("\033[1;33m"+"Digite su opción:")
    print("\033[0;36m"+"1. Ver jugar a la IA")
    print("\033[0;36m"+"2. Entrenar la IA")
    opcion = int(input("\033[1;32m"+"Opcion: "+"\033[1;37m"))
    if opcion == 1:
        nJuegos = int(input("\033[1;32m"+"¿Cuantos juegos quiere que haga la IA?: "+"\033[1;37m"))
        JuegoIA(ANCHO_VENTANA, ALTO_VENTANA, ALTO_INFO, COLOR_BORDE, COLOR_TABLERO, TAM, 999999999999, nJuegos).loopJuego()
    elif opcion == 2:
        nJuegos = int(input("\033[1;32m"+"¿Cuantos juegos desea que entrene?: "+"\033[1;37m"))
        visualizar = input("\033[1;32m"+"¿Desea visualizar? S/N: "+"\033[1;37m")
        visualizar = visualizar == "S" or visualizar == "s"
        Entrenamiento(ANCHO_VENTANA, ALTO_VENTANA, ALTO_INFO, COLOR_BORDE, COLOR_TABLERO, TAM, 9999999999, nJuegos, visualizar).entrenar()
    else:
        print("\033[1;31m"+"Opcion no valida")
        input("\033[1;36m"+"Presione enter para continuar"+"\033[0;37m")
    