###### © Miguel Pardo 2022
------------
# Acerca de
Este programa pretende proporcionar un positivado base para todo un rollo a color a la vez (para asegurar consistencia dentro de cada rollo), con ajustes relativamente sencillos para cosas difíciles de lograr en programas como Lightroom.
El programa no pretende ser una solución para generar imágenes finales, por lo que se recomienda finalizar la edicición de las imágenes en Lr o Photoshop, los cuales ahora funcionarían de manera normal al ser una imágen positiva y no negativa. Tampoco pretende proporcionar imágenes fieles al proceso tradicional ni mantener características particulares de ciertos rollos, si el usuario así lo desea.

## Cómo funciona:
El programa decodifica, analiza y trabaja directamente sobre el el archivo RAW de la cámara en espacio lineal y sin realizar calibraciones de color. La razón de esto es que (a diferencia de Lr o Ps), se hace muchísimo más fácil analizar las imágenes antes de realizar ajustes no lineales que son complicados de revertir. Además, las correcciones de gamma y colores por defecto en estos programas, que funcionan muy bien para representar una foto normal de una escena común, no se ajustan a los valores óptimos para revelar fotos; y por lo demás varían entre cada combinación de cámara, iluminación y rollo, sobre todo si son rollos vencidos o mal procesados.

## Características:
- Consistencia para todas las fotos de un mismo rollo
- Recorte automático de negativos
- Cálculo automático de niveles blanco y negro
- Ajuste manual de gamma para cada canal RGB
- Mezclador de canales RGB
- Corrección de luminosidad base (si existe)
- Re-colorización rápida para tandas ya procesadas

## Limitaciones conocidas:
- El programa solo ha sido probado con una Canon 6D, y actualmente solo soporta archivos .CR2
- La interfaz podría ser más bonita
- El programa puede ser intensivo en recursos. Se recomienda tener al menos 16GB de RAM
- Solo soporta Windows

## Como usar FilmProcesser:
### Prerequisitos y suposiciones:
- Usas una cámara Canon para digitalizar rollos
- La exposición a lo largo de todo el rollo se mantiene constante, y sin recorte de blancos
- Los portanegativos usados se mantienen en la misma posición, y son muy cercanos a un negro puro en la imagen

### Uso básico:
- Juntar todas las fotos de un mismo rollo en una sola carpeta
- Ingresar o arrastrar la carpeta que contiene las fotos de los negativos
- Confirmar los diálogos que puedan aparecer
- Ingresar valores de colorización
	- Se recomienda partir con All-gamma
	- Luego el gamma correspondiente a cada canal
	- Después puntos de blanco/negro si es necesario
	- Finalmente el mezclador de canales
- Esperar a que el programa exporte las fotos

###Para la corrección de luminosidad:
- Sacar una foto al portanegativo, pero sin un negativo presente
- No es necesario que la foto tenga la misma exposición, pero se recomienda que tenga la misma apertura
- La foto no debe recortarse en los blancos
- Renombrar la foto como **"vig.cr2"**

###Controles en el visualizador:
- Z para ir a la imagen anterior
- X para ir a la imagen siguiente
- Esc para finalizar (**TIENE QUE FINALIZARSE CON ESC**, de lo contrario habrán errores)

##Sobre el tratamiento de los archivos:
- El programa mueve todos los CR2 presentes a una carpeta llamada "original"
- Dentro de esa carpeta se crea un archivo .npy para las previsualizaciones de las fotos
- También se crea un archivo "params.txt" con información sobre la colorización de las fotos

###Descripción de barras (en orden de aparición):
     
- Clipping:
	Activar este slider (valor 1) para mostrar áreas donde ocurra recorte de negros o blancos
- Compress shadows:
	Ajustar para levantar y comprimir valores negros. Útil para imágenes oscuras
- Black Point:
	Ajuste del nivel mínimo para el recorte de negros
- White Point:
	Ajuste del nivel mínimo para recorte de blancos
- All-gamma:

	Gamma global de la imagen. El valor específico está definido por $$4^{gamma / 50 - 1.6}$$
- R/G/B gamma:

	Gamma por canal de la imagen. El valor específico está definido por $$3^{gamma / 50 - 1.6}$$
- Autoset:
	Si está activado, el mezclador se ajustará automáticamente para mantenerse en niveles adecuados
- Normalize:
	Si se activa, reajustará los valores por canal del mezclador para evitar recortar luces
- Disable CCM:
	Si se activa, se puede ver una comparación de mezclador por defecto con el actual
- Reset:
	Si se activa, se devuelve al mezclador por defecto
- X-Y:
	Mezclador. Agrega un poco del canal Y al canal de salida X. Mismo funcionamiento que en Photoshop o programas similares. Por limitaciones de la interfaz, el punto medio (0%) es 250, por lo que 100% queda en 350.
            