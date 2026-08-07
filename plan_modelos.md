# Seleccion de modelos y plan de procesamiento de imagenes

Documento del avance. Define que modelos se van a entrenar, por que se eligieron,
bajo que protocolo se comparan y que transformaciones de imagen tienen sentido
para este problema.

---

## Protocolo comun de comparacion

Todos los modelos se entrenan sobre el mismo tensor `02_tensores.npz`, con la misma
semilla y la misma particion, y comparten el presupuesto de entrenamiento. Lo que
queda fijo y lo que queda libre se resume asi:

| Fijo para todos | Libre en cada modelo |
|---|---|
| Datos, semilla y particion train/val/test | Arquitectura |
| Tope de 30 epocas con EarlyStopping (paciencia 4) | Optimizador y learning rate |
| Metricas registradas y formato de salida | Batch size |
| Restauracion de los mejores pesos | Regularizacion (dropout, batchnorm, L2) |

La razon de separarlo asi es que el enunciado pide probar varias configuraciones
hasta encontrar la mejor, de tal forma que los hiperparametros deben poder variar;
no obstante, si tambien variara el presupuesto de entrenamiento no se podria saber
si un modelo gano por su arquitectura o simplemente porque se entreno mas tiempo.
Bajo esta idea, se fija el presupuesto y se libera el diseño.

La seleccion entre variantes se hace sobre **validacion**. La particion de prueba
se reserva y se usa una sola vez, en la etapa de comparacion entre las tres familias
de modelos, para poder reportar un numero que no este contaminado por las decisiones
que se tomaron al elegir.

### Metricas

Se registra accuracy, la matriz de confusion y el tiempo de entrenamiento.

Cabe mencionar que no se reportan precision, recall ni F1. Estas metricas son utiles
cuando las clases estan desbalanceadas o cuando un tipo de error cuesta mas que el
otro, y en este caso no ocurre ninguna de las dos cosas: el dataset tiene exactamente
3,000 imagenes por clase, y confundir una letra con otra tiene el mismo costo sin
importar cual sea el par, ya que no hay una clase "positiva" cuyo falso positivo
implique un riesgo distinto al de un falso negativo. Dicho esto, la accuracy resume
el desempeño de forma honesta y la matriz de confusion aporta el detalle por clase,
que es donde realmente interesa mirar, ya que ahi se ve si el error se concentra en
los grupos visualmente similares.

El tiempo de entrenamiento si se registra, porque es el que sostiene la comparacion
entre una CNN y una red densa: no basta con saber cual acierta mas, tambien importa
a que costo lo logra.

---

## Modelos a entrenar

### Redes convolucionales (ejercicio 4)

**CNN base.** Dos bloques de convolucion y pooling, seguidos de una capa densa.
Es la arquitectura minima que aprovecha la estructura espacial de la imagen y sirve
como punto de partida para saber cuanto rinde el enfoque sin ningun ajuste.

**CNN profunda con regularizacion.** Tres bloques convolucionales con normalizacion
por lotes y dropout antes de la capa de salida. La hipotesis es que mas profundidad
permite capturar los detalles finos que distinguen a M, N y S, que se diferencian
unicamente por la posicion del pulgar, mientras que la regularizacion contiene el
sobreajuste que trae el aumento de parametros.

Sobre la mejor de las dos se probaran variaciones de learning rate y batch size.

### Red neuronal simple (ejercicio 5)

Una red *fully-connected* que recibe la imagen aplanada, con una o dos capas ocultas.
Su proposito no es competir sino servir de linea base: al aplanar la imagen se destruye
la relacion espacial entre pixeles vecinos, de tal forma que la red pierde la nocion de
que dos pixeles contiguos forman parte de la misma forma. Comparar su desempeño contra
la CNN permite cuantificar cuanto aporta exactamente la convolucion en este problema,
en vez de asumirlo.

### Algoritmo clasico (ejercicio 6)

**Random Forest** sobre la imagen aplanada. Se eligio principalmente porque es el
algoritmo con el que mayor familiaridad tenemos como grupo, de tal forma que podemos
concentrar el tiempo en interpretar los resultados y no en aprender el metodo.

Adicional, considero que la forma en que funciona encaja bien con el problema: un
Random Forest entrena varios arboles y cada uno trabaja sobre una muestra distinta del
conjunto de datos, de tal forma que cada arbol termina fijandose en aspectos diferentes
de las imagenes y la decision final se toma combinando todos. Bajo esta idea, creo que
esto puede resultar util para poder aprender de mejor manera los diversos patrones que
hay que identificar en las señas, ya que las letras no se distinguen todas por lo mismo:
unas se diferencian por la posicion del pulgar, otras por cuantos dedos estan extendidos
y otras por la orientacion de la mano.

---

## Plan de transformaciones (ejercicio 7)

El enunciado pregunta por que un flip horizontal podria cambiar el significado de una
seña en vez de solo aumentar los datos. La respuesta corta es que una seña no es una
figura simetrica cualquiera: esta hecha con una mano especifica y su imagen espejo
corresponde a la otra mano.

Transformaciones que se van a aplicar:

| Transformacion | Rango | Por que si |
|---|---|---|
| Rotacion leve | +/- 10 grados | La mano nunca queda perfectamente alineada frente a la camara |
| Zoom | +/- 10% | Simula que la persona este mas cerca o mas lejos |
| Desplazamiento | +/- 10% | La mano no siempre queda centrada en el encuadre |
| Brillo y contraste | +/- 20% | Cubre condiciones de iluminacion distintas a las del estudio |

Transformaciones que se descartan:

| Transformacion | Por que no |
|---|---|
| Flip horizontal | Convierte la seña en su version con la otra mano, y el dataset completo esta hecho con una sola mano |
| Flip vertical | Produce una mano invertida, una pose que no ocurre en el uso real |
| Rotacion fuerte | Con giros grandes la orientacion deja de ser reconocible y algunas letras se acercan visualmente a otras |
