
<p align="center">
 <img src="https://user-images.githubusercontent.com/77422159/157056166-aa1ef8bd-fa1d-42c0-8846-860d0e81f54f.png">
  </p>

<h1 align="center"> Instituto Tecnológico de Tijuana </h1>
<h3 align="center"> Subdirección académica departamento de sistemas y computación.</h3>
<h4 align="center"> Datos Masivos</h4>

<h4 align="center"> JOSE CHRISTIAN ROMERO HERNANDEZ.</h4>


<h4 align="center"> Perez Mora Ana Ivonne 18212074</h4>
<h4 align="center"> Perez Ortega Victoria Valeria 18210718</h4>
<h4 align="center"> Lopez Pablo Israel 17210585</h4>
<h4 align="center"> Madrigal Ramos Ulises Omar 18210496</h4>
 



<h1 align="center"> Indice </h1>
<h1 align="center"> Linear Support Vector Machine </h1>
Support vector machine (SVM) es un algoritmo de aprendizaje supervisado que se utiliza en muchos problemas de clasificación y regresión, incluidas aplicaciones médicas de procesamiento de señales, procesamiento del lenguaje natural y reconocimiento de imágenes y voz.


<h1 align="center"> ¿Cómo funciona?</h1>
Las Máquinas de Vectores Soporte (creadas por Vladimir Vapnik) constituyen un método basado en aprendizaje para la resolución de problemas de clasificación y regresión. En ambos casos, esta resolución se basa en una primera fase de entrenamiento (donde se les informa con múltiples ejemplos ya resueltos, en forma de pares {problema, solución}) y una segunda fase de uso para la resolución de problemas. En ella, las SVM se convierten en una “caja negra” que proporciona una respuesta (salida) a un problema dado (entrada).
<h1 align="center"> Objetivo </h1>
El objetivo del algoritmo SVM es encontrar un hiperplano que separe de la mejor forma posible dos clases diferentes de puntos de datos. El algoritmo sólo puede encontrar este hiperplano en problemas que permiten separación lineal; en la mayoría de los problemas prácticos, el algoritmo maximiza el margen flexible permitiendo un pequeño número de clasificaciones erróneas.

<h1 align="center"> Kernel </h1>
La manera más simple de realizar la separación es mediante una línea recta, un plano recto o un hiperplano N-dimensional. Desafortunadamente los universos a estudiar no se suelen presentar en casos idílicos de dos dimensiones como en el ejemplo anterior, sino que un algoritmo SVM debe tratar con:
Más de dos variables predictoras.
Curvas no lineales de separación.
Casos donde los conjuntos de datos no pueden ser completamente separados.
Clasificaciones en más de dos categorías.

Debido a las limitaciones computacionales de las máquinas de aprendizaje lineal, éstas no pueden ser utilizadas en la mayoría de las aplicaciones del mundo real. La representación por medio de funciones Kernel ofrece una solución a este problema, proyectando la información a un espacio de características de mayor dimensión el cual aumenta la capacidad computacional de la máquinas de aprendizaje lineal. Es decir, mapeamos el espacio de entradas X a un nuevo espacio de características de mayor dimensionalidad (Hilbert).

<h1 align="center"> Tipos de kernel</h1>

1. Polinomial-homogénea.

2. Perceptron.

3. Función de base radial Gaussiana.

<h1 align="center"> Características </h1>
Las SVM son básicamente clasificadores para 2 clases.
Se puede cambiar la formulación del algoritmo QP para permitir clasificación multiclase. Más comúnmente, los datos son divididos “inteligentemente” en dos partes de diferentes formas y una SVM es entrenada para cada forma de división. La clasificación multiclase es hecha combinando la salida de todos los clasificadores.

<h1 align="center"> Ventajas </h1>

- El entrenamiento es relativamente fácil. 
No hay óptimo local, como en las redes neuronales.
- Se escalan relativamente bien para datos en espacios dimensionales altos. 
- El compromiso entre la complejidad del clasificador y el error puede ser controlado explícitamente.
- Datos no tradicionales como cadenas de caracteres y árboles pueden ser usados como entrada a la SVM, en vez de vectores de características.

<h1 align="center"> Desventajas </h1>

- Se necesita una “buena” función kernel, es decir, se necesitan metodologías eficientes para sintonizar los parámetros de inicialización de la SVM. 

<h1 align="center"> Aplicaciones </h1>

- Reconocimiento óptico de caracteres.
- Detección de caras para que las cámaras digitales enfoquen correctamente.
- Filtros de spam para correo electrónico.
- Reconocimiento de imágenes a bordo de satélites (saber qué partes de una imagen tienen nubes, tierra, agua, hielo, etc.)
 
 https://youtu.be/_YPScrckx28
  
  https://youtu.be/kl6tyEi5eso

<h1 align="center"> Ejemplo </h1>  

```scala
// Linear Support Vector Machine

// Import the "LinearSVC" library.
import org.apache.spark.ml.classification.LinearSVC

// Load training data
val training  = spark.read.format("libsvm").load("C:/Spark/spark-2.4.8-bin-hadoop2.7/data/mllib/sample_libsvm_data.txt")

//Set the maximum number of iterations and the regularization parameter 
val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)


// Fit the model
val lsvcModel = lsvc.fit(training)

// Print the coefficients and intercept for linear svc
println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")

```



<h1 align="center"> Referencias </h1>

[1] Machine Learning y Support Vector Machines: porque el tiempo es dinero | Blog. (s. f.). Merkle. https://www.merkleinc.com/es/es/blog/machine-learning-support-vector-machines
[2] Support Vector Machine (SVM). (s. f.). MATLAB & Simulink. https://la.mathworks.com/discovery/support-vector-machine.html
[3]Heras, J. M. (2019, May 28). Máquinas de Vectores de Soporte (SVM). IArtificial.net. https://www.iartificial.net/maquinas-de-vectores-de-soporte-svm/
[4]colaboradores de Wikipedia. (2022, May 2). Máquinas de vectores de soporte. Wikipedia, la enciclopedia libre. Retrieved May 9, 2022, from https://es.wikipedia.org/wiki/M%C3%A1quinas_de_vectores_de_soporte#Idea_b%C3%A1sica
[5]https://www.researchgate.net/publication/49588125_LAS_MAQUINAS_DE_SOPORTE_VECTORIAL_SVMs
