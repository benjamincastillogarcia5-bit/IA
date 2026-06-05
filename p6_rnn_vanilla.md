#### 2.1 Actividad 2.1: Mapeo de Variables
La metáfora del poema se alinea con precisión a la ecuación fundamental de la red recurrente:

* **$x_t$**: "Soy la novedad pura, el pulso del instante, la matriz de características que el mundo me da en este segundo."
* **$h_{t-1}$**: "el fantasma del pasado, que trae consigo el resumen de todo lo que hemos vivido hasta ayer."
* **$W_{hx}, W_{hh}$**: "peajes inmutables, barreras que multiplican nuestra importancia y deciden qué tanto valemos."
* **$b$**: "un pequeño desvío inevitable"
* **$\tanh$**: "un muro curvo que nos comprime entre el -1 y el 1, evitando que nuestra energía explote hacia el infinito."
* **$h_t$**: "nazco yo, una nueva identidad. Soy tu estado actual, la respuesta de hoy, y estoy listo para ser el fantasma de tu mañana."

---

#### 2.2 Actividad 2.2: El Análisis de Dimensionalidad (El tamaño de los peajes)
Para que las operaciones de álgebra lineal sean computables, las dimensiones internas de las matrices deben coincidir. Sabiendo que $x_t \in \mathbb{R}^{20}$ (un vector de $20 \times 1$) y que el estado oculto requiere $h_t \in \mathbb{R}^{64}$ (un vector de $64 \times 1$):

1.  **Dimensiones de $W_{hx}$**: 
    Para proyectar el vector de entrada de $20$ características al espacio oculto de $64$ dimensiones, la matriz de pesos debe ser de **$64 \times 20$**. Al multiplicar una matriz de $64 \times 20$ por un vector de $20 \times 1$, el producto resultante es un vector de $64 \times 1$.
2.  **Dimensiones de $W_{hh}$**: 
    Para transformar el estado oculto anterior ($h_{t-1}$) y mantenerlo en el mismo espacio dimensional, la matriz recurrente debe ser cuadrada, con dimensiones de **$64 \times 64$**. El producto de $64 \times 64$ por $64 \times 1$ da como resultado $64 \times 1$.
3.  **Dimensión final de $h_t$**: 
    Como se estableció en el diseño del espacio oculto, y corroborado por la suma de los vectores resultantes más el sesgo (que también se transmite a lo largo de 64 dimensiones), el vector resultante es de **$\mathbb{R}^{64}$**.

---

#### 2.3 Actividad 2.3: La Estrofa Perdida (Pensamiento Lateral)
El sesgo ($b$) es un escalar o vector que permite desplazar la función de activación, garantizando que la red pueda modelar patrones incluso cuando las entradas son cero.

*Estrofa añadida:*
> "No dependo de lo que entra ni del eco de lo que fue; soy el ancla silenciosa que ajusta el umbral. Un paso independiente que mueve el centro del mundo, asegurando que, incluso en el vacío absoluto, siempre tengamos una voz."

---

#### 2.4 Actividad 2.4: El Límite del Muro Curvo (Análisis de Saturación)
El uso de la función tangente hiperbólica es crucial para el control del flujo de información, pero tiene debilidades inherentes en los extremos.

1.  **La Derivada**: La función $f(z) = \tanh(z)$ tiene forma de 'S'. Su derivada, $f'(z) = 1 - \tanh^2(z)$, tiene forma de campana: alcanza su máximo de $1$ cuando $z = 0$ y tiende a $0$ a medida que $z$ se aleja hacia el infinito positivo o negativo.
2.  **El Fenómeno en $z = 500$**: Si la combinación lineal arroja un valor extremo como $500$, la salida de la activación es casi idéntica a $1$. Al evaluar la derivada en este punto, el resultado es efectivamente **$0$** ($1 - 1^2 = 0$).
3.  **Consecuencia Catastrófica**: Este fenómeno se conoce como el **desvanecimiento del gradiente** (vanishing gradient). Cuando la derivada es cero, los gradientes de error que se propagan hacia atrás se anulan en la multiplicación. La red se vuelve "ciega" a sus propios errores en esas neuronas saturadas, los pesos no se actualizan en las iteraciones de entrenamiento y la red detiene su aprendizaje.

---

#### 2.5 Actividad 2.5: El Eco del Castigo (Trazo del Gradiente)
El proceso de retropropagación a través del tiempo (BPTT) es estrictamente una aplicación de la **Regla de la Cadena** del cálculo diferencial multivariable.

Para que la señal de castigo (el error) viaje desde el estado actual $h_t$ hasta el estado anterior $h_{t-1}$, debe recorrer el camino inverso de la propagación hacia adelante:
Primero, el error choca contra el "muro curvo", por lo que debe multiplicarse por la derivada de la función de activación ($1 - \tanh^2(z)$ evaluado en ese paso). Luego, la señal debe atravesar el "peaje inmutable" del pasado, lo que matemáticamente significa multiplicarse por la matriz transpuesta de los pesos recurrentes ($W_{hh}^T$). Solo así el gradiente llega al "fantasma del pasado" para ajustar cómo la red percibe el contexto previo.

---

#### 2.6 Actividad 2.6: Depuración del Oráculo (Inspección de Código NumPy)
Un error clásico en la implementación de transformaciones afines en Python que interrumpe cualquier flujo de procesamiento vectorial.

1.  **El Error Matemático**: 
    En NumPy, el operador `*` ejecuta una multiplicación de elementos uno a uno (producto de Hadamard). En el álgebra lineal de redes neuronales, necesitamos el **producto punto** (o multiplicación de matrices) para aplicar las proyecciones de $W_{hx}$ sobre $x_t$. Si se usa `*` con matrices de diferentes dimensiones que no puedan someterse a *broadcasting*, el programa lanzará un `ValueError`. Si casualmente las dimensiones permiten *broadcasting*, el cálculo se realizará sin fallar, pero el resultado matemático será completamente erróneo, destruyendo la integridad del modelo.

2.  **La Corrección del Código**:
    Se debe utilizar el operador de multiplicación matricial `@` (introducido en Python 3.5) o la función `np.dot()`.

```python
def paso_rnn_corregido(x_t, h_prev, W_hx, W_hh, b):
    # Uso del operador @ para multiplicación de matrices (producto punto)
    combinacion = (W_hx @ x_t) + (W_hh @ h_prev) + b
    return np.tanh(combinacion)
```