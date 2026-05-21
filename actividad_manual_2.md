Actividad 6: Matriz de atencion completa
Frase: LA NIÑA PEQUEÑA COME FRUTA

Asignacion de puntos (0-10) y normalizacion a porcentajes por fila:
(La logica: Cada palabra califica que tan dependiente es de las demas para entender su rol en la oracion).

1. Fila de "LA":
Puntajes: LA -> 3, NIÑA -> 9, PEQUEÑA -> 3, COME -> 0, FRUTA -> 0 = 15
Porcentajes: LA(20%), NIÑA(60%), PEQUEÑA(20%), COME(0%), FRUTA(0%)
*Razon: El articulo depende del sustantivo al que modifica.*
2. Fila de "NIÑA":
Puntajes: LA -> 3, NIÑA -> 4, PEQUEÑA -> 7, COME -> 9, FRUTA -> 0 = 23
Porcentajes: LA(13%), NIÑA(17%), PEQUEÑA(31%), COME(39%), FRUTA(0%)
*Razon: Se define por sus adjetivos y la accion que realiza.*
3. Fila de "PEQUEÑA":
Puntajes: LA -> 0, NIÑA -> 9, PEQUEÑA -> 5, COME -> 2, FRUTA -> 0 = 16
Porcentajes: LA(0%), NIÑA(56%), PEQUEÑA(31%), COME(13%), FRUTA(0%)
4. Fila de "COME":
Puntajes: LA -> 0, NIÑA -> 8, PEQUEÑA -> 2, COME -> 5, FRUTA -> 9 = 24
Porcentajes: LA(0%), NIÑA(33%), PEQUEÑA(8%), COME(21%), FRUTA(38%)
5. Fila de "FRUTA":
Puntajes: LA -> 0, NIÑA -> 3, PEQUEÑA -> 0, COME -> 9, FRUTA -> 4 = 16
Porcentajes: LA(0%), NIÑA(19%), PEQUEÑA(0%), COME(56%), FRUTA(25%)

Respuestas:

1. ¿La fila de COME se parece a la de FRUTA? ¿Por que deberian diferir?
No se parecen. Difieren porque buscan cosas distintas: "COME" busca a su sujeto (niña) y su objeto (fruta). "FRUTA" busca principalmente a la accion (come) que recae sobre ella para entender que esta siendo comida.
2. ¿Alguna fila reparte atencion casi pareja?
Si la frase tuviera una conjuncion (como "Y"), esa palabra repartiria atencion pareja a los dos elementos que une, actuando como un puente.
3. ¿Cuantas celdas tendria una tabla de 150 palabras?
Tendria 150 x 150 = 22,500 celdas. Esto explica el coste de memoria: si duplicas las palabras (300), la tabla no se duplica, se cuadruplica (90,000 celdas), consumiendo exponencialmente la memoria de la tarjeta grafica.

---

Actividad 7: Softmax a mano (de puntajes a probabilidades)

Datos: NIÑA (4.0), PEQUEÑA (0.5), COME (0.1), FRUTA (2.0)
Calculo con exponencial (e^x):
e^4.0 = 54.60
e^0.5 = 1.65
e^0.1 = 1.11
e^2.0 = 7.39
Suma total = 64.75

Porcentajes (Softmax):
NIÑA: (54.60 / 64.75) * 100 = 84%
PEQUEÑA: (1.65 / 64.75) * 100 = 3%
COME: (1.11 / 64.75) * 100 = 2%
FRUTA: (7.39 / 64.75) * 100 = 11%

Respuesta:
¿Por que no basta con dividir los puntajes 0-10 entre su suma?
Porque Softmax usa exponenciales para "exagerar" las diferencias. Si solo dividieramos, un puntaje de 4 vs 2 daria porcentajes de 67% y 33%. Al usar e^x, el 4 aplasta al 2 (84% vs 11%), lo que ayuda a la IA a tomar decisiones mas definitivas y claras, eliminando el "ruido".

---

Actividad 8: Mezcla de "vectores contenido" (Values)
Vectores base: LA(2,1), NIÑA(5,6), PEQUEÑA(2,3), COME(6,2), FRUTA(7,4)
Porcentajes (pesos) de COME: LA(0.02), NIÑA(0.40), PEQUEÑA(0.08), COME(0.15), FRUTA(0.35)

Calculo del nuevo vector de COME:
Multiplicamos cada vector por su porcentaje y sumamos las 'x' y las 'y':

Eje X: (2*0.02) + (5*0.40) + (2*0.08) + (6*0.15) + (7*0.35)
X = 0.04 + 2.00 + 0.16 + 0.90 + 2.45 = 5.55

Eje Y: (1*0.02) + (6*0.40) + (3*0.08) + (2*0.15) + (4*0.35)
Y = 0.02 + 2.40 + 0.24 + 0.30 + 1.40 = 4.36

Vector salida de COME = (5.55, 4.36)

Resumen: Al inicio, COME estaba en (6,2). Despues de atender a las demas palabras, su nuevo vector (5.55, 4.36) lo ha acercado "espacialmente" a NIÑA y FRUTA. El verbo "absorbio" el significado de sus vecinas.

---

Actividad 9: Mascara de padding
Frase 1: EL GATO COME [PAD] [PAD]

Cuadricula 5x5:
Palabra       | EL  GATO  COME  [PAD]  [PAD]
EL            |  v    v     v     x      x
GATO          |  v    v     v     x      x
COME          |  v    v     v     x      x
[PAD]         |  x    x     x     x      x
[PAD]         |  x    x     x     x      x

Respuestas:
¿Por que Frase 2 (sin PAD) no necesita celdas tachadas?
Porque todas sus posiciones contienen informacion real y util para el contexto.
¿Que pasaria si el modelo atendiera al PAD?
El modelo se confundiria intentando encontrar conexiones logicas con el "vacio", diluyendo la atencion de las palabras reales y aprendiendo patrones falsos.

---

Actividad 10: Atencion cruzada (Cross-attention)
Encoder (Español): YO QUIERO CAFE
Decoder (Ingles generado hasta ahora): I WANT ___

Puntajes para adivinar el hueco 3:
Hacia YO -> 2
Hacia QUIERO -> 2
Hacia CAFE -> 12

Porcentajes: YO(12.5%), QUIERO(12.5%), CAFE(75%)

Respuestas:

1. ¿CAFE deberia ganar? ¿Por que?
Si, porque el Decoder ya tradujo "YO" y "QUIERO". Para saber cual es la siguiente palabra, debe mirar directamente al objeto de la oracion original (CAFE).
2. ¿La fila de "I" podria mirar mucho a "YO"?
Si, tendria todo el sentido, ya que en el primer paso de traduccion, el Decoder necesita saber quien es el sujeto de la accion.

---

Actividad 11: Adivinar la palabra tapada (BERT / MLM)
Frase: EL GATO ___ PESCADO
Candidatos: COME, DUERME, VERDE, RAPIDO

Puntajes para el hueco (basado en compatibilidad con vecinos):
COME -> 9 (Encaja con un sujeto animal y un objeto comestible)
DUERME -> 2 (Encaja con el gato, pero "duerme pescado" no tiene sentido sintactico)
VERDE -> 1 (No encaja sintacticamente despues del sujeto)
RAPIDO -> 0 (Solo podria encajar como adverbio mal colocado)

Respuestas:

1. ¿Por que COME supera a VERDE?
Porque la IA detecta que falta una accion (verbo) entre el sujeto y el objeto.
2. ¿DUERME podria tener sentido?
Tendido atencion bidireccional, DUERME atiende bien al lado izquierdo (GATO), pero fracasa rotundamente al atender al lado derecho (PESCADO).
3. ¿Por que BERT necesita ver PESCADO?
Porque si solo viera "EL GATO", "DUERME" seria una respuesta perfecta. Ver el contexto derecho (PESCADO) es lo que lo obliga a buscar un verbo transitivo como "COME".

---

Actividad 12: Dos capas de atencion
Perfiles actuales (Capa 1): LA(15), NIÑA(48), PEQUEÑA(32), COME(55), FRUTA(70)
Mision: FRUTA vuelve a atender (Capa 2)

Puntajes (0-10) desde FRUTA hacia:
LA -> 1
NIÑA -> 7 (Ahora sabe que Niña esta conectada a Come)
PEQUEÑA -> 1
COME -> 9 (Sigue siendo su verbo directo)
FRUTA -> 4

Resumen:
"En la segunda capa, FRUTA ya 'sabe' indirectamente que fue comida por una niña, porque la primera capa contamino a COME con el significado de NIÑA, y ahora FRUTA se conecta con ese COME ya actualizado."

---

Actividad 13: RNN vs Transformer
Modo RNN (A -> B -> C -> D -> E)

* Enlaces de A a E: 4 saltos. (La informacion se degrada en cada salto).

Modo Atencion (Matriz todos miran a todos)

* Celdas/Enlaces: 5x5 = 25 conexiones directas. (A llega a E en 1 solo salto).

Respuestas:

1. Saltos en RNN = 4. Saltos en Atencion = 1.
2. Crecimiento en 120 palabras: RNN crece linealmente (120 enlaces secuenciales). Transformer crece cuadradamente (14,400 celdas).
3. Usamos Transformers porque la calidad no se pierde sin importar la distancia, y el procesamiento es paralelo (se calcula todo de golpe), mientras que la RNN es lenta porque espera a terminar una palabra para procesar la siguiente.

---

Actividad 14: Escalar por Raiz de d_k
Puntajes brutos: [10, 2, 2, 2] -> Suma: 16

Softmax sin escalar:
e^10 = 22026.5
e^2 = 7.4
(La primera palabra acapararia el 99.9% de la atencion, saturando la red y borrando al resto).

Puntajes escalados (Dividiendo entre 2): [5, 1, 1, 1]
Softmax escalado:
e^5 = 148.4
e^1 = 2.7
Suma e^x = 148.4 + 2.7 + 2.7 + 2.7 = 156.5
Porcentajes:
Palabra 1: 148.4 / 156.5 = 94.8%
Resto: 2.7 / 156.5 = 1.7% cada una.

Conclusion:
La palabra ganadora sigue ganando (94.8%), pero ahora las demas palabras (1.7%) aun conservan una fraccion de la atencion (sus gradientes). "Dividir entre la raiz es bajar el volumen para que la palabra dominante no apague por completo las voces menores".