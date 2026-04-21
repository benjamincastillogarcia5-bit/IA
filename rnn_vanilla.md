## El Termómetro Emocional (o cómo funciona una RNN básica)

Imagina que tus emociones funcionan exactamente igual que una Red Neuronal Recurrente (RNN) muy sencilla. En este modelo, tu estado de ánimo no solo depende de lo que te pasa hoy, sino también de la "resaca emocional" de ayer. 

La fórmula que define esto es:

$$
h_t = x_t + \alpha h_{t-1}
$$

¿Qué significa esto en la vida real?
* **h_t**: Es tu estado de ánimo total el día de hoy.
* **x_t**: Es el evento que te acaba de ocurrir hoy.
* **$\alpha$**: Es tu factor de memoria (qué tanto te aferras al pasado). Si tu memoria es de **0.5**, significa que mañana solo retendrás el **50%** de la emoción que sientes hoy.

Veamos cómo se comporta esto en tres situaciones de la vida diaria:

---

### Caso 1: El lunes espectacular (y por qué la alegría se esfuma)

Supongamos que tu memoria emocional retiene la mitad de las cosas (**0.5**). 

El lunes tienes un día increíble y te ocurre un evento de nivel **10**. Sin embargo, de martes a viernes la semana se vuelve aburrida y no pasa absolutamente nada nuevo (**0**). Así se vería tu ánimo a lo largo de la semana:

* **Día 1:** Estás a tope, en **10**.
* **Día 2:** Sin cosas nuevas, la emoción baja a la mitad: **5**.
* **Día 3:** Vuelve a bajar a la mitad: **2.5**.
* **Día 4:** Queda en **1.25**.
* **Día 5:** Terminas el viernes sintiendo apenas un **0.625**.

**La lección:** La emoción de un evento fuerte se desvanece muy rápido con los días si no lo alimentas. En IA, a esto se le conoce como el problema de *vanishing memory* (memoria que desaparece).

---

### Caso 2: El rescate emocional (remontando una mala racha)

Esta vez, empiezas la semana con el pie izquierdo. El Día 1 tienes un mal evento (**-6**) y el Día 2 te pasa otra cosa mala (**-4**). El Día 3 es neutral (**0**).

Tu "termómetro" va sumando el bajón:
* **Día 1:** Tu ánimo está en **-6**.
* **Día 2:** Al mal evento de hoy (**-4**) le sumas la mitad del dolor de ayer (**-3**). ¡Caes a **-7**!
* **Día 3:** Aunque hoy no pasa nada malo (**0**), sigues arrastrando la mitad de la tristeza de ayer. Tu ánimo está en **-3.5**.

Si quieres llegar al Día 4 sintiéndote positivo (es decir, que tu ánimo sea mayor a **0**), las matemáticas nos dicen lo siguiente:

$$
h_4 = x_4 + 0.5(-3.5) > 0
$$

Al resolverlo, descubrimos que el evento del Día 4 (**x_4**) tiene que ser obligatoriamente mayor a **1.75**.

**La lección:** Para salir de una mala racha, no basta con que dejen de pasar cosas malas. Necesitas que te ocurra un evento positivo lo suficientemente fuerte para vencer la inercia negativa que vienes arrastrando.

---

### Caso 3: El gran evento vs. Las pequeñas alegrías diarias

¿Qué te hace más feliz a largo plazo? Vamos a comparar dos formas de vivir la semana:

* **Escenario A (El pico de emoción):** Tienes un lunes increíble (**10**) y el resto de la semana es totalmente gris (**0**). Como vimos en el Caso 1, llegas al viernes con un ánimo casi apagado de **0.625**.
* **Escenario B (Constancia):** Todos los días te pasa algo pequeñito pero bueno, digamos un modesto **3**.
    * **Día 1:** Empiezas en **3**.
    * **Día 2:** Tu pequeño éxito de hoy (**3**) más la mitad del de ayer (**1.5**). Subes a **4.5**.
    * **Día 3:** Sigues sumando constancia. Llegas a **5.25**.
    * **Día 4:** Subes a **5.625**.
    * **Día 5:** Terminas el viernes con un muy sólido **5.8125**.

**El resultado:** El Escenario B gana por goleada (**5.8125** contra **0.625**).

---

### En resumen

Pensar en una red neuronal básica (Vanilla RNN) es como ver cómo procesamos los recuerdos a corto plazo:

1.  Tienen **memoria muy corta**.
2.  Las cosas del pasado pierden peso rapidísimo a medida que avanza el tiempo.
3.  El sistema siempre va a premiar la **información reciente** y, sobre todo, la **constancia**. Las pequeñas cosas positivas todos los días impactan más a largo plazo que un solo evento espectacular que queda en el pasado.