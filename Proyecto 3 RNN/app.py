import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Activation
from fastapi import FastAPI
from pydantic import BaseModel
import os

# --- 1. PREPROCESAMIENTO OPTIMIZADO ---
# Reducimos la memoria a 20 para que la RNN Vanilla no sufra amnesia
SEQ_LENGTH = 20 
# Usamos paso 1 para generar muchísimos más datos de entrenamiento
STEP = 1 

raw_text = open('dataset.c', 'r', encoding='utf-8').read()
# EL TRUCO MAESTRO: Le pegamos 20 espacios al inicio del texto de entrenamiento.
# Así el modelo aprende que los espacios vacíos significan "inicio de código".
text = (" " * SEQ_LENGTH) + raw_text 

chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

if not os.path.exists('rnn_model.keras'):
    print("Preparando datos con STEP=1 (Tomará un momento)...")
    sentences = []
    next_chars = []
    for i in range(0, len(text) - SEQ_LENGTH, STEP):
        sentences.append(text[i: i + SEQ_LENGTH])
        next_chars.append(text[i + SEQ_LENGTH])

    X = np.zeros((len(sentences), SEQ_LENGTH, len(chars)), dtype=bool)
    y = np.zeros((len(sentences), len(chars)), dtype=bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    # --- 2. MODELO VANILLA RNN ---
    print("Entrenando Vanilla RNN...")
    model = Sequential([
        SimpleRNN(128, input_shape=(SEQ_LENGTH, len(chars))),
        Dense(len(chars)),
        Activation('softmax')
    ])
    
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # Como tenemos muchos más datos (STEP=1), 60 épocas serán suficientes
    model.fit(X, y, batch_size=128, epochs=60) 
    model.save('rnn_model.keras')
else:
    print("Cargando modelo existente...")
    model = tf.keras.models.load_model('rnn_model.keras')

# --- 3. FUNCIONES DE PREDICCIÓN ---
def sample(preds, temperature=0.2):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-7) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(seed, chars_to_generate=15, temp=0.2):
    generated = ''
    # Ahora el rjust usa SEQ_LENGTH (20), y como le enseñamos espacios, no fallará
    seed = seed[-SEQ_LENGTH:].rjust(SEQ_LENGTH, ' ') 
    
    for _ in range(chars_to_generate):
        x_pred = np.zeros((1, SEQ_LENGTH, len(chars)))
        for t, char in enumerate(seed):
            if char in char_indices:
                x_pred[0, t, char_indices[char]] = 1.
        
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temp)
        next_char = indices_char[next_index]
        
        generated += next_char
        seed = seed[1:] + next_char 
    return generated

# --- 4. API CON FASTAPI ---
app = FastAPI()

class CodeContext(BaseModel):
    context: str

@app.post("/predict")
def predict_code(data: CodeContext):
    # Generamos 10 caracteres para que sea más exacto en autocompletar variables o tipos
    suggestion = generate_text(data.context, chars_to_generate=10, temp=0.2)
    return {"suggestion": suggestion}