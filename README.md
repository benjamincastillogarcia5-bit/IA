Portafolio de Evidencias: Inteligencia Artificial
Este repositorio reúne los proyectos prácticos desarrollados a lo largo del curso, abarcando desde la implementación de algoritmos de aprendizaje de máquina tradicionales hasta el ajuste de Modelos de Lenguaje de Gran Escala (LLMs).

📑 Índice de Proyectos
Proyecto 1: Aprendizaje Basado en Reglas y Redes Neuronales en Videojuegos

Descripción: Desarrollo de un entorno interactivo utilizando Pygame donde se recolectan datos de juego para entrenar un Perceptrón Multicapa (MLP). El agente aprende a tomar decisiones en tiempo real basándose en las variables del entorno.

Componentes clave: Script de ejecución principal (juego_pygame_P.py).

Proyecto 2: Clasificación de Imágenes mediante Redes Neuronales Convolucionales (CNN)

Descripción: Diseño, entrenamiento y evaluación de una arquitectura CNN utilizando PyTorch para la clasificación de imágenes (reconocimiento de animales). Incluye el pipeline completo de preprocesamiento, aumentación de datos y análisis de curvas de aprendizaje.

Componentes clave: Notebook principal de experimentación (CNN_Animales_PyTorch_Final.ipynb) y script de preparación de datos (crear_dataset.py).

Proyecto 3: Predicción de Secuencias con Redes Neuronales Recurrentes (RNN)

Descripción: Implementación de un modelo enfocado en el procesamiento de secuencias y autocompletado de texto. Incluye el despliegue de una interfaz o API local para interactuar con las predicciones del modelo en tiempo real.

Componentes clave: Lógica del servidor/aplicación (app.py) y el módulo de autocompletado (rnn-autocompleter/).

Proyecto 4: Arquitectura RAG Local y Ajuste Fino (Fine-Tuning)

Descripción: Configuración de un sistema de Generación Aumentada por Recuperación (RAG) a nivel local utilizando un modelo cuantizado Llama-3.2-3B. El sistema está diseñado para contextualizar las respuestas del LLM a partir de un corpus de texto específico almacenado en una base de datos vectorial.

Componentes clave: Pipeline de recuperación y generación (rag_local.py) y archivo de configuración del modelo para Ollama (Modelfile).