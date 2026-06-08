# Clasificador de Emociones de Perros

Proyecto de clasificación de emociones en perros utilizando **Deep Learning**, **Convolutional Neural Networks (CNN)** y **Transfer Learning**.  
El modelo busca reconocer automáticamente la emoción de un perro a partir de una imagen.

Las clases utilizadas son:

```text
angry
happy
relaxed
sad
```

---

## Descripción del proyecto

Este proyecto tiene como objetivo construir un clasificador de emociones de perros usando imágenes.  
Para ello se utiliza un dataset de Kaggle con imágenes separadas en cuatro carpetas, una por cada emoción.

Dataset utilizado:

[Dog Emotion Dataset - Kaggle](https://www.kaggle.com/datasets/danielshanbalico/dog-emotion/data)

La estructura original del dataset es:

```text
Data/
├── angry/
│   ├── dog_angry_1.jpg
│   └── ...
├── happy/
│   ├── dog_happy_1.jpg
│   └── ...
├── relaxed/
│   ├── dog_relaxed_1.jpg
│   └── ...
└── sad/
    ├── dog_sad_1.jpg
    └── ...
```

Cada clase contiene aproximadamente **1000 imágenes**, por lo que el dataset está balanceado. Esto es útil porque evita que el modelo aprenda más de una clase que de otra.

Paper usado como referencia [CLASSIFICATION OF DOG EMOTIONS USING CONVOLUTIONAL NEURAL NETWORK METHOD](https://drive.google.com/file/d/1xbXEbpDo8rO1zZc3t9-56L_ABptaHiDW/view?usp=sharing)

Carpeta del Google Drive: [Dog Emotions](https://drive.google.com/drive/folders/1AYk4SK1HflfkWiDXzO6UvaRKisGqGZSH)

## Objetivo

El objetivo principal es entrenar un modelo capaz de clasificar imágenes de perros en una de las siguientes emociones:

| Clase     | Significado    |
| --------- | -------------- |
| `angry`   | Perro enojado  |
| `happy`   | Perro feliz    |
| `relaxed` | Perro relajado |
| `sad`     | Perro triste   |

## Flujo general del proyecto

El proyecto sigue este proceso:

```text
1. Cargar el dataset desde Google Drive
2. Dividir imágenes en train, validation y test
3. Aplicar preprocesamiento y data augmentation
4. Crear generadores de imágenes
5. Construir un primer modelo con VGG16
6. Entrenar y validar el primer modelo
7. Construir un segundo modelo con EfficientNetV2B0
8. Entrenar con callbacks
9. Aplicar fine-tuning
10. Evaluar el modelo con test
11. Analizar métricas y matriz de confusión
```

---

# Separación del dataset

El dataset original solo viene separado por clases, pero no por entrenamiento, validación y prueba.  
Por eso se crea una nueva carpeta llamada `Split`.

El flujo utilizado hace lo siguiente:

- Toma las imágenes de cada clase.
- Las revuelve de forma aleatoria.
- Separa el dataset en tres partes:
  - 70% para entrenamiento.
  - 15% para validación.
  - 15% para prueba.
- Crea automáticamente las carpetas necesarias.
- Copia las imágenes a su nueva ubicación.

Ejemplo con 1000 imágenes por clase:

```text
700 imágenes para train
150 imágenes para validation
150 imágenes para test
```

De esta forma confirmamos que este bien balanceado los datos

Después de procesar el dataset, la estructura esperada queda así:

```text
Dogs Emotions/
├── Data/
│   ├── angry/
│   ├── happy/
│   ├── relaxed/
│   └── sad/
│
├── Split/
│   ├── train/
│   │   ├── angry/
│   │   ├── happy/
│   │   ├── relaxed/
│   │   └── sad/
│   │
│   ├── validation/
│   │   ├── angry/
│   │   ├── happy/
│   │   ├── relaxed/
│   │   └── sad/
│   │
│   └── test/
│       ├── angry/
│       ├── happy/
│       ├── relaxed/
│       └── sad/
│
├── best_dog_emotion_model.keras
└── README.md
```

---

# Preparación de datos

## Data augmentation

Para el primer modelo se utiliza `ImageDataGenerator` con normalización y aumento de datos.

El `data augmentation` ayuda a que el modelo no memorice exactamente las imágenes de entrenamiento.

En lugar de aprender una imagen específica, el modelo aprende patrones más generales. Esto es útil porque en la vida real una imagen puede tener diferentes condiciones de luz, ángulo, distancia o posición.

En este proyecto, el data augmentation es especialmente importante porque las emociones de perros pueden variar visualmente dependiendo de:

- La postura del perro.
- La raza.
- La iluminación.
- El encuadre.
- La orientación del rostro.
- El fondo de la imagen.

La normalización 1./255 lo que hace es que convierte los pixeles de rango `0-255` a rango `0-1`, esto para que al modelo sea mucho más fácil procesarlo

Esto se genera solamente al conjunto de entrenamiento.

Pues la validación y el test deben mantenerse lo más cercanos posible a imágenes reales, porque su propósito es medir el desempeño del modelo en condiciones no alteradas.

Por eso:

```text
Train      → sí lleva data augmentation
Validation → solo preprocesamiento básico
Test       → solo preprocesamiento básico
```

---

## Generadores de imágenes

Después se crean los generadores, estos se carga las imágenes directamente desde carpetas `flow_from_directory` donde detecta automáticamente las clases según el nombre de las carpetas:

```text
angry
happy
relaxed
sad
```

Parámetros importantes:

| Parámetro                  | Función                                     |
| -------------------------- | ------------------------------------------- |
| `target_size=(150, 150)`   | Redimensiona todas las imágenes a 150x150   |
| `batch_size=32`            | Envía imágenes al modelo en grupos de 32    |
| `class_mode="categorical"` | Usa etiquetas para clasificación multiclase |

Ya con esa configuración y con la verificación que los datos estén bien entrenados ahora pasamos a la parte más importante que son los modelo

---

# Primer modelo: VGG16

## Arquitectura

El primer enfoque del proyecto utiliza **VGG16**, una red neuronal convolucional preentrenada.

VGG16 ya fue entrenada previamente con una gran cantidad de imágenes, por lo que ya aprendió a detectar características visuales generales como bordes, formas, texturas y patrones.

En este proyecto se usa como base para extraer información visual de las imágenes de perros.

Esto se hace para poder aprovechar un modelo que ya aprendió de otro conjunto de datos y adaptarlo a un nuevo problema.

En lugar de entrenar una red desde cero, se usa un modelo preentrenado como punto de partida. por eso en este proyecto:

```text
Modelo preentrenado → aprende características generales de imágenes
Clasificador nuevo  → aprende las emociones específicas de perros
```

Esto ayuda a entrenar más rápido y con mejores resultados, especialmente cuando no se tiene un dataset extremadamente grande.

La arquitectura del primer modelo puede entenderse así:

```text
Imagen de entrada
↓
VGG16 preentrenada
↓
Conversión de características a vector
↓
Capa densa intermedia
↓
Capa final de 4 clases
↓
Predicción de emoción
```

La base VGG16 funciona como extractor de características. Después se agregan capas nuevas para adaptar el modelo a las cuatro emociones del proyecto.

La última capa tiene 4 neuronas porque existen 4 clases y se usa `softmax` porque devuelve una probabilidad para cada clase.

Ejemplo:

```text
angry:   0.10
happy:   0.70
relaxed: 0.15
sad:     0.05
```

En este caso, el modelo predice `happy`.

Ahora el modelo se entrena durante 10 épocas.

---

## Evaluación del entrenamiento

Interpretación general del primer modelo:

- El modelo sí aprende.
- La precisión de entrenamiento y validación suben progresivamente.
- La pérdida disminuye.
- No se observa sobreajuste fuerte.
- La calidad es aceptable, pero todavía mejorable.

Resultado aproximado observado:

```text
Accuracy de validación cercana a 60%
Accuracy de test cercana a 63%
```

Las curvas de precisión y pérdida muestran un entrenamiento estable, con mejoras progresivas y diferencias reducidas entre los resultados de entrenamiento y validación. Esto indica que no existe evidencia significativa de overfitting. Sin embargo, al alcanzar una precisión cercana al 60 % en ambos conjuntos, el modelo presenta un ligero underfitting. Por esta razón, se desarrollará un nuevo modelo con ajustes en su arquitectura e hiperparámetros para reducir el subajuste y mejorar su rendimiento.

---

# Segundo modelo: EfficientNetV2B0

Después del primer modelo, se construye un segundo modelo usando **EfficientNetV2B0**.

EfficientNetV2B0 es una arquitectura más moderna y eficiente que busca obtener buen rendimiento con menos parámetros y mejor uso computacional.

Este segundo modelo también usa transfer learning.

## Arquitectura general del segundo modelo

La arquitectura del segundo modelo es:

```text
Imagen 224 x 224
↓
EfficientNetV2B0 preentrenada
↓
Global Average Pooling
↓
Capa densa de 256 neuronas
↓
Dropout
↓
Capa final Softmax de 4 clases
```

# Función de cada parte

| Componente             | Función                                                       |
| ---------------------- | ------------------------------------------------------------- |
| EfficientNetV2B0       | Extrae características visuales importantes de la imagen.     |
| Global Average Pooling | Resume la información visual en un vector compacto.           |
| Capa densa             | Aprende relaciones entre las características y las emociones. |
| Dropout                | Reduce el riesgo de memorización excesiva.                    |
| Softmax                | Produce la probabilidad de cada una de las 4 emociones.       |

## Ventaja de Global Average Pooling

En lugar de aplanar toda la información visual con una capa muy grande, se resume cada mapa de características mediante un promedio.

Esto reduce la cantidad de parámetros y hace que el modelo sea más ligero.

También puede ayudar a reducir el sobreajuste, porque evita que el clasificador final sea demasiado pesado.

## Función de Dropout

Dropout apaga aleatoriamente una parte de las neuronas durante el entrenamiento.

Esto obliga al modelo a no depender siempre de las mismas conexiones internas y mejora su capacidad de generalización.

En palabras simples:

> Dropout ayuda a que el modelo no memorice demasiado las imágenes de entrenamiento.

# Callbacks de entrenamiento

Para mejorar el proceso de entrenamiento se utilizan callbacks. Estos son mecanismos que supervisan el entrenamiento y toman decisiones automáticamente.

Los callbacks usados son:

| Callback          | Función                                                |
| ----------------- | ------------------------------------------------------ |
| EarlyStopping     | Detiene el entrenamiento si el modelo deja de mejorar. |
| ModelCheckpoint   | Guarda automáticamente la mejor versión del modelo.    |
| ReduceLROnPlateau | Reduce la tasa de aprendizaje si el modelo se estanca. |

# Fine-tuning

Después del entrenamiento inicial, se aplica ajuste fino.

```python
conv_base.trainable = True

for layer in conv_base.layers[:-30]:
    layer.trainable = False
```

Esto significa:

- Se permite entrenar la base EfficientNet.
- Pero se mantienen congeladas casi todas sus capas.
- Solo se entrenan las últimas 30 capas.

Las capas iniciales aprenden patrones generales como bordes y texturas.  
Las últimas capas aprenden patrones más específicos, como rasgos faciales y características de animales.

Luego se recompila el modelo con una tasa de aprendizaje muy baja:

```python
model_2.compile(
    loss="categorical_crossentropy",
    optimizer=optimizers.RMSprop(learning_rate=1e-6),
    metrics=["accuracy"]
)
```

Finalmente se entrena de nuevo:

```python
history_fine = model_2.fit(
    train_generator_2,
    epochs=15,
    validation_data=val_generator_2,
    callbacks=callbacks
)
```

El `fine-tuning` ayuda a que EfficientNet se adapte mejor al problema específico de emociones de perros.

---

# Evaluación final

## Generador de test

Antes de evaluar, se crea el generador de prueba:

```python
test_generator = test_datagen_2.flow_from_directory(
    "Split/test",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)
```

## Classification report y matriz de confusión

Este bloque calcula:

| Métrica     | Significado                                                 |
| ----------- | ----------------------------------------------------------- |
| `precision` | Qué tan correctas son las predicciones de una clase         |
| `recall`    | Cuántas imágenes reales de una clase encontró correctamente |
| `f1-score`  | Balance entre precision y recall                            |
| `support`   | Cantidad de imágenes reales por clase                       |
| `accuracy`  | Porcentaje total de imágenes clasificadas correctamente     |

El modelo obtuvo un desempeño aproximado de:

```text
Accuracy: 0.63
Macro F1-score: 0.63
Weighted F1-score: 0.63
```

Reporte de clasificación:

```text
              precision    recall  f1-score   support

       angry       0.67      0.64      0.66       150
       happy       0.61      0.77      0.68       150
     relaxed       0.62      0.62      0.62       150
         sad       0.83      0.65      0.73       150

    accuracy                           0.67       600
   macro avg       0.68      0.67      0.67       600
weighted avg       0.68      0.67      0.67       600
```

Matriz de confusión:

```text
[[ 96  30  16   8]
 [ 14 115  18   3]
 [ 13  35  93   9]
 [ 20   9  23  98]]
```

---

# Interpretación de resultados

Para la evaluación inicial del modelo se utilizaron métricas estándar de clasificación multiclase, debido a que el objetivo del proyecto es clasificar imágenes de perros en cuatro emociones: angry, happy, relaxed y sad. Las métricas seleccionadas fueron accuracy, precision, recall, F1-score y matriz de confusión.

La selección de estas métricas se basa en el artículo Classification of Dog Emotions Using Convolutional Neural Network Method, donde los autores evalúan modelos de clasificación de emociones de perros mediante métricas como accuracy, precision, recall, F1-score y ROC-AUC.

Con eso podemos decir que el modelo logró una accuracy de aproximadamente **67%** en test.

Como hay cuatro clases, un modelo que adivinara al azar tendría un desempeño cercano a:

Por lo tanto, el resultado de 67% muestra que el modelo sí aprendió patrones útiles.

## Mejor clase

La clase con mejor desempeño fue:

```text
sad
```

con:

```text
precision: 0.83
recall: 0.65
f1-score: 0.73
```

Esto significa que el modelo reconoce relativamente bien imágenes de perros tristes.

## Clase con mayor recall

La clase `happy` tuvo el recall más alto:

```text
recall: 0.77
```

Esto significa que el modelo encontró correctamente la mayoría de las imágenes reales de perros felices.

Sin embargo, su precision fue menor:

```text
precision: 0.61
```

Esto indica que el modelo predijo muchas imágenes como `happy`, incluso cuando pertenecían a otra clase.

## Principales confusiones

La matriz de confusión muestra que el modelo confunde principalmente:

```text
angry → happy
relaxed → happy
```

Esto sugiere que algunas expresiones visuales de perros enojados o relajados pueden parecerse a perros felices, especialmente si hay diferencias de iluminación, posición o expresión facial.

---

# Comparación con el documento de referencia

El proyecto se apoyó en un documento científico sobre clasificación de emociones de perros mediante CNN.

Resultados reportados en el documento:

| Modelo   | Accuracy |
| -------- | -------: |
| CNN      |   74.75% |
| VGG16    |   68.67% |
| ResNet50 |   65.10% |

Resultado obtenido en este proyecto:

| Modelo                         | Accuracy |
| ------------------------------ | -------: |
| EfficientNetV2B0 + Fine-tuning |      67% |

Conclusión:

- El modelo supera ampliamente el azar.
- Todavía no supera los resultados principales del documento de referencia.
- Está cerca del rendimiento reportado para VGG16.

---

# Conclusión

El proyecto logró construir un clasificador funcional de emociones de perros usando redes neuronales convolucionales y Transfer Learning.

El primer modelo con VGG16 permitió establecer una base inicial.  
Posteriormente, el segundo modelo con EfficientNetV2B0, callbacks y fine-tuning permitió implementar una arquitectura más robusta y controlada.

El modelo final obtuvo un desempeño aproximado de **67% de accuracy** en el conjunto de prueba, superando claramente el azar y mostrando capacidad para identificar patrones visuales relacionados con emociones caninas.

Sin embargo, todavía existe margen de mejora, especialmente en las confusiones entre:

```text
angry y happy
relaxed y happy
```

Esto indica que el modelo puede beneficiarse de más limpieza de datos, mejor ajuste del augmentation, más fine-tuning y comparación con otras arquitecturas.
