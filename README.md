# LLM Email Guard: Detección de Phishing con BERT e IA Explicable

## Descripción General

Este proyecto, denominado **LLM Email Guard**, aborda el problema persistente de los correos electrónicos de phishing desarrollando un sistema inteligente. Su objetivo principal es detectar correos electrónicos maliciosos con alta precisión y, de manera crucial, proporcionar explicaciones claras y comprensibles para sus clasificaciones. Esto busca aumentar la confianza del usuario y su comprensión, yendo más allá de los filtros de spam tradicionales que a menudo son opacos. El sistema integra modelos avanzados de Procesamiento del Lenguaje Natural (PLN) con Modelos de Lenguaje Grandes (LLMs) y presenta sus funcionalidades a través de una aplicación web intuitiva.

## Tecnologías Principales

El proyecto aprovecha un conjunto de potentes librerías de Python y modelos de IA:

*   **BERT (Representaciones de Codificador Bidireccional de Transformers):** Se utiliza específicamente el modelo `distilbert-base-uncased` por su eficiencia y sólido rendimiento en la clasificación de secuencias. BERT se ajusta finamente para clasificar correos electrónicos como legítimos o de phishing.
*   **Modelos de Lenguaje Grandes (LLMs):** Se emplean LLMs, como IBM Granite o ChatGPT (el notebook hace referencia principal a Granite/ChatGPT y `langchain-ollama`), para proporcionar explicaciones contextuales de las clasificaciones de phishing. Este componente de "IA explicable" ayuda a los usuarios a entender *por qué* un correo electrónico se considera sospechoso.
*   **Hugging Face Transformers:** Esta librería facilita el acceso a modelos Transformer pre-entrenados (como BERT) y herramientas para su tokenización y ajuste fino.
*   **Hugging Face Datasets:** Utilizada para la carga, procesamiento y gestión eficiente de grandes conjuntos de datos.
*   **PyTorch (`torch`):** El framework de aprendizaje profundo subyacente utilizado para construir y entrenar los modelos.
*   **Accelerate:** Una librería de Hugging Face diseñada para simplificar y optimizar los procesos de entrenamiento en diversas configuraciones de hardware.
*   **`ibm_watsonx_ai`:** SDK de Python para interactuar con los servicios de IBM Watsonx.ai, utilizado probablemente para acceder a modelos fundacionales como Granite.
*   **Gradio (`gradio`):** Una librería para construir rápidamente interfaces web personalizables para modelos de machine learning, lo que permite una fácil demostración e interacción con el sistema desplegado.
*   **LangChain (`langchain`, `langchain-core`, `langchain-ollama`):** Frameworks para desarrollar aplicaciones impulsadas por modelos de lenguaje, utilizados aquí para integrar LLMs en la generación de explicaciones.
*   **Matplotlib (`matplotlib`):** Para la visualización de datos, particularmente en la evaluación del rendimiento del modelo (por ejemplo, matrices de confusión).

## Conjunto de Datos

El proyecto utiliza el conjunto de datos **`PhishingEmailDetectionv2.0`**, obtenido de Hugging Face (`cybersectony/PhishingEmailDetectionv2.0`).

*   **Contenido:** El conjunto de datos contiene tanto contenido de correos electrónicos como URLs, categorizados en cuatro clases:
    *   `0`: Correo Electrónico Legítimo
    *   `1`: Correo Electrónico de Phishing
    *   `2`: URL Legítima
    *   `3`: URL de Phishing
*   **Estadísticas (Conjunto de Datos Original):**
    *   Muestras Totales: 22,644 correos electrónicos, 177,356 URLs.
    *   Distribución de Clases: Aproximadamente equilibrada entre las categorías legítimas/phishing para correos electrónicos y URLs.
*   **Pasos de Preprocesamiento:**
    1.  **Extracción de Datos de Correo Electrónico:** Para este proyecto, solo se extraen las muestras relacionadas con correos electrónicos (etiquetas 0 y 1).
    2.  **Manejo de Datos Faltantes:** Se identifican y eliminan las muestras con contenido "vacío" (`empty`).
    3.  **Eliminación de Duplicados:** Se eliminan las entradas de correo electrónico duplicadas basadas en su contenido para asegurar muestras de entrenamiento únicas.
    4.  **División Entrenamiento-Prueba:** El conjunto de datos de correo electrónico limpio se baraja (`seed=0` para reproducibilidad) y se divide en conjuntos de entrenamiento y prueba, con un 80% para entrenamiento y un 20% para prueba.
    5.  **Creación de "Toy Dataset":** Debido a posibles restricciones de recursos computacionales (falta de GPUs dedicadas), se construye un pequeño "toy dataset" (conjunto de datos de juguete) para demostración. Este consiste en un número muy limitado de correos electrónicos legítimos y de phishing para entrenamiento (por ejemplo, 3 de cada) y una única muestra de prueba, lo que permite mostrar el proceso de entrenamiento incluso en entornos con recursos limitados.

## Arquitectura del Modelo y Entrenamiento (Ajuste Fino de BERT)

*   **Selección del Modelo Base:** Se elige `distilbert-base-uncased` como el modelo pre-entrenado principal. Esta elección se basa en que DistilBERT es una versión "destilada" de BERT, ofreciendo un tamaño significativamente reducido y una inferencia más rápida (aproximadamente un 40% más pequeño y un 60% más rápido) mientras conserva alrededor del 97% del rendimiento de BERT. La versión "uncased" ignora la sensibilidad a mayúsculas y minúsculas, lo cual es beneficioso para la detección de phishing, ya que los atacantes a menudo varían el uso de mayúsculas para evadir filtros de palabras clave simples.
*   **Tokenizador:** Se carga un `AutoTokenizer` correspondiente a `distilbert-base-uncased`. Convierte el texto sin procesar en tokens numéricos (subpalabras o caracteres) que el modelo BERT puede procesar.
    *   **Relleno (Padding):** Asegura que todas las secuencias de entrada tengan una longitud uniforme añadiendo tokens especiales `[PAD]`.
    *   **Truncamiento (Truncation):** Recorta las secuencias más largas que la longitud máxima de entrada del modelo.
    *   **Máscara de Atención (`attention_mask`):** Generada para informar al modelo qué tokens son contenido real (`1`) y cuáles son relleno (`0`), de modo que los tokens de relleno se ignoren durante los cálculos de atención.
*   **Modelo de Clasificación de Secuencias:** Se carga `AutoModelForSequenceClassification` con `num_labels=2` (para clasificación binaria: legítimo o phishing). Esto inicializa el DistilBERT pre-entrenado con una nueva cabeza de clasificación (un MLP con una capa de salida softmax) lista para el ajuste fino en la tarea específica.

## IA Explicable con LLMs

Una característica clave de LLM Email Guard es su capacidad para proporcionar explicaciones.

*   **Rol de los LLMs:** A diferencia de los clasificadores tradicionales que solo emiten una predicción binaria, los LLMs pueden analizar los correos electrónicos de phishing detectados y generar justificaciones comprensibles para el usuario sobre la clasificación.
*   **Beneficios:**
    *   **Extracción de Palabras Clave Sospechosas:** Los LLMs pueden identificar y resaltar frases o palabras comúnmente asociadas con intentos de phishing (por ejemplo, "verifique su cuenta", "urgente", "haga clic aquí").
    *   **Transparencia y Confianza del Usuario:** Al explicar el razonamiento (por ejemplo, "El dominio del remitente es inusual y el mensaje crea urgencia"), el sistema genera confianza en el usuario y promueve la concienciación sobre ciberseguridad.
    *   **Detección Adaptativa:** Los LLMs, debido a sus vastos datos de entrenamiento, pueden adaptarse a tácticas de phishing nuevas y en evolución de manera más efectiva que los sistemas basados en reglas fijas, actuando como una capa adicional de seguridad.

## Aplicación Web (Gradio)

El proyecto incluye una aplicación web construida con Gradio, diseñada para:

*   Proporcionar una interfaz de usuario amigable para interactuar con el sistema de detección de phishing.
*   Permitir a los usuarios introducir el contenido del correo electrónico para su análisis.
*   Mostrar el resultado de la clasificación (phishing/legítimo).
*   Presentar la explicación generada por el LLM para la clasificación.

## Configuración e Instalación

Las siguientes librerías de Python son necesarias. Se pueden instalar usando `pip`:

```bash
pip install torch==2.8.0
pip install transformers==4.55.4
pip install datasets==3.6.0
pip install accelerate==1.10.1
pip install ibm-watsonx-ai==1.4.7
pip install gradio==5.45.0
pip install matplotlib==3.10.6
pip install langchain langchain-core langchain-ollama
```

## Archivos Auxiliares

El directorio del proyecto también contiene `PY0101EN-1-1-Write_your_first_python_code (1).ipynb`. Este notebook de Jupyter es un tutorial introductorio básico sobre programación en Python, que cubre conceptos fundamentales como "Hello World", comentarios, tipos de datos (enteros, flotantes, cadenas, booleanos), conversión de tipos, expresiones y variables. Sirve como un recurso de aprendizaje general para principiantes en Python y no está directamente integrado en la aplicación LLM Email Guard.
