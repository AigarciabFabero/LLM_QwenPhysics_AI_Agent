<h1 style="color:#1abc9c;">QwenPhysics: Fine-Tuning de QWEn2.5-3B con Artículos de Física de arXiv</h1>

  - [Objetivo de la aplicación](#objetivo-de-la-aplicación)
  - [Modelos empleados](#modelos-empleados)
  - [Tecnologías Utilizadas](#tecnicas-utilizadas)
  - [Instalación y Ejecución](#cómo-instalar-la-aplicación-y-las-dependencias)
  - [Cómo ejecutar la aplicación](#como-ejecutar-la-aplicación)
  - [Posibles mejoras Futuras](#posibles-mejoras-futuras)

## Objetivo de la aplicación

El objeto principal de este proyecto es desarrollar una aplicación para la generación de texto en un ámbito más exigente de lo convencional, como es la Física. Es sabido por los usuarios que los modelos generativos carecen en muchas ocasiones, de coherencia lógica-matemática y de conocimiento específico en áreas concretas. Por esto, dentro del hardware que disponemos y la limitación de recursos, se realiza *finnetuning* de un modelo Qwen pre-entrenado, con el fin de mejorar su rendimiento en tareas específicas relacionadas con la física.

Además, se pondrá a prueba en diferentes contextos de entrenamiento para evaluar el mismo y ver si es capaz de generar texto coherente. Revisar video adjunto para ver el funcionamiento de la aplicación.

## Modelos empleados

El modelo base pre-entrenado, sin tener en cuenta la cuantización, es el Qwen versión 2.5 de 3Billones de parámetros. Este modelo ha sido desarrollado por Alibaba Group, especializado en instrucciones y generación de código. Cabe destacar su capacidad para generar texto en multiples idiomas y sus capacidades de razonamiento lógico y matemático base. Además, es compatible con llama.cpp y herramientas de cuantización y ajuste fino (QLoRa,LoRa). Auqnue en este desarrollo no se le va a sacar partido, soporta *tool-calling* para invocar funciones externas (cálculo, busqueda web, APIs de datos) mediante plantillas de *chat templates* específicas.

## Tecnicas utilizadas

Para obtener el modelo final, es decir, nuetro modelo *finetunned*, se ha empleado la técnica de cuantización QLoRa, que permite reducir el tamaño del modelo y mejorar su rendimiento en tareas específicas. El entrenamiento se ha realizado empleando precisión float16 y finalmente se ha gusrdado el modelo cuantizado en formato GGUF con el fin de tener un modelo optimizado, que consume menos memoria a costa de una ligera pérdida de precisión y, por otro lado, se ha guardado una versión LoRa del modelo, que permite un ajuste fino más eficiente y rápido.

Podemos concluir así, que durante el proceso de desarrollo de esta aplicación, se ha empleado un modelo pre-entrenado de la familia Qwen y las diferentes formas de nuestro modelo *finetunned*.

## Cómo instalar la aplicación y las dependencias

Si se emplea Google Colab, no es necesario instalar nada, ya que el entorno ya cuenta con las librerías necesarias para ejecutar la aplicación. Este es el entorno que se ha empleado para el desarrollo de la aplicación. 

En caso de que se quiera ejecutar en local, es recomendable usar los entornos de Anaconda, Python o Miniconda, ya que permiten crear entornos virtuales y gestionar las dependencias de manera más eficiente.

Para conda:
```bash
# Crear un nuevo entorno conda
conda create -n qwenphysics python=3.10
conda activate qwenphysics

# Instalar PyTorch con soporte CUDA 
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia

pip install -r requirements.txt

pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

Instalación en un entorno virtual de Python:
```bash
# Crear y activar entorno
python -m venv qwenphysics_env
# En Windows:
qwenphysics_env\Scripts\activate
# En Linux/Mac:
source qwenphysics_env/bin/activate

pip install -r requirements.txt

pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

## Como ejecutar la aplicación

La ejecución de la aplicación se puede ver en el documento .ipynb adjunto. Principalmente debemos carcar nuestro modelo ya entrenado ya bien sea nuestro modelo LoRa, el modelo cuantizado o la versión en float16. 

```python
# Import necessary libraries
from langchain_community.chat_models import ChatLlamaCpp
import gradio as gr

# Create the LLM
# Although it's slower, we load the full model instead of just the LoRA adapters
# This approach simplifies deployment and provides better compatibility with llama.cpp
llm = ChatLlamaCpp(
    model_path="/content/drive/MyDrive/unsloth-F16.gguf",
    n_gpu_layers=25,
    stop=["<|im_end|>\n"],
    n_ctx=4096,
    max_tokens=4096,
    streaming=True,
    n_batch=256,
    temperature=0.1,
)
```

Una vez cargado nuestro modelo, lo lanzamos con un chatbot con ayuda de gradio. En este caso, se ha empleado un chatbot de tipo *chatbot* que permite la interacción con el modelo y la generación de texto en tiempo real. 

```python
# Create the llama handler
def llama_cpp(message, history):

    messages = history + [
        {
            "role": "user",
            "metadata": None,
            "content": message,
            "options": None,
        }
    ]

    with open('messages.txt', 'w') as f:
      f.write(str(messages))
    response = ""

    for c in llm.stream(messages):
        response += c.content
        yield response

# Create and launch the web chatbot
demo = gr.ChatInterface(
    llama_cpp,
    type="messages",
    flagging_mode="manual",
    flagging_options=[],
    save_history=True,
)

if __name__ == "__main__":
    demo.launch()
```

Una vez hemos lanzado el chatbot, podemos interactuar con el modelo y ver su rendimiento en tiempo real. En el caso de este proyecto, se ha empleado un conjunto de datos de artículos de física extraídos de arXiv, que se han utilizado para entrenar el modelo y mejorar su rendimiento en tareas específicas relacionadas con la física.

---------------------------------------------- Adjuntar imagenes de la aplicación 

**Algunas cuestiones para evaluar el modelo:**
- Riemann zeta function?  
- When was the michelson morley experiment?    
- Can you write the expresion for riemann zeta function?
- What is the term used to describe black holes?

**Pregunta extra por curiosidad de ver hasat que punto Qwuen tiene razonamiento lógico-matemático:**
- ES: ¿Cuánto vale una integral de una función impar definida en un intervalo simétrico centrada en el origen? EN: What is the value of an integral of an odd function defined on a symmetric interval centered at the origin? Respuesta/answer: 0 or null.

## Posibles mejoras futuras

- Emplear un modelo más potente, como el Qwen 7B o 13B.

- Emplear el conjunto de datos de arXiv en su totalidad, para mejorar el rendimiento del modelo.

- Entrenar otro modelo con un conjunto de datos del ámbito de la fisica y mergear ambos modelos para obtener un modelo más potente.