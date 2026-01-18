Este modulo utiliza la salida del módulo "combine" y recorre cada sección y tarea haciendo una selección de los keyframes que son relevantes para la documentación.

Por cada tarea, el módulo construye un prompt para un agente multimodal. El prompt incluye los segmentos de texto del topic, de la tarea, y las imágenes de los keyframes asociados a la tarea. Las imágenes se deben extraer de sqlite y copiar a un directorio temporal, que se borrará al finalizar la ejecución del módulo.

El prompt debe pedir al modelo de lenguage que selecccione de entre las imágenes incluidas en el prompt, solo una por cada acción que se deba documentar. La selección debe basarse en la relevancia de la imagen para la acción descrita en el texto. Ojo, porque una tarea puede implicar varias acciones, como hacer click en varios elementos en secuencia.

El resultado del modelo debe ser un json estructurado con la lista de imágenes seleccionadas para cada tarea, junto con un "caption" que asignar a cada imagen elegida. Se debe usar structured output en ollama, para asegurar que la salida del modelo es un json válido, igual que en el caso del módulo "topics".

El modelo por defecto a usar sera qwen3-vl:4b, con ollama, y una ventana de contexto de 64k tokens.
El prompt debe incluir instrucciones claras para que el modelo entienda que solo debe seleccionar una imagen por cada acción relevante, y que debe describir la acción en el caption.
Nota: qwen3-vl:2b suele devolver respuestas vacías con este prompt; qwen3-vl:4b es más fiable para esta tarea.

Para reducir errores de asociación entre imágenes y hashes, las imágenes se marcan con un watermark visible con el índice en una esquina (aprox. 28x28 px). El prompt solo muestra los índices y exige que el modelo devuelva únicamente esos índices. En caso de muchas imágenes, se filtran hashes consecutivos con distancia de Hamming baja para reducir redundancia.

Este es un ejemplo de uso de qwen3 extraído del blog de ollama:

```python

Using Qwen3-VL 235B

You can use Ollama’s cloud for free to get started with the full model using Ollama’s CLI, API, and JavaScript / Python libraries.
JavaScript Library

Install Ollama’s JavaScript library

npm i ollama 

Pull the model

ollama pull qwen3-vl:235b-cloud

Example non-streaming output with image

import ollama from 'ollama'

const response = await ollama.chat({
  model: 'qwen3-vl:235b-cloud',
  messages: [{ 
	  role: 'user', 
	  content: 'What is this?', 
	  images: ['./image.jpg']
	  }],
})
console.log(response.message.content)

Example streaming the output with image

import ollama from 'ollama'

const message = { 
	role: 'user', 
	content: 'What is this?', 
	images: ['./image.jpg'] 
	}
const response = await ollama.chat({
  model: 'qwen3-vl:235b-cloud',
  messages: [message],
  stream: true,
})
for await (const part of response) {
  process.stdout.write(part.message.content)
}

Ollama’s JavaScript library page on GitHub has more examples and API documentation.
Python Library

Install Ollama’s Python library

pip install ollama

Pull the model

ollama pull qwen3-vl:235b-cloud

Example non-streaming output with image

from ollama import chat
from ollama import ChatResponse

response: ChatResponse = chat(
	model='qwen3-vl:235b-cloud', 
	messages=[
  {
    'role': 'user',
    'content': 'What is this?',
    'images': ['./image.jpg']
  },
])
print(response['message']['content'])
# or access fields directly from the response object
print(response.message.content)

Example streaming the output with image

from ollama import chat

stream = chat(
    model='qwen3-vl:235b-cloud',
    messages=[{
    'role': 'user', 
    'content': 'What is this?',
    'images': ['./image.jpg']
    }],
    stream=True,
)

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)

Ollama’s Python library page on GitHub has more examples and API documentation.
```
