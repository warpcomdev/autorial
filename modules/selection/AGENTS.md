Este modulo utiliza la salida del módulo "combine" y recorre cada sección y tarea haciendo una selección de los keyframes que son relevantes para la documentación.

Por cada tarea, el módulo construye un prompt para un agente multimodal. El prompt incluye los segmentos de texto del topic, de la tarea, y las imágenes de los keyframes asociados a la tarea. Las imágenes se deben extraer de sqlite y copiar a un directorio temporal, que se borrará al finalizar la ejecución del módulo.

El prompt debe pedir al modelo de lenguage que selecccione de entre las imágenes incluidas en el prompt, solo una por cada acción que se deba documentar. La selección debe basarse en la relevancia de la imagen para la acción descrita en el texto. Ojo, porque una tarea puede implicar varias acciones, como hacer click en varios elementos en secuencia.

El resultado del modelo debe ser un json estructurado con la lista de imágenes seleccionadas para cada tarea, junto con un "caption" que asignar a cada imagen elegida. Se debe usar structured output en ollama, para asegurar que la salida del modelo es un json válido, igual que en el caso del módulo "topics".

El modelo por defecto a usar sera ministral-3:3b, con ollama, y una ventana de contexto de 64k tokens.
El prompt debe incluir instrucciones claras para que el modelo entienda que solo debe seleccionar una imagen por cada acción relevante, y que debe describir la acción en el caption.
Nota: qwen3-vl:2b suele devolver respuestas vacías con este prompt; ministral-3:3b es más fiable y rápido para esta tarea.

Para reducir errores de asociación entre imágenes y hashes, las imágenes se marcan con un watermark visible con el índice en una esquina (aprox. 28x28 px). El prompt solo muestra los índices y exige que el modelo devuelva únicamente esos índices. En caso de muchas imágenes, se filtran hashes consecutivos con distancia de Hamming baja para reducir redundancia.
