PotholeVision AI – Detección automática de baches con YOLOv8m

Este repositorio contiene el proyecto final de Procesamiento Digital de Imágenes (PDI), cuyo objetivo es desarrollar un sistema inteligente capaz de detectar baches en vías urbanas a partir de imágenes y videos capturados con una cámara vehicular (dashcam).

El sistema utiliza YOLOv8m sobre PyTorch (Transfer Learning + Fine-Tuning) para detectar baches, estima la severidad del daño en función del área del bounding box y ofrece una interfaz gráfica en Tkinter para que el usuario pueda cargar imágenes o videos y visualizar las detecciones de forma amigable.

El proyecto se enmarca en el contexto de Ciudades Inteligentes y movilidad urbana.

Antes de empezar, un video demo de la interfaz: https://www.youtube.com/watch?v=lD-4wQr6Nqo

El código completo de colab y jupyter se encuentra en el repositorio llamado: ProyectoBachesFinalPDI.ipynb

1. Descripción general

PotholeVision AI permite:

Detectar automáticamente baches en imágenes y videos.

Clasificar la severidad de cada bache (leve, moderado, severo) según el área aproximada de la detección.

Procesar videos tipo dashcam frame por frame y generar un video de salida con las detecciones superpuestas.

Mostrar los resultados mediante una interfaz gráfica (Tkinter) que permite:

Cargar imágenes y videos.

Ver el material original.

Ver el material con detecciones.

Ver contador de baches detectados y severidad.

El modelo fue entrenado utilizando una combinación de imágenes de un dataset base y frames extraídos de videos reales, anotados manualmente en formato YOLO.

2. Arquitectura del proyecto

A nivel general, el flujo del sistema es:

Entrada de datos: imagen o video (dashcam).

Preprocesamiento con OpenCV (lectura, redimensionamiento, extracción de frames en caso de video).

Inferencia con YOLOv8m entrenado (modelo en PyTorch).

Post-procesamiento:

Filtrado por confianza.

Cálculo del área del bounding box.

Clasificación de severidad (leve, moderado, severo).

Visualización y exportación:

Imágenes anotadas.

Videos procesados.

Conteo de baches y severidad en la interfaz Tkinter.

3. Requisitos del sistema

Sistema operativo: Windows 10/11 (probado) o equivalente con Python 3.10+.

GPU NVIDIA (opcional pero recomendado) para acelerar la inferencia:

Ejemplo: RTX 3070 Ti.

Python 3 instalado.

Pip actualizado.

Dependencias principales (ver también requirements.txt):

PyTorch + CUDA (si se usa GPU).

ultralytics (YOLOv8).

opencv-python.

matplotlib.

numpy.

tkinter (incluido en muchas instalaciones de Python).

pyinstaller (opcional, solo para generar el .exe).

4. Estructura del repositorio

La estructura puede variar ligeramente, pero en general se espera algo como:

notebooks/

Notebook(s) de entrenamiento, evaluación y pruebas (código documentado por bloques).

src/

Scripts Python principales del proyecto (por ejemplo, app de Tkinter, utilidades de inferencia).

runs_detect/

Carpeta generada por YOLO con los resultados de entrenamiento (best.pt, gráficos, etc.).

dataset_final_yolo/

data.yaml

images/ (train, val)

labels/ (train, val)

Video/

Videos de prueba (dashcam).

dist/ (opcional)

Ejecutable generado por PyInstaller.

requirements.txt

README.md (este documento).

5. Código documentado

El código del proyecto está documentado de dos formas:

Notebook(s) de entrenamiento y pruebas:

Organizados por “BLOQUES” numerados (BLOQUE 1, BLOQUE 2, etc.).

Cada bloque tiene comentarios que describen claramente qué se hace (preparación de entorno, carga de dataset, entrenamiento YOLOv8m, Fine-Tuning, métricas, pruebas en imágenes, pruebas en video).

Script principal de la aplicación (Tkinter / inferencia):

Comentarios en las funciones clave describiendo su propósito.

Docstrings cortos explicando el rol de funciones como:

carga de modelo,

procesamiento de imagen/video,

cálculo de severidad,

actualización de la interfaz.

Esta organización permite entender el flujo completo del sistema desde el dataset hasta la interfaz de usuario.

6. Instrucciones de instalación

Clonar el repositorio:

git clone https://github.com/Dabetho2/ProyectobachesPDI.git

Entrar a la carpeta del proyecto:

cd ProyectobachesPDI

(Opcional pero recomendado) Crear y activar un entorno virtual.

Instalar dependencias:

pip install -r requirements.txt

Instalar PyTorch con soporte CUDA según la GPU disponible (ver página oficial de PyTorch si es necesario).

7. Instrucciones de ejecución

Existen varias formas de ejecutar el proyecto según el componente que se quiera probar.

7.1. Ejecutar el notebook de entrenamiento / pruebas

Abrir el notebook principal (por ejemplo, en Jupyter Notebook o VS Code).

Ejecutar los bloques en orden:

Preparación del entorno.

Carga del dataset.

Entrenamiento YOLOv8m.

Fine-Tuning.

Visualización de métricas.

Pruebas con imágenes (prueba1, prueba5, etc.).

Pruebas con videos dashcam.

Esto permite reproducir el entrenamiento y ver paso a paso cómo se evaluó el modelo.

7.2. Ejecutar la aplicación Tkinter (interfaz gráfica)

Asegurarse de tener el modelo entrenado (best.pt) en la ruta esperada dentro del proyecto (por ejemplo, en runs_detect/potholes_finalRTX/weights/best.pt o similar).

Ejecutar el script principal de la app (por ejemplo):

python pothole_app.py

Se abrirá la interfaz gráfica con botones para:

Cargar imagen.

Cargar video.

Procesar entrada.

Ver resultados con detección de baches y severidad.

7.3. Ejecutar el archivo .exe (si está disponible)

Si en la carpeta dist del repositorio se incluye un ejecutable:

Navegar a la carpeta dist.

Ejecutar el archivo .exe de la app (doble clic).

Utilizar la interfaz de la misma forma que en la versión Python.

8. Datos y datasets utilizados

El proyecto utiliza un dataset personalizado que combina:

Imágenes de un dataset base de baches.

Frames extraídos de videos dashcam grabados en escenarios reales.

Anotaciones manuales en formato YOLO (archivos .txt).

Por temas de tamaño, los datos se referencian mediante el siguiente enlace:

Enlace a datasets y material de proyecto:
https://drive.google.com/drive/folders/1jYnNe7JN-7NyKCbKozbaDvItqRErBXTN?usp=sharing

Dentro de este enlace se encuentran:

Carpeta con las imágenes finales usadas para entrenamiento (dataset_final_yolo).

Carpeta con videos de prueba (dashcam).

Carpeta PDI/Proyectobaches con la estructura de trabajo utilizada durante el desarrollo.

9. Guía de usuario

A continuación se describe brevemente cómo usar la aplicación desde la perspectiva de un usuario final (sin conocimientos de programación).

Abrir la aplicación (ya sea con Python o con el .exe).

En la ventana de la interfaz, seleccionar una de las opciones:

“Cargar imagen”: abrir un archivo PNG/JPG donde se sospeche que hay baches.

“Cargar video”: abrir un archivo MP4 grabado con una cámara vehicular.

Hacer clic en el botón de “Procesar” o equivalente.

El sistema:

Carga el modelo entrenado.

Procesa la imagen o video.

Dibuja cajas alrededor de los baches detectados.

Calcula y muestra la severidad de cada detección (leve, moderado, severo).

En el caso de video:

Se procesa frame por frame.

Se puede generar un nuevo video de salida con las detecciones superpuestas.

Opcionalmente, se puede guardar la imagen o video anotado, junto con un resumen de baches encontrados.

10. Limitaciones conocidas

El dataset, aunque personalizado, sigue siendo relativamente pequeño; algunos tipos de baches muy específicos pueden no detectarse bien.

El sistema está optimizado para videos dashcam con un ángulo de cámara similar al usado durante el entrenamiento.

La severidad es una aproximación basada en el área del bounding box, no en una medición física del tamaño o profundidad real del bache.

En CPU, la velocidad de procesamiento de video es más limitada que en GPU.

11. Trabajos futuros

Integrar el sistema con GPS para asociar cada bache detectado a una coordenada geográfica.

Crear una plataforma colaborativa tipo “Waze de baches”, donde diferentes usuarios puedan reportar y confirmar la presencia de baches severos.

Extender el modelo a múltiples tipos de daños viales (grietas, hundimientos, parches, etc.).

Mejorar la estimación de severidad con información de profundidad u otros sensores.

12. Autor

Proyecto desarrollado como trabajo final de la asignatura Procesamiento Digital de Imágenes.

Autor: David Bernat Thorp
Correo: carlos.bernat@uao.edu.co