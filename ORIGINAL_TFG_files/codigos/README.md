README - TFG
=========================

Título: Adaptación de una política robótica multimodal para control por lenguaje natural y visión en un UR5e con cámara IDS  
Autor: Adrián Losada Álvarez  
Convocatoria: Junio/Julio 2025


Resumen del proyecto
---------------------
Este Trabajo de Fin de Grado evalúa la integración de la política multimodal Octo para tareas de manipulación en el brazo
robótico UR5e, asistido por una cámara IDS.El objetivo es analizar el comportamiento del modelo al interpretar instrucciones
multimodales (imagen y/o texto) y generar acciones en un entorno real, incluyendo procesos de fine-tuning y evaluación experimental.


Estructura del proyecto
------------------------

1. **build_data.py**
   - Script para la recolección de datos de demostración en el entorno real.
   - Captura imágenes, poses del TCP y acciones (incluyendo el estado del gripper).
   - Almacena los datos en formato `.npy` para entrenamiento posterior.

2. **reset_pose.py**
   - Script para reposicionar el robot en su pose inicial antes de comenzar una demostración.
   - Publica comandos URScript y controla la apertura/cierre del gripper.

3. **save_image.py**
   - Captura y guarda una imagen de la cámara IDS.
   - Utilizado para registrar imágenes objetivo (goal images) por tarea.

4. **finetune.py**
   - Script principal de entrenamiento para ajustar el modelo Octo con nuevos datos recogidos.
   - Utiliza configuración definida en `finetune_config.py`.

5. **finetune_config.py**
   - Contiene todos los parámetros y rutas del proceso de fine-tuning.
   - Define el tipo de modalidad (texto, imagen, o multimodal), arquitectura congelada, parámetros de entrenamiento, etc.

6. **inference_finetuned_full_trajectory.py**
   - Ejecuta inferencias del modelo fine-tuned con secuencias completas de observaciones.
   - Evalúa en entorno real publicando comandos URScript para realizar tareas.


Requisitos y configuración
--------------------------
- **Sistema operativo**: Ubuntu 22.04 LTS
- **ROS 2**: Humble Hawksbill
- **Robot**: UR5e (IP: 169.254.128.101)
- **Cámara IDS**: GV-5860CP (IP: 169.254.128.216)
- **Red**: Configurar la interfaz de red del PC con IP 169.254.128.210 y máscara 255.255.0.0


Pasos básicos del workflow
---------------------------
1. **Conexión y preparación**
   - Conectar el PC al robot y a la cámara vía Ethernet.
   - Configurar IPs como indicado en el archivo `workflow_ur5e.txt`.
   - Comprobar conexión con `ping`.

2. **Inicialización**
   - Terminal 1:  
     `ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5e robot_ip:=169.254.128.101 kinematics_params_file:="${HOME}/calibrations/my_robot_calibration.yaml"`
   - Terminal 2:  
     `ros2 launch aravis_driver_ros2 aravis_camera1_launch.xml`
   - Terminal 3:  
     (Opcional) Cargar trayectorias en el Teach Pendant.
   - Terminal 4:  
     Ejecutar `rqt` para monitorización.

3. **Recolección de datos**
   - Ejecutar `reset_pose.py` para inicializar la posición.
   - Ejecutar `build_data.py` para comenzar la recolección de demostraciones.

4. **Fine-tuning**
   - Transferir datos a servidor con acceso a GPU.
   - Lanzar `finetune.py` con la configuración deseada.

5. **Inferencia y validación**
   - Ejecutar `inference_finetuned_full_trajectory.py` en el entorno real.
   - Evaluar las predicciones y comportamiento del robot frente a las instrucciones dadas.


Notas finales
-------------
- Para que todo funcione correctamente se deberá renombrar los archivos "cmakelists.txt" a "CMakeLists.txt"
- El conjunto de datos se puede descargar en el siguiente enlace: https://nubeusc-my.sharepoint.com/:u:/g/personal/adrian_losada_alvarez_rai_usc_es/EdM1PONbKiRCmhFU3Wgw_-kB838ehijZmLYjicw9LEuVSw?e=PgdKMW
- Todos los scripts usan tópicos ROS 2 y nodos diseñados para trabajar en paralelo usando `MultiThreadedExecutor`.
- Se ha trabajado principalmente con el modelo `octo-small-1.5`, utilizando su versión fine-tuned en tres tareas diferentes.
- Las instrucciones sobre la instalación completa del entorno se encuentran en el Apéndice C de la memoria escrita del TFG.