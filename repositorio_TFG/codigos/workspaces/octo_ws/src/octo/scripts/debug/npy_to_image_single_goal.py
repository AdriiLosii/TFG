import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import os

def extract_episode_index(filename):
    match = re.search(r'episode_(\d+)\.npy$', filename)
    return int(match.group(1)) if match else -1

# Find all episode_*.npy files in the directory
file_list = glob.glob("../data/episode_*.npy")
file_list = sorted(file_list, key=extract_episode_index)

if not file_list:
    raise FileNotFoundError("No episode_*.npy files found in ../data/")

# Procesar un solo archivo (puedes iterar sobre file_list si quieres)
filename = "../data/episode_237.npy"
print(f"\n--- Viewing: {os.path.basename(filename)} ---")

# Load the data
data = np.load(filename, allow_pickle=True).item()

# Extract frames
frames = data['steps']
print(f"Found {len(frames)} frames")

# Habilitar modo interactivo para múltiples ventanas
plt.ion()

for i, frame in enumerate(frames):
    try:
        # 1) Obtener imagen de observación
        obs = frame['observation']
        if isinstance(obs, dict):
            image = None
            for key in ['image', 'rgb', 'pixels', 'camera']:
                if key in obs:
                    image = obs[key]
                    break
            if image is None:
                print(f"No image found in observation dict for frame {i}")
                continue
        else:
            image = obs

        # Convertir a uint8 y a 3 canales si hace falta
        if hasattr(image, 'dtype') and image.dtype == object:
            try:
                image = np.array(image, dtype=np.uint8)
            except Exception as e:
                print(f"Could not convert object array to image in frame {i}: {e}")
                continue
        if image.ndim == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.ndim == 2:
            image = np.stack([image]*3, axis=2)

        # 2) Obtener imagen objetivo de "image_primary"
        goal_image = frame.get('image_primary', None)
        if goal_image is not None:
            if hasattr(goal_image, 'dtype') and goal_image.dtype == object:
                try:
                    goal_image = np.array(goal_image, dtype=np.uint8)
                except:
                    pass
            if goal_image.ndim == 3 and goal_image.shape[2] == 1:
                goal_image = np.repeat(goal_image, 3, axis=2)
            elif goal_image.ndim == 2:
                goal_image = np.stack([goal_image]*3, axis=2)

        # 3) Obtener acción
        action = frame.get('action', np.zeros(8))

        # 4) Mostrar simultáneamente en dos ventanas
        # Ventana 1: observación + acción
        fig1 = plt.figure(num=1)
        plt.imshow(image)
        plt.title(
            f"{os.path.basename(filename)} - Frame {i}\n"
            f"Action:\nX: {action[0]:.3f} Y: {action[1]:.3f} Z: {action[2]:.3f}\n"
            f"R: {action[3]:.3f} P: {action[4]:.3f} Y: {action[5]:.3f}\n"
            f"G: {action[6]:.0f}"
        )
        plt.axis("off")

        # Ventana 2: imagen objetivo
        if goal_image is not None:
            fig2 = plt.figure(num=2)
            plt.imshow(goal_image)
            plt.title(f"Goal Image - Frame {i}")
            plt.axis("off")

        # Pausar para que se actualicen ambas ventanas
        plt.pause(0.1)

        # Limpiar figuras para el siguiente ciclo
        fig1.clf()
        if goal_image is not None:
            fig2.clf()

    except Exception as e:
        print(f"Error processing frame {i}: {e}")
        continue

plt.close('all')