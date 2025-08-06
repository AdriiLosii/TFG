import argparse
import tqdm
import importlib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress debug warning messages
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import wandb


WANDB_ENTITY = None
WANDB_PROJECT = 'vis_rlds'


parser = argparse.ArgumentParser()
parser.add_argument('dataset_name', help='name of the dataset to visualize')
args = parser.parse_args()

if WANDB_ENTITY is not None:
    render_wandb = True
    wandb.init(entity=WANDB_ENTITY,
               project=WANDB_PROJECT)
else:
    render_wandb = False


# create TF dataset
dataset_name = args.dataset_name
print(f"Visualizing data from dataset: {dataset_name}")
ds = tfds.load(dataset_name, split='train')
ds = ds.shuffle(100)

# Count unique language instructions
language_instructions = set()
for episode in tqdm.tqdm(ds.take(500)):
    for step in episode['steps']:
        instruction = step['language_instruction'].numpy().decode()
        language_instructions.add(instruction)

print(f"Number of unique language instructions: {len(language_instructions)}")

# visualize action and state statistics
actions, states = [], []
for episode in tqdm.tqdm(ds.take(500)):
    for step in episode['steps']:
        actions.append(step['action'].numpy())
actions = np.array(actions)
action_mean = actions.mean(0)

def vis_stats(vector, vector_mean, tag):
    assert len(vector.shape) == 2
    assert len(vector_mean.shape) == 1
    assert vector.shape[1] == vector_mean.shape[0]

    n_elems = vector.shape[1]
    fig = plt.figure(tag, figsize=(5*n_elems, 5))
    for elem in range(n_elems):
        plt.subplot(1, n_elems, elem+1)
        plt.hist(vector[:, elem], bins=20)
        plt.title(vector_mean[elem])

    if render_wandb:
        wandb.log({tag: wandb.Image(fig)})

vis_stats(actions, action_mean, 'action_stats')

if not render_wandb:
    plt.show()
