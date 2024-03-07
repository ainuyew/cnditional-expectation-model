import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax import random
import os
import seaborn as sns
import orbax.checkpoint as ocp
import optax
from flax.training import train_state, orbax_utils
import glob

from unet import Unet

sns.set_theme()

# HELPER FUNCTIONS
def calculate_moments(xs, N=7, columns=None):
    ys = np.array(xs.flatten(), dtype=np.float64)
    return np.array([sp.stats.moment(ys, n) for n in range(1, N)])

def print_moments(xs, N=7, columns=None):
    for index in range(2):
        df = pd.DataFrame([[n for n in range(1, N)]] + [[stats.moment(x, n)[index] for n in range(1, N)] for x in xs]).transpose()
        df.columns = ['moment'] + (list(range(len(xs))) if columns is None else columns)

        print(f'\nmoments for x_{index+1}')
        print(tabulate(df, headers=df.columns, tablefmt='orgtbl', showindex=False))

def histplot2d(x):
    fig, ax = plt.subplots()
    df = pd.DataFrame(x, columns=['x1', 'x2'])
    sns.histplot(df, x='x1', y='x2', bins=30, cbar=True)
    #_ = ax.set_xlim(-5.5,5.5)
    #_ = ax.set_xticks([-4, -2, 0, 2, 4])
    #_ = ax.set_ylim(-5.5 ,5.5)
    #_ = ax.set_yticks([-4, -2, 0, 2, 4])
    _ = plt.show()

def find_latest_file(file_path):
    list_of_files = glob.glob(file_path) # * means all if need specific format then *.csv
    return max(list_of_files, key=os.path.getctime)

def find_latest_pytree(file_path):
    latest_file = find_latest_file(file_path)
    parts = latest_file.split('_')
    return latest_file, int(parts[-3]), int(parts[-2]), float(parts[-1][:-4])

def save_pytree(params, file_path):
    flat_params, _ = ravel_pytree(params)
    jnp.save(file_path, flat_params)

def load_pytree(params, file_path):
    flat_params = jnp.load(file_path)
    _, unravel_params = ravel_pytree(params)
    return unravel_params(flat_params)

def load_loss_log(file_path):
    if os.path.isfile(file_path):
        f =  open(file_path, 'rb')
        loss_log=np.load(f)
        f.close()
    else:
        loss_log=np.array([])
    return loss_log

def save_loss_log(loss_log, file_path):
    original_loss_log=load_loss_log(file_path)
    if original_loss_log.shape[0] > 0:
        loss_log = np.append(original_loss_log, loss_log, axis=0)
    with open(file_path, 'wb') as f:
        np.save(f, loss_log)
    return loss_log

def create_training_state(params_file=None, key=None):
    if key is None:
        key = random.PRNGKey(42)

    key, subkey = random.split(key)

    neural_network = Unet(
        dim=64,
        dim_mults = (1, 2, 4,),
    )

   # optimizer = optax.adam(learning_rate=1e-4, b1=.9, b2=.99, eps=1e-8)
    optimizer = optax.adam(learning_rate=2e-5)

    params = neural_network.init(subkey, jnp.ones([1, 28, 28, 1]), jnp.ones((1,)))
    if params_file:
        params = load_pytree(params, params_file)

    # Create training state
    state = train_state.TrainState.create(
        apply_fn = neural_network.apply,
        params = params,
        tx=optimizer
    )

    return state

def save_checkpoint(file_path, state, epoch, step):
    #ocp.test_utils.create_empty(PROJECT_DIR) # ensure directory exists before saving
    #ckptr = ocp.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())  # A stateless object, can be created on the fly.
    ckptr = ocp.Checkpointer(ocp.CompositeCheckpointHandler('state', 'metadata'))
    #ckpt = {'model': state, 'epoch': epoch, 'step': step}
    metadata = {
        'epoch': epoch,
        'step': step
        }
    #args = orbax_utils.save_args_from_target(ckpt) # TODO this is deprecated. https://orbax.readthedocs.io/en/latest/custom_handlers.html#typehandler
    #ckptr.save(file_path, ckpt, save_args=args, force=True)
    ckptr.save(file_path,
               args=ocp.args.Composite(
                   state=ocp.args.StandardSave(state),
                   metadata=ocp.args.JsonSave(metadata),
               ),
               force=True)

def restore_checkpoint(file_path, key=None):
    #ckptr = ocp.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())  # A stateless object, can be created on the fly.
    ckptr = ocp.Checkpointer(ocp.CompositeCheckpointHandler('state', 'metadata'))
    #ckpt = {'model': create_training_state(key), 'epoch': -1, 'step': -1}
    #args = orbax_utils.restore_args_from_target(ckpt, mesh=None)
    #args = ocp.args.StandardSave(ckpt)
    #ckpt = ckptr.restore(file_path, item=ckpt, restore_args = args)
    ckpt = ckptr.restore(file_path)
    return ckpt.state, ckpt.metadata['epoch'], ckpt.metadata['step']

# Helper functions for images.
def show_img(img, ax=None, title=None):
    """Shows a single image."""
    if ax is None:
      ax = plt.gca()
    ax.imshow(img[..., 0], cmap='gray')
    _ = ax.set_xticks([])
    _ = ax.set_yticks([])
    if title:
      _ = ax.set_title(title)

def show_img_grid(imgs, titles=None):
    """Shows a grid of images."""
    n = int(np.ceil(len(imgs)**.5))
    _, axs = plt.subplots(n, n, figsize=(3 * n, 3 * n))
    _ = plt.tight_layout()
    if titles is None:
        titles = [None] * len(imgs)
    for i, (img, title) in enumerate(zip(imgs, titles)):
        show_img(img, axs[i // n][i % n], title)

# TIME SCHEDULES
def linear_time_schedule(min_beta, max_beta, timesteps):
    return np.linspace(min_beta, max_beta, timesteps, dtype=jnp.float64)

def exponential_time_schedule(T, K):
    t1 = T/K
    gamma = 1 - (T/t1)**(1/(1-K)) # equation 42
    ts = [0.] + [t1*(1-gamma)**(1-k) for k in range(K)] + [T]
    return np.array(ts)

def normalize_to_zero_to_one(x):
    return (x - x.min())/(x.max() - x.min())

def normalize_to_neg_one_to_one(x):
    return normalize_to_zero_to_one(x) * 2 - 1

def unnormalize_image(xs):
    assert len(xs.shape) == 4
    ys = []
    for x in xs:
      ys.append(normalize_to_zero_to_one(x) * 255)
    return ys
