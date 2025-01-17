from jax import jit, random, value_and_grad
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import os
import time

import unet
import utils
import mnist

@jit
def forward_process(x_0, t, eta):
    assert x_0.shape[0] == t.shape[0]
    assert x_0.shape == eta.shape
    assert len(t.shape) == 1

    n = x_0.shape[0]
    # equation 17
    return x_0 * jnp.exp(-t/2).reshape((n, 1, 1, 1)) + eta * jnp.sqrt(1-jnp.exp(-t)).reshape((n, 1, 1, 1))

def fit(state, training_data, time_schedule, key, batch_size, n_epoch, step=0, epoch_start=0):
    @jit
    def lambda_t(t):
        return jnp.where(t>0.0, t/(jnp.exp(t)-1.), jnp.ones(t.shape))

    @jit
    def weighted_mse_loss(params, inputs, time, targets):
        assert inputs.shape[0] == time.shape[0]

        n = targets.shape[0]
        predictions = predict_fn(params, inputs, time)
        weight = lambda_t(time) # (n,)
        diffs = predictions.reshape((n, - 1)) - targets.reshape((n, -1)) # differences of the flattened images (n, 28*28)
        return (weight * (diffs * diffs).mean(axis=1)).mean()

    n_batch=training_data.shape[0] // batch_size

    loss_log = None
    best_loss = 1.
    predict_fn = state.apply_fn

    ks = jnp.array(range(len(time_schedule)))

    for epoch in range(epoch_start, n_epoch):
      key, subkey = random.split(key)
      perms = random.permutation(subkey, training_data.shape[0])
      perms = perms[: n_batch * batch_size] # skip incomplete batch
      perms = perms.reshape((n_batch, batch_size))

      loss_log = []

      for perm in tqdm(perms, desc=f'epoch {epoch}'):

          # randomly pick a subset of the entire sample size
          x_0_batch = training_data[perm, ...]

          # regenerate a new random keys
          key, key2, key3 = random.split(key, 3)

          t = random.choice(key2, time_schedule, shape=(x_0_batch.shape[0],))
          eta = random.normal(key3, shape=x_0_batch.shape)

          training_inputs = forward_process(x_0_batch, t, eta)
          training_targets = x_0_batch

          loss, grads = value_and_grad(weighted_mse_loss)(state.params, training_inputs, t, training_targets)

          state = state.apply_gradients(grads=grads)

          step = step+1
          loss_log.append((epoch, step, loss))

      utils.save_checkpoint(CHECKPOINT_DIR, state, epoch, step)
      utils.save_loss_log(loss_log, LOSS_LOG)

      epoch_loss = np.mean([loss for _, _, loss in loss_log])

      if epoch_loss < best_loss:
          best_loss = epoch_loss
          utils.save_pytree(state.params, f'{PROJECT_DIR}/cem_params_{epoch}_{step}_{best_loss:.5f}')

    return state

if __name__ == '__main__':
    PROJECT_DIR=os.path.abspath('.')
    CHECKPOINT_DIR=os.path.abspath('/tmp/cem')
    LOSS_LOG= f'{PROJECT_DIR}/cem_loss_log.npy'
    SEED=42
    BATCH_SIZE=10
    N_EPOCH=10
    T = 10.
    K = 200

    key = random.PRNGKey(SEED)
    key, key2, key3 = random.split(key, 3)

    RESUME=False
    epoch_start=-1
    step=-1

    if RESUME:
        state, epoch_start, step = utils.restore_checkpoint(CHECKPOINT_DIR)
        print(f'restore checkpoint from epoch {epoch_start} and step {step}')
    else:
        state = utils.create_training_state(key=key2)
        utils.save_checkpoint(CHECKPOINT_DIR, state, epoch_start, step)

    training_data = mnist.get_training_data()[:1000]
    time_schedule = utils.exponential_time_schedule(T, K)[1:] # ignore 0.0

    start = time.time()
    state = fit(state, training_data, time_schedule, key3, BATCH_SIZE, N_EPOCH, step=step+1, epoch_start=epoch_start+1)
    end = time.time()

    print(f'elapsed: {end - start}s')
