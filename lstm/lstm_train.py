import jax
import jax.numpy as jnp
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import optax
import pickle
from lstm import LSTM

sns.set_style("darkgrid")
points = 1000
key = jax.random.PRNGKey(seed=42)
data = jnp.arange(0, points) * 1/30 + jax.random.normal(key=key, shape=(points,))

line_plot = sns.lineplot(data)
plt.title("Plot of synthetic data")
plt.xlabel("Timestep")
plt.ylabel("Value")
fig = line_plot.get_figure()
fig.savefig("line_plot.png")

def preprocess_ts(
    x: jnp.ndarray, 
    y: jnp.ndarray,
    timestep: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    x_window = np.lib.stride_tricks.sliding_window_view(
        x, window_shape=(timestep,), axis=0
    )
    y_window = np.lib.stride_tricks.sliding_window_view(
        y, window_shape=(timestep,), axis=0
    )

    x_window = jnp.expand_dims(x_window, axis=(1, 3, 4))
    y_window = jnp.expand_dims(jnp.expand_dims(jnp.array(y_window), axis=2), axis=(3, 4))
    return jnp.array(x_window), jnp.array(y_window)

x_window, y_window = preprocess_ts(data, data[1:], 5)

num_epochs = 200
batch_size = 32
num_batches = x_window.shape[0] // batch_size
learning_rate = 1e-3

optimiser = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=learning_rate)
)

archi_params, params = LSTM.init_params(seed=42, input_dim=1, hidden_dim=70, output_dim=1)

opt_state = optimiser.init(params)

for epoch in range(num_epochs):
    print(f"Epoch: {epoch + 1}")
    results = LSTM.forward_batch(archi_params, params, x_window)
    fig, ax = plt.subplots()
    sns.lineplot(results[:,-1,-1], ax=ax)
    sns.lineplot(data, ax=ax)
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.title("Plot of Prediction and Data")
    fig.savefig(f"plot_{epoch}.png")
    with open(f"save_file_{epoch}.pkl", "wb") as f:
        pickle.dump(params, f)

    for i in range(num_batches):
        x_batch = x_window[i * batch_size : (i + 1) * batch_size]
        y_batch = y_window[i * batch_size : (i + 1) * batch_size]

        cur_grad = LSTM.backward(archi_params, params, x_batch, y_batch)[0]
        if (jnp.any(jnp.isnan(cur_grad.wf))):
            continue
        updates, opt_state = optimiser.update(cur_grad, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        params = new_params

fig, ax = plt.subplots()
sns.lineplot(results[:,-1,-1], ax=ax)
sns.lineplot(data, ax=ax)
plt.xlabel("Timestep")
plt.ylabel("Value")
plt.title("Plot of Prediction and Data")
fig.savefig(f"plot_final.png")

with open(f"save_file_final.pkl", "wb") as f:
    pickle.dump(params, f)
