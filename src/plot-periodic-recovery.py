import os
import sys
import yaml
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap

import matplotlib
import matplotlib.pyplot as plt

import recording
import serialization
import structures_d.structures as structures

DTYPE = jnp.float64
R = 10.0
K_order = 4                  # measure performance up to (k = 10 ** K_order) windows
rng = jax.random.PRNGKey(1)  # random generator used for test-error sampling

column_count = 3
learned_line_width = 1.0
error_line_width = 1.0

fig = plt.figure(figsize=(8, 6))
gs = fig.add_gridspec(2,1)
gs_top = gs[0].subgridspec(1, column_count, hspace=.01)
ax_main_one = fig.add_subplot(gs_top[:-1])
ax_main_bis = fig.add_subplot(gs_top[-1], sharey=ax_main_one)
ax_main_two = fig.add_subplot(gs[1])


has_plotted_reference = False


for raw_filename in sys.argv[2:]:
    print(f"\r\033[KLoading file {raw_filename}", end='', flush=True)
    fp = open(raw_filename, "r")
    data = yaml.safe_load(fp)
    fp.close()

    eid = data["experiment_id"]
    datakey, starkey, initkey, trainkey = jax.random.split(jax.random.PRNGKey(data["random_seed"]), 4)
    generator_class, generator_static_parameters = data["generator"]
    generator = structures.Generator.class_dict[generator_class](starkey, **generator_static_parameters)

    n = data["training_config"]["n"]
    batch_size = data["training_config"]["batch_size"]
    keyring = jax.random.split(datakey, n // batch_size)

    data_seen = jnp.hstack([ generator.sample_data(batch_size, data_subkey) for data_subkey in keyring ])
    data_seen = data_seen[0,:]

    model_class_name, model_static_parameter_dict = data["training_config"]["model"]
    model_class = structures.Model.class_dict[model_class_name]
    model_static_parameters = model_class.Hyperparameters(**model_static_parameter_dict)
    model_forward = vmap(model_class.forward, in_axes=(None, None, 0, None), out_axes=0)

    def forward(params, u):
        y = model_forward(model_static_parameters, params, u, None)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:,0]
        return y

    data_entry = data["training_data"][-1]
    params = serialization.deserialize_nested(data_entry["params"])

    x = np.linspace(-R / 2, 5 * R, 1000)
    y = forward(params, x)
    y_star = generator.forward(x)

    x_span = (np.max(x) - np.min(x)) / (column_count - 1)
    x_offset = 97 * R

    x_bis = np.linspace(x_offset, x_offset + x_span, 1000)
    y_bis = forward(params, x_bis)
    y_star_bis = generator.forward(x_bis)

    label = data["arch_name"] if ("arch_name" in data) else eid

    if not has_plotted_reference:
        ax_main_one.plot(x, y_star, color='black', label="Signal")
        ax_main_one.set_xlim(np.min(x), np.max(x))
        has_plotted_reference = True

        ax_main_bis.plot(x_bis, y_star_bis, color='black', label="Signal")
        ax_main_bis.set_xlim(np.min(x_bis), np.max(x_bis))

    learned_line, = ax_main_one.plot(x, y, label=label, linewidth=learned_line_width)
    ax_main_bis.plot(x_bis, y_bis, label=label, color=learned_line.get_color(), linewidth=learned_line_width)

    n_hat = int(1e4)  # number of points sampled per round to estimate the k-window error
    n_rounds = 10     # number of rounds to estimate k-window error (cannot merge with n_hat to avoid OOM Errors)

    K = 1 + 9 * K_order + 1
    u = np.zeros(K)
    v = np.ones(K)
    idx = 1
    for o in range(K_order+1):
        for j in range(1, 10 if o < K_order else 2):
            print(f"\r\033[KComputing error for {eid} [{idx}/{K}]", end='', flush=True)
            i = j * (10 ** o)
            u[idx] = i
            v[idx] = 0
            for _ in range(n_rounds):
                rng, subkey = jax.random.split(rng, 2)
                x = jax.random.uniform(subkey, shape=(n_hat,), minval=0, maxval=i*R)
                y = forward(params, x)
                y_star = generator.forward(x)
                assert y.shape == y_star.shape, f"Got y shape {y.shape} but y_star with shape {y_star.shape}"
                v[idx] += jnp.mean(jnp.square(y - y_star)) / n_rounds
            idx += 1
    v[0] = jnp.mean(jnp.square(forward(params, data_seen) - generator.forward(data_seen)))

    x = jax.random.uniform(rng, shape=(int(1e6),), minval=0, maxval=100*R)
    squared_norm = jnp.mean(jnp.square(generator.forward(x)))

    ax_main_two.plot(u[1:], v[1:], '-o', color=learned_line.get_color(), label=label, clip_on=False, zorder=10, linewidth=error_line_width, markersize=4)

print("\r\033[K", end='')

ax_main_one.set_ylim(-2.5, +3.5)

y_min, y_max = ax_main_one.get_ylim()
ax_main_one.set_ylim(y_min, y_max)
ax_main_one.vlines([0.0, R], y_min, y_max, linewidth=1, color='red', linestyles='dotted')
ax_main_one.grid(alpha=.2)

ax_main_one.legend(fontsize='small', loc='upper right').set_zorder(20)

ax_main_one.set_xlabel("Input value")
ax_main_one.set_ylabel("Learned function")

ax_main_bis.grid(alpha=.2)
ax_main_bis.tick_params(labelleft=False)

ax_main_two.set_yscale('log')
ax_main_two.set_xlabel(r"Number of measured windows $k$")
ax_main_two.set_ylabel(r"Average Error $\mathcal{L}_k$")
xticks = [ i * 10. ** j for j in range(4) for i in np.arange(1, 10) ]
ax_main_two.set_xscale('log')
ax_main_two.set_xticks(xticks, labels = [ "" ] * len(xticks), minor=True)
ax_main_two.set_xlim(1, np.max(u))

y_min, y_max = ax_main_two.get_ylim()
ax_main_two.set_yticks(10. ** (-22 + np.arange(40)), labels=[ "" ] * 40, minor=True)
ax_main_two.grid(alpha=.2, which='both')
ax_main_two.set_ylim(y_min, y_max)
ax_main_two.vlines(1.5, y_min, y_max, linewidth=.5, color='black', linestyles='dotted')
ax_main_two.hlines(squared_norm, *ax_main_two.get_xlim(), linestyles='dashed', color='grey', label="Signal squared norm")
ax_main_two.legend(fontsize='small', loc='lower right').set_zorder(20)

plt.tight_layout()
plt.savefig(f'images/{sys.argv[1]}.png', dpi=500)
