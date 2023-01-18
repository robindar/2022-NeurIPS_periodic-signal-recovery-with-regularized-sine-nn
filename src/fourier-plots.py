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


fig = plt.figure(figsize=(6, 6))
gs = fig.add_gridspec(2,2)
ax_main_0 = fig.add_subplot(gs[0,0])
ax_main_1 = fig.add_subplot(gs[0,1], sharey=ax_main_0)
ax_main_2 = fig.add_subplot(gs[1,0], sharey=ax_main_0)
ax_main_3 = fig.add_subplot(gs[1,1], sharey=ax_main_0)
axes = [ ax_main_0, ax_main_1, ax_main_2, ax_main_3 ]

titles = {
        "E04_B3": "Normal iid initalization",
        "E04_C3": "Unregularized",
        "E04_D3": "L1 regularization",
        "E04_E3": "Non-convex reg. (ours)",
        }


for idx, (raw_filename, ax) in enumerate(zip(sys.argv[1:], axes)):
    print(f"\r\033[KLoading file {raw_filename}", end='', flush=True)
    fp = open(raw_filename, "r")
    data = yaml.safe_load(fp)
    fp.close()

    eid = data["experiment_id"]
    datakey, starkey, initkey, trainkey = jax.random.split(jax.random.PRNGKey(data["random_seed"]), 4)
    generator_class, generator_static_parameters = data["generator"]
    generator = structures.Generator.class_dict[generator_class](starkey, **generator_static_parameters)

    title = titles[eid]

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

    trajectory_u, trajectory_v = [], []
    for data_entry in data["training_data"]:
        params = serialization.deserialize_nested(data_entry["params"])
        w, a, b = params
        r = jnp.sqrt(a ** 2 + b ** 2)
        trajectory_u.append(w)
        trajectory_v.append(r)
    trajectory = np.stack([ trajectory_u, trajectory_v])
    for i in range(trajectory.shape[2]):
        ax.plot(trajectory[0,:,i], trajectory[1,:,i], color='red', alpha=1., linewidth=1)

    data_entry = data["training_data"][-1]
    params = serialization.deserialize_nested(data_entry["params"])

    k = 99
    x = jax.random.uniform(jax.random.PRNGKey(2), shape=(int(1e7),), minval=0, maxval=k * 10.0)
    y_star = generator.forward(x)
    y = forward(params, x)
    lk_loss = jnp.mean(jnp.square(y - y_star))

    w, a, b = params
    r = jnp.sqrt(a ** 2 + b ** 2)

    ax.scatter(w, r, s=2, color='black', zorder=10)
    ax.scatter(*generator.coordinates(), s=12, marker='d', edgecolor='green', color=None)
    ax.set_title(title + "\n" + r"($\mathcal{L}_0$" + "={:.0e},".format(data_entry["train_loss"]) + r" $\mathcal{L}_{" + str(k) + "}$" + "={:.0e})".format(lk_loss), fontsize='medium')
    ax.set_xticks(([-1] if idx == 0 else []) + [0, 1, 2, 3, 4, 5], minor=True)
    ax.grid(alpha=.1, which='both')
print("\r\033[K", end='')

ax_main_0.set_ylim(None, 1.2)

ax_main_0.set_xlabel(r"Frequency $\omega_i$")
ax_main_1.set_xlabel(r"Frequency $\omega_i$")
ax_main_2.set_xlabel(r"Frequency $\omega_i$")
ax_main_3.set_xlabel(r"Frequency $\omega_i$")
ax_main_0.set_ylabel(r"Amplitude $\sqrt{a_i^2 + b_i^2}$")
ax_main_2.set_ylabel(r"Amplitude $\sqrt{a_i^2 + b_i^2}$")

ax_main_1.tick_params(labelleft=False)
ax_main_3.tick_params(labelleft=False)

plt.tight_layout()
plt.savefig(f'images/fourier.png', dpi=400)
