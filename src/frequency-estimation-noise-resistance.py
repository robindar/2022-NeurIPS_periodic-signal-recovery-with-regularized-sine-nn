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

noise_indices = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]
noise_levels = 10. ** (- np.array(noise_indices))
noise_levels[0] = np.sqrt(0.5)

guessed_errors = [ [] for _ in noise_indices ]
periodogram_errors = [ [] for _ in noise_indices ]

plt.figure(figsize=(4,4.5))  # original submission: (4, 3.5)

EXP_ID = sys.argv[1]
MAX_OMEGA_COUNT = 100 # hardcoded, according to computed experiments

for array_idx, noise_idx in enumerate(noise_indices):
  for omega_idx in range(MAX_OMEGA_COUNT):
    raw_filename = f"data/E{EXP_ID}_{omega_idx:03d}_{noise_idx:02d}.yml"

    if not os.path.exists(raw_filename):
        continue

    print(f"\r\033[KLoading file {raw_filename}", end='', flush=True)
    fp = open(raw_filename, "r")
    data = yaml.safe_load(fp)
    fp.close()

    if not "training_data" in data or data["training_data"] is None:
        continue

    eid = data["experiment_id"]
    datakey, starkey, initkey, trainkey = jax.random.split(jax.random.PRNGKey(data["random_seed"]), 4)
    generator_class, generator_static_parameters = data["generator"]
    generator = structures.Generator.class_dict[generator_class](starkey, **generator_static_parameters)

    n = data["training_config"]["n"]
    batch_size = data["training_config"]["batch_size"]
    keyring = jax.random.split(datakey, n // batch_size)

    data_seen = jnp.hstack([ generator.sample_data(batch_size, data_subkey) for data_subkey in keyring ])
    data_x, data_y = data_seen[0,:], data_seen[1,:]

    model_class_name, model_static_parameter_dict = data["training_config"]["model"]
    model_class = structures.Model.class_dict[model_class_name]
    model_static_parameters = model_class.Hyperparameters(**model_static_parameter_dict)
    model_forward = vmap(model_class.forward, in_axes=(None, None, 0), out_axes=0)


    def forward(params, u):
        y = model_forward(model_static_parameters, params, u)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:,0]
        return y

    data_entry = data["training_data"][-1]
    params = serialization.deserialize_nested(data_entry["params"])

    w, a, b = params
    r = a**2 + b**2

    w_star, a_star = generator.coordinates()

    omega = np.linspace(1, 50, 25000)
    u = data_y[None,:] * jnp.cos(omega[:,None] * data_x[None,:])
    v = data_y[None,:] * jnp.sin(omega[:,None] * data_x[None,:])
    periodogram = u.sum(axis=1) ** 2 + v.sum(axis=1) ** 2

    periodogram_w = omega[np.argmax(periodogram)]

    guessed_errors[array_idx].append(np.square(w_star[0] - w[np.argmax(r)]))
    periodogram_errors[array_idx].append(np.square(w_star[0] - periodogram_w))

print("\r\033[K", end='')

t = lambda u: np.sqrt(u)
guessed_data = np.array([ [ t(np.mean(s)), t(np.mean(s)) - np.min(t(s)), np.max(t(s)) - t(np.mean(s)) ]  if len(s) > 0 else [ 0, 0, 0 ] for s in guessed_errors ])
periodogram_data = np.array([ [ t(np.mean(s)), t(np.mean(s)) - np.min(t(s)), np.max(t(s)) - t(np.mean(s)) ]  if len(s) > 0 else [ 0, 0, 0 ] for s in periodogram_errors ])

inv_noise_levels = 1.0 / noise_levels
snr = (1./2) * 1.0 / noise_levels ** 2

plt.title(r"With $n$ = " + str(n) + " samples", fontsize='small')

plt.errorbar(snr, periodogram_data[:,0], fmt='-o', yerr=periodogram_data[:,[1,2]].T, label="Periodogram maximizer", markersize=3, linewidth=1, capsize=2)
plt.errorbar(snr, guessed_data[:,0], fmt='-o', yerr=guessed_data[:,[1,2]].T, label="Sine-NN max amplitude", markersize=3, linewidth=1, capsize=2)
plt.yscale('log')
plt.xscale('log')

x_min, x_max = plt.gca().get_xlim()

tics = 10.0 ** np.arange(20)
plt.gca().set_xticks(tics, labels=[""]*len(tics), minor=True)

t = np.linspace(x_min, x_max, 10)
y_min, y_max = plt.gca().get_ylim()

plt.xlim(x_min, x_max)
plt.ylim(3e-8, 3e-1)

plt.xlabel(r"Signal to noise ratio $\frac{1}{2 \sigma^2}$")
plt.ylabel(r"Root mean squared error")
plt.legend(fontsize='small', loc='center right')
plt.grid(alpha=.5, which='both', linewidth=.5)
plt.tight_layout()
plt.savefig(f'images/freq-estimation-noise-resistance_E{sys.argv[1]}.png', dpi=450)
