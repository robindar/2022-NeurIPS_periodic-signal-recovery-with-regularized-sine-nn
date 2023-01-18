import numpy as np
import jax

configurations_available = {}

#####  NeurReps - Experiments with Periodic Recovery #####
#
#  E04 : neuron-tracking with different regularizations
#
#  E05 : single-tone periodic signal
#
#  E06 : multi-tone periodic signal
#
#  E11 : frequency recovery (E11b for 100 samples, E11c for 1000 samples)


## E04

freq = 1.115
delta = 1.0 / 10.0
E04_config = {
        "B":  { "m": 50, "delta": delta, "alpha":  None, "lam": None, "reg":  None, "rand_init": True  },
        "C":  { "m": 50, "delta": delta, "alpha":  None, "lam": None, "reg":  None, "rand_init": False },
        "D":  { "m": 50, "delta": delta, "alpha":  None, "lam": 1.00, "reg":  "l1", "rand_init": False },
        "E":  { "m": 50, "delta": delta, "alpha":  None, "lam": 1.00, "reg": "exp", "rand_init": False },
        }

for letter, params in E04_config.items():
    configurations_available[f"E04_{letter}3"] = {
            "random_seed": 1,
            "generator": ["Sine", { "freq": freq, "k": 4, "R": 10.0 }],
            "training_config": {
                "model": [ "SineD", params ],
                "num_iterations": 5e5,
                "step_size": 1e-3,
                "optimizer": "momentum",
                "batch_size": 100,
                "n": 1e3,
                },
            "recorder_config": {
                "print_every": 1e1,
                "dump_every": 5e2,
                "savedir": "data/",
                },
            }


## E05

delta = np.pi * 1.0 / 10.0
E05_config = {
        "A":  [ "MLP", { "l": 2, "m": 1200 } ],
        "B":  [ "SineD", { "m": 1200, "delta": delta, "alpha":  None, "lam": None, "reg":  None, "rand_init": True  } ],
        "C":  [ "SineD", { "m": 50, "delta": delta, "alpha":  None, "lam": 1.00, "reg": "exp", "rand_init": False } ],
        "D":  [ "MLP", { "l": 2, "m": 1200, "nn":  "tanh" } ],
        "E":  [ "MLP", { "l": 2, "m": 1200, "nn":  "snake" } ],
        "Z":  [ "MLP", { "l": 4, "m": 50 } ],
        "Y":  [ "SineD", { "m": 120, "delta": delta, "alpha":  None, "lam": None, "reg":  None, "rand_init": False  } ],
        "X":  [ "SineD", { "m": 1200, "delta": delta/10, "alpha":  None, "lam": None, "reg":  None, "rand_init": False  } ],
        }
arch_names = {
        "A": "2-layer ReLU",
        "B": "Sine (normal init)",
        "C": "Sine (regularized)",
        "D": "2-layer Tanh",
        "E": "2-layer Snake",
        "Z": "4-layer ReLU",
        "Y": "Sine (unregularized, diverse init, $\delta=\pi/10$)",
        "X": "Sine (unregularized, diverse init, $\delta=\pi/100$)",
        }

for letter, params in E05_config.items():
    configurations_available[f"E05_{letter}2"] = {
            "random_seed": 1,
            "generator": ["Sine", { "freq": 2, "k": 1, "R": 10.0, "phase": +np.pi/12 }],
            "arch_name": arch_names[letter],
            "training_config": {
                "model": params,
                "num_iterations": 1e7,
                "step_size": 1e-4,
                "optimizer": "adam" if letter in [ "A", "B", "C", "D", "E", "Z" ] else "momentum",
                "batch_size": 100,
                "n": 1e3,
                },
            "recorder_config": {
                "print_every": 1e1,
                "dump_every": 1e3,
                "savedir": "data/",
                },
            }


## E06

delta = np.pi * 1.0 / 10.0
E06_config = {
        "A":  [ "MLP", { "l": 2, "m": 12000 } ],
        "B":  [ "SineD", { "m": 5000, "delta": delta, "alpha":  None, "lam": None, "reg":  None, "rand_init": True  } ],
        "C":  [ "SineD", { "m": 1000, "delta": delta, "alpha":  None, "lam": 1.00, "reg": "exp", "rand_init": False } ],
        "E":  [ "MLP", { "l": 2, "m": 12000, "nn":  "snake" } ],
        "Z":  [ "MLP", { "l": 4, "m": 80 } ],
        }
arch_names = {
        "A": "2-layer ReLU",
        "B": "Sine (normal init)",
        "C": "Sine (regularized)",
        "E": "2-layer Snake",
        "Z": "4-layer ReLU",
        }

for letter, params in E06_config.items():
    configurations_available[f"E06_{letter}2"] = {
            "random_seed": 1,
            "generator": ["Sine", { "freq": 2, "k": 25, "R": 10.0, "phase": +np.pi/12 , "noise_std": 0}],
            "arch_name": arch_names[letter],
            "training_config": {
                "model": params,
                "num_iterations": 1e7,
                "step_size": 1e-4,
                "optimizer": "adam" if letter in [ "A", "B", "C", "D", "E", "Z" ] else "momentum",
                "batch_size": 100,
                "n": 1e3,
                },
            "recorder_config": {
                "print_every": 1e1,
                "dump_every": 1e4,
                "savedir": "data/",
                },
            }


## E11

frequencies = jax.random.uniform(jax.random.PRNGKey(42), shape=(100,), minval=1.5, maxval=15)
noise_levels = [ np.sqrt(0.5), 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9 ]
for letter, samples in [ ('a', 25), ('b', 100), ('c', 1000) ]:
  for i, freq in enumerate(frequencies):
    for n_idx, n in enumerate(noise_levels):
      configurations_available[f"E11{letter}_{i:03d}_{n_idx:02d}"] = {
              "random_seed": 1000*(n_idx+1)+i+1,
              "generator": ["Sine", { "freq": float(freq), "k": 1, "R": 10.0, "phase": +np.pi/12, "noise_std": n }],
              "arch_name": "Sine (regularized)",
              "training_config": {
                  "model":  [ "SineD", { "m": 50, "delta": delta, "alpha":  None, "lam": 1.00, "reg": "exp", "rand_init": False } ],
                  "num_iterations": 1e5,
                  "step_size": 1e-3,
                  "optimizer": "momentum",
                  "batch_size": samples,
                  "n": samples,
                  },
              "recorder_config": {
                  "print_every": 1e1,
                  "dump_every": 1e3,
                  "savedir": "data",
                  },
              }
