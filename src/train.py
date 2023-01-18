import os
import sys
import time
import math
import jax
import jax.numpy as jnp
from jax import jit, grad, random, vmap
from jax.example_libraries import optimizers

import serialization
import recording
import structures_d.structures as structures
from experiments import configurations_available


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in configurations_available:
        print("Incorrect arguments. Use: <executable> <experiment-id>")
        exit(1)

    eid = sys.argv[1]
    config = configurations_available[eid]

    global_rng = jax.random.PRNGKey(config["random_seed"])
    datakey, starkey, initkey, trainkey = jax.random.split(global_rng, 4)
    del global_rng

    recorder = recording.Recorder(sys.argv[1], **config["recorder_config"])
    training_config = config["training_config"]
    batch_size = int(training_config["batch_size"])
    n = int(training_config["n"])
    assert n % batch_size == 0, f"dataset of size {n} cannot be cleanly cut into batches of size {batch_size}"
    keyring = jax.random.split(datakey, n // batch_size)
    reg_tau = config["training_config"].get("reg_tau", 10)

    generator_class_name, generator_static_parameters = config["generator"]
    generator = structures.Generator.class_dict[generator_class_name](starkey, **generator_static_parameters)

    model_class_name, model_static_parameters = training_config["model"]
    model_class = structures.Model.class_dict[model_class_name]
    model_static_parameters = model_class.Hyperparameters(**model_static_parameters)

    step_size = training_config["step_size"]
    model_neglog_likelihood = vmap(model_class.neglog_likelihood, in_axes=(None, None, 0, 0, None), out_axes=0)

    if training_config["optimizer"] == "sgd":
        opt_init, opt_update, get_params = optimizers.sgd(step_size)
    elif training_config["optimizer"] == "momentum":
        opt_init, opt_update, get_params = optimizers.momentum(step_size, .9)
    elif training_config["optimizer"] == "adagrad":
        opt_init, opt_update, get_params = optimizers.adagrad(step_size, .9)
    elif training_config["optimizer"] == "adam":
        opt_init, opt_update, get_params = optimizers.adam(step_size)
    else:
        raise ValueError(f"Optimizer '{training_config['optimizer']}' not supported")

    def loss(params, static_params, x, y, reg_param, rng):
        nll = model_neglog_likelihood(static_params, params, x, y, rng)
        reg = model_class.regularizer(static_params, params, x, y, rng)
        return jnp.mean(nll) + reg_param * reg

    @jit
    def update(iteration_idx, opt_state, x, y, reg_param, rng):
        params = get_params(opt_state)
        grads = grad(loss)(params, model_static_parameters, x, y, reg_param, rng)
        return opt_update(iteration_idx, grads, opt_state)

    try:
        init_params = model_class.init_random_params(model_static_parameters, initkey)
        opt_state = opt_init(init_params)
        recorder.dump_global_header({ "experiment_id": eid, **config })
        max_iter = float(os.environ.get("MAX_ITER", jnp.inf))

        for iteration in range(int(training_config["num_iterations"])):
            if iteration > max_iter:
                print()
                sys.stderr.write(f"WARNING: Aborting after {iteration-1} iterations (MAX_ITER variable set)\n")
                sys.stderr.flush()
                exit(0)

            trainkey, train_subkey, loss_subkey, reg_loss_subkey = jax.random.split(trainkey, 4)
            data_subkey = keyring[ iteration % (n // batch_size) ]
            x, y = generator.sample_data(batch_size, data_subkey)

            reg_param = jnp.exp(- step_size * iteration / reg_tau)
            opt_state = update(iteration, opt_state, x, y, reg_param, train_subkey)
            train_loss = loss(get_params(opt_state), model_static_parameters, x, y, 0.0, loss_subkey)
            reg_train_loss = loss(get_params(opt_state), model_static_parameters, x, y, reg_param, reg_loss_subkey)

            if jnp.isnan(train_loss):
                sys.stderr.write("\nERROR: nan ecountered. Shutting down\n")
                sys.stderr.flush()
                exit(1)

            if recorder.should_print(iteration):
                nel = math.ceil(math.log10(training_config["num_iterations"]))
                msg = f"[{eid}] I:{iteration:0{nel}d} L:{train_loss:.12f} B:{reg_train_loss:+.5e} R:{reg_param:.2e}"
                sys.stderr.write("\r\033[0K" + msg)

            if recorder.should_dump(iteration):
                recorder.dump_state({
                    "iteration": iteration,
                    "time": time.time(),
                    "params": serialization.serialize_nested(get_params(opt_state)),
                    "train_loss": float(train_loss)
                    })
        print()

    except KeyboardInterrupt:
        recorder.teardown()
        sys.stderr.write(f"Training interrupted\n")
        sys.stderr.flush()
