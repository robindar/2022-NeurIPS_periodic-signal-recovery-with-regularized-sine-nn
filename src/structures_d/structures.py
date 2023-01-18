import jax
import jax.numpy as jnp
from typing import NamedTuple

class Model():
    class_dict = {}
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Model.class_dict[cls.__name__] = cls

    @staticmethod
    def regularizer(static_params, params, x, y, rng):
        return 0.0

class Generator():
    class_dict = {}
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Generator.class_dict[cls.__name__] = cls


class Sine(Generator):
    def __init__(self, randkey, freq, k=1, R=1, phase=0, noise_std = 0):
        self.freq = freq
        self.k = k
        self.R = R
        self.phase = phase
        self.noise_std = float(noise_std)

    def sample_data(self, n, randkey):
        x_key, y_key = jax.random.split(randkey, 2)
        x = jax.random.uniform(x_key, shape=(n,), minval=0, maxval=self.R)
        y = self.forward(x) + jax.random.normal(y_key, shape=(n,)) * self.noise_std
        return (x,y)

    def coordinates(self):
        return self.freq * (1 + jnp.arange(self.k)), jnp.abs((-.8) ** jnp.arange(self.k))

    def forward(self, x):
        y = jnp.zeros_like(x)
        for i in range(self.k):
            y += ((-.8) ** i) * jnp.cos((i+1) * self.freq * x + self.phase)
        return y


class MLP(Model):
    class Hyperparameters(NamedTuple):
        l : int
        m : int
        nn : str = "relu"

    @staticmethod
    def init_random_params(hyperparams: Hyperparameters, randkey):
        l, m = hyperparams.l, hyperparams.m
        keys = jax.random.split(randkey, l)
        layer_sizes = [ (1,m) ] + [ (m, m) ] * (l-2) + [ (m,1) ]
        layers = []
        for i in range(hyperparams.l):
            w_key, b_key = jax.random.split(keys[i], 2)
            w = jax.random.normal(w_key, shape=layer_sizes[i]) / jnp.sqrt(m)
            b = jax.random.normal(b_key, shape=(layer_sizes[i][-1],))
            layers.append([w,b])
        return layers

    @staticmethod
    def forward(hyperparams : Hyperparameters, params, x, rng):
        nn = {
                "relu": jax.nn.relu,
                "tanh": jax.nn.tanh,
                "snake": lambda x: x + (1./27.5) * jnp.sin(27.5 * x) ** 2,
                }[hyperparams.nn]
        z = x[None]
        for w, b in params[:-1]:
            z = nn(z @ w + b)
        w, b = params[-1]
        return z @ w + b

    @classmethod
    def neglog_likelihood(cls, hyperparams : Hyperparameters, params, x, y, rng):
        y_hat = cls.forward(hyperparams, params, x, rng)
        return jnp.square(y - y_hat)


class SineD(Model):
    class Hyperparameters(NamedTuple):
        m : int
        delta : float = 1.0
        alpha : float = None
        lam : float = None
        reg : str = None
        rand_init : bool = True

    @staticmethod
    def init_random_params(hyperparams: Hyperparameters, randkey):
        m = hyperparams.m
        w_key, p_key, a_key = jax.random.split(randkey, 3)
        if hyperparams.rand_init:
            w = jax.random.normal(w_key, shape=(m,))
            a = jax.random.normal(p_key, shape=(m,)) / jnp.sqrt(m)
            b = jax.random.normal(a_key, shape=(m,)) / jnp.sqrt(m)
        else:
            w = jnp.arange(m) * hyperparams.delta
            a = jnp.zeros(m)
            b = jnp.zeros(m)
        return (w, a, b)

    @staticmethod
    def forward(hyperparams : Hyperparameters, params, x, rng):
        w, a, b = params
        return jnp.dot(jnp.sin(w * x), a) + jnp.dot(jnp.cos(w * x), b)

    @classmethod
    def neglog_likelihood(cls, hyperparams : Hyperparameters, params, x, y, rng):
        y_hat = cls.forward(hyperparams, params, x, rng)
        return jnp.square(y - y_hat)

    @classmethod
    def regularizer(cls, hyperparams, params, x, y, rng):
        if hyperparams.lam is None:
            return 0.0
        w, a, b = params
        if hyperparams.reg == "l1":
            return hyperparams.lam * jnp.sum(jnp.sqrt(jnp.square(a) + jnp.square(b) + 1e-80))
        elif hyperparams.reg == "exp":
            return - hyperparams.lam * jnp.sum(jnp.exp(- jnp.sqrt(jnp.square(a) + jnp.square(b) + 1e-80)))
