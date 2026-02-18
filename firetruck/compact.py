from functools import partial

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import seed, trace
from numpyro.infer import (MCMC, NUTS, SVI, DiscreteHMCGibbs, Predictive,
                           TraceGraph_ELBO, TraceMeanField_ELBO)
from numpyro.infer.autoguide import AutoNormal
from numpyro.primitives import Messenger


class compact(Messenger):
    def __init__(self, fn, top_level=True):
        self.fn = fn
        self.top_level = top_level

    def __setattr__(self, name, val):
        if isinstance(val, dist.Distribution):
            super().__setattr__(name, numpyro.sample(name, val))
        elif isinstance(val, jax.Array):
            super().__setattr__(name, numpyro.deterministic(name, val))
        else:
            super().__setattr__(name, val)

    def __call__(self, *args, **kwargs):
        if self.top_level:
            args = (self, *args)
        obs = super().__call__(*args, **kwargs)
        if self.top_level:
            self.obs = obs
        return obs

    def condition_on(self, obs):
        return conditioned(self, obs=obs)

    def add_input(self, *args, **kwargs):
        return compact(partial(self, *args, **kwargs), top_level=False)

    def sample_predictive(
        self,
        rng_key,
        *model_args,
        num_samples: int = 1000,
        posterior_samples: dict = None,
        params: dict = None,
        **model_kwargs,
    ):
        pred = Predictive(
            self,
            num_samples=num_samples,
            posterior_samples=posterior_samples,
            params=params,
        )
        return pred(rng_key, *model_args, **model_kwargs)


class conditioned(compact):
    def __init__(
        self,
        fn,
        obs,
    ) -> None:
        self.obs = jnp.array(obs)
        super(conditioned, self).__init__(fn, top_level=False)

    def process_message(self, msg) -> None:
        if msg["name"] != "obs":
            return
        msg["value"] = self.obs
        msg["is_observed"] = True

    def _has_latent_discrete(self, *model_args, **model_kwargs) -> bool:
        # We use init strategy to get around ImproperUniform which does not have
        # sample method.
        _prototype_trace = trace(seed(self, jax.random.key(0))).get_trace(
            *model_args, **model_kwargs
        )
        _gibbs_sites = [
            name
            for name, site in _prototype_trace.items()
            if site["type"] == "sample"
            and site["fn"].has_enumerate_support
            and not site["is_observed"]
            and site["infer"].get("enumerate", "") != "parallel"
        ]
        return len(_gibbs_sites) != 0

    def sample_posterior(
        self,
        rng_key,
        *model_args,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        num_chains: int = 4,
        target_accept_prob: float = 0.9,
        dense_mass: bool = False,
        max_tree_depth: int = 10,
        **model_kwargs,
    ) -> dict:
        rng_key = jax.random.PRNGKey(0)
        kernel = NUTS(
            self,
            dense_mass=dense_mass,
            max_tree_depth=max_tree_depth,
            target_accept_prob=target_accept_prob,
        )
        if self._has_latent_discrete(*model_args, **model_kwargs):
            kernel = DiscreteHMCGibbs(kernel)
        mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_warmup,
            num_chains=num_chains,
        )
        mcmc.run(rng_key, *model_args, **model_kwargs)
        mcmc.print_summary()
        return mcmc

    def meanfield_vi(
        self,
        rng_key,
        *model_args,
        num_steps: int = 10_000,
        step_size: float = 1e-4,
        **model_kwargs,
    ):
        guide = AutoNormal(model=self)
        optimizer = numpyro.optim.Adam(
            step_size=step_size,
        )
        if self._has_latent_discrete(*model_args, **model_kwargs):
            loss = TraceGraph_ELBO()
        else:
            loss = TraceMeanField_ELBO()
        svi = SVI(self, guide, optimizer, loss=loss)
        svi_result = svi.run(
            rng_key, num_steps, *model_args, stable_update=True, **model_kwargs
        )
        return svi_result
