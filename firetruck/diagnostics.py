import warnings
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pandas as pd
from jax.nn import logsumexp
from numpyro.infer.importance import _fit_generalized_pareto
from tqdm import tqdm


@dataclass
class ELPDResults:
    elpd_loo: float
    elpd_loo_se: float
    elpd_loo_i: np.ndarray
    ks: np.ndarray
    k_target: float

    def __repr__(self):
        return str(self)

    def __str__(self):
        res = ""
        res += f"ELPD_loo = {self.elpd_loo:.2f}±{self.elpd_loo_se:.2f}\n\n"
        res += "Pareto K distribution:\n---------------------\n"
        histogram_width = 10
        okay = min(self.k_target, 0.5)
        percent_okay = jnp.sum(self.ks < okay) / len(self.ks)
        n_bars = int(percent_okay * histogram_width)
        res += (
            "█" * n_bars
            + " " * (histogram_width - n_bars + 1)
            + str(int(100 * percent_okay))
            + f"% Okay (k<{okay:.2f})"
            + "\n"
        )
        if self.k_target > 0.5:
            potentially_problematic = jnp.sum(
                (self.ks > 0.5) & (self.ks < self.k_target)
            ) / len(self.ks)
            n_bars = int(potentially_problematic * histogram_width)
            res += (
                "█" * n_bars
                + " " * (histogram_width - n_bars + 1)
                + str(int(100 * potentially_problematic))
                + f"% Potentially problematic (0.5<k<{self.k_target:.2f})"
                + "\n"
            )
        problematic = jnp.sum(self.ks >= self.k_target) / len(self.ks)
        n_bars = int(problematic * histogram_width)
        res += (
            "█" * n_bars
            + " " * (histogram_width - n_bars + 1)
            + str(int(100 * problematic))
            + f"% Problematic ({self.k_target:.2f}<k)"
            + "\n"
        )
        return res


def genpareto_icdf(x, k, sigma):
    if k != 0:
        return sigma / k * (jnp.power((1 - x), -k) - 1)
    else:
        return sigma * jnp.log(1 - x)


def _pareto_smoothing(_loglik):
    _lw = 1 - _loglik
    _lw = jnp.ravel(_lw)
    _lw -= _lw.max()
    lw_indices = jnp.argsort(_lw)
    _lw = _lw[lw_indices]
    S = len(_lw)
    M = int(jnp.ceil(jnp.minimum(0.2 * S, 3 * jnp.sqrt(S))))
    cutoff_ind = -(M + 1)
    lw_cutoff = jnp.maximum(jnp.log(jnp.finfo(float).tiny), _lw[cutoff_ind])
    lw_tail = _lw[_lw > lw_cutoff]
    if len(lw_tail) < 5:
        warnings.warn(
            "Not enough tail samples for reliable PSIS diagnostic.",
        )
        return 1 - _loglik, jnp.nan
    tail = jnp.exp(lw_tail) - jnp.exp(lw_cutoff)
    k, sigma = _fit_generalized_pareto(tail)
    z = jnp.arange(len(lw_tail)) + 1
    smoothed = jnp.log(genpareto_icdf((z - 1 / 2) / len(lw_tail), k, sigma))
    log_weights = jnp.ravel(1 - _loglik)
    log_weights = log_weights.at[lw_indices[_lw > lw_cutoff]].set(smoothed)
    log_weights = log_weights.reshape(_loglik.shape)
    return log_weights, k


def psis_loo(loglik: dict, show_progress_bar=True) -> float:
    """Compute PSIS k-hat from an array of raw log importance weights."""
    loglik = {
        site: lik.reshape(lik.shape[0], -1) for site, lik in loglik.items()
    }
    loglik = jnp.concatenate(jax.tree.leaves(loglik))
    log_weights = []
    ks = []
    for _loglik in tqdm(
        loglik.T,
        "Pareto smoothing weights for all datapoints",
        disable=not show_progress_bar,
    ):
        lw, k = _pareto_smoothing(_loglik)
        log_weights.append(lw)
        ks.append(k)
    ks = jnp.array(ks)
    log_weights = jnp.stack(log_weights).T
    k_target = jnp.minimum(1 - 1 / jnp.log10(loglik.shape[0]), 0.7)
    if jnp.any(ks > 0.5) and ~jnp.any(ks < k_target):
        warnings.warn(
            "Pareto k>0.5 for some datapoints, the estimated ELPD might be biased."
        )
    elif jnp.any(ks > k_target):
        warnings.warn(
            "Pareto k>0.7 for some datapoints, the estimated ELPD should not be trusted. We recommend you resort to K-fold cross validation instead."
        )
    elpd_loo_i = logsumexp(log_weights + loglik, axis=0) - logsumexp(
        log_weights, axis=0
    )
    elpd_loo = jnp.nansum(elpd_loo_i)
    n = len(elpd_loo_i)
    elpd_loo_se = jnp.sqrt(jnp.nansum(jnp.square(elpd_loo_i - elpd_loo / n)))
    return ELPDResults(
        elpd_loo_i=np.array(elpd_loo_i),
        elpd_loo=float(elpd_loo),
        elpd_loo_se=float(elpd_loo_se),
        ks=np.array(ks),
        k_target=np.array(k_target),
    )


def pseudo_bma_plus(rng_key, elpds: dict[str, ELPDResults], num_samples=1000):
    "Pseudo-BMA weight calculation with Bayesian Bootstrap."
    model_names = []
    _elpd = []
    for model_name, res in elpds.items():
        model_names.append(model_name)
        _elpd.append(res)
    z = jnp.stack([res.elpd_loo_i for res in _elpd])
    n = z.shape[1]
    bootstrap_distribution = dist.Dirichlet(jnp.ones(n))
    keys = jax.random.split(rng_key, num=num_samples)

    def get_w_k_b(key):
        a = bootstrap_distribution.sample(key)
        z_k_b = jnp.sum(a[None, :] * z, axis=1)
        log_w_k_b = (n * z_k_b) - logsumexp(n * z_k_b)
        return log_w_k_b

    weight_distribution = jnp.exp(jax.vmap(get_w_k_b)(keys))
    return dict(zip(model_names, np.array(weight_distribution.T)))


def compare(elpds: dict[str, ELPDResults], rng_key=None, bb_num_samples=1000):
    if rng_key is None:
        rng_key = jax.random.key(0)
    weight_distributions = pseudo_bma_plus(
        rng_key=rng_key, elpds=elpds, num_samples=bb_num_samples
    )
    model_names = []
    _elpd = []
    for model_name, res in elpds.items():
        model_names.append(model_name)
        _elpd.append(res)
    z = jnp.stack([res.elpd_loo_i for res in _elpd])
    mean_estimates = jnp.mean(z, axis=1)
    order = jnp.argsort(-mean_estimates)
    i_reference = order[0]
    diff_records = []
    for rank, i_model in enumerate(order):
        model_name = model_names[i_model]
        if i_reference == i_model:
            diff_records.append(
                dict(
                    model_name=model_name,
                    rank=rank,
                    w=float(jnp.mean(weight_distributions[model_name])),
                    elpd_diff=0,
                    se_diff=np.nan,
                    p_worse=np.nan,
                    elpd_loo=elpds[model_name].elpd_loo,
                    elpd_loo_se=elpds[model_name].elpd_loo_se,
                )
            )
        else:
            diff = z[i_reference] - z[i_model]
            mean_diff = jnp.nanmean(diff)
            se_diff = jax.scipy.stats.sem(diff, nan_policy="omit")
            p_worse = 1 - dist.Normal(mean_diff, se_diff).cdf(0)
            diff_records.append(
                dict(
                    model_name=model_name,
                    rank=rank,
                    w=float(jnp.nanmean(weight_distributions[model_name])),
                    elpd_diff=float(mean_diff),
                    se_diff=float(se_diff),
                    p_worse=float(p_worse),
                    elpd_loo=elpds[model_name].elpd_loo,
                    elpd_loo_se=elpds[model_name].elpd_loo_se,
                )
            )
    df = (
        pd.DataFrame.from_records(diff_records)
        .set_index("model_name")
        .round(3)
    )
    return df
