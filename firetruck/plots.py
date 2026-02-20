from operator import attrgetter

import jax.numpy as jnp
from jax.scipy.stats import gaussian_kde
from numpyro.diagnostics import effective_sample_size, hpdi
from numpyro.infer import MCMC


def get_plotly():
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        import plotly.subplots as subplots
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "You can only use plots in firetruck once plotly is installed"
        ) from e
    return px, go, subplots


def get_rvs(mcmc):
    # Exclude deterministic sites
    sites = mcmc._states[mcmc._sample_field]
    if isinstance(sites, dict):
        state_sample_field = attrgetter(mcmc._sample_field)(mcmc._last_state)
        if isinstance(state_sample_field, dict):
            sites = {
                k: v
                for k, v in mcmc._states[mcmc._sample_field].items()
                if k in state_sample_field
            }
    rv_sites = list(sites.keys())
    return rv_sites


def get_divergences(mcmc):
    extra_fields = mcmc.get_extra_fields(group_by_chain=True)
    if "diverging" in extra_fields:
        return extra_fields["diverging"]
    else:
        return None


def plot_trace(mcmc: MCMC, variables: list[str] | None = None):
    px, go, subplots = get_plotly()
    samples = mcmc.get_samples(group_by_chain=True)
    if variables is None:
        variables = get_rvs(mcmc)
    subplot_titles = []
    for v in variables:
        subplot_titles.extend([v, v])
    divergences = get_divergences(mcmc)
    fig = subplots.make_subplots(
        subplot_titles=subplot_titles,
        rows=len(variables),
        cols=2,
        vertical_spacing=0.1,
        horizontal_spacing=0.01,
        column_widths=[0.3, 0.7],
    )
    dashes = [None, "dash", "dot", "dashdot"]
    colors = px.colors.qualitative.Dark24
    for i_variable, var_name in enumerate(variables):
        var_samples = samples[var_name]
        for i_chain, chain in enumerate(var_samples):
            chain_divergences = divergences[i_chain]
            div_ind, *_ = jnp.where(chain_divergences)
            var_range = jnp.max(var_samples) - jnp.min(var_samples)
            fig.add_trace(
                go.Scattergl(
                    name="Divergences",
                    y=[jnp.min(var_samples) - var_range * 0.1] * len(div_ind),
                    x=div_ind,
                    marker=dict(color="black", symbol="line-ns-open", size=12),
                    mode="markers",
                    showlegend=False,
                ),
                col=2,
                row=i_variable + 1,
            )
            chain = jnp.reshape(chain, (-1, chain.shape[-1]))
            dash = dashes[i_chain % len(dashes)]
            for i_level, level in enumerate(chain):
                dens = gaussian_kde(level)
                grid = jnp.linspace(jnp.min(level), jnp.max(level), 50)
                color = colors[i_level % len(colors)]
                fig = fig.add_trace(
                    go.Scattergl(
                        x=jnp.arange(len(level)),
                        y=level,
                        line=dict(color=color, dash=dash),
                        showlegend=False,
                    ),
                    col=2,
                    row=i_variable + 1,
                )
                fig = fig.add_trace(
                    go.Scattergl(
                        x=grid,
                        y=dens.pdf(grid),
                        line=dict(color=color, dash=dash),
                        showlegend=False,
                    ),
                    col=1,
                    row=i_variable + 1,
                )
                fig = fig.add_trace(
                    go.Scattergl(
                        name="Divergences",
                        y=[0] * len(div_ind),
                        x=level[div_ind],
                        marker=dict(color="black", symbol="line-ns-open", size=12),
                        mode="markers",
                        showlegend=False,
                    ),
                    col=1,
                    row=i_variable + 1,
                )
    fig = fig.update_layout(template="plotly_white", margin=dict(t=20, b=0, l=0, r=0))
    return fig


def plot_forest(
    samples_or_mcmc: MCMC | dict, prob: float = 0.94, variables: list[str] | None = None
):
    px, go, subplots = get_plotly()
    if isinstance(samples_or_mcmc, MCMC):
        samples = samples_or_mcmc.get_samples()
        mcmc = samples_or_mcmc
    else:
        samples = samples_or_mcmc
        mcmc = None
    if variables is None:
        if mcmc is not None:
            variables = get_rvs(mcmc)
        else:
            variables = samples.keys()
    colors = px.colors.qualitative.Dark24
    fig = go.Figure()
    for i_variable, var_name in enumerate(variables):
        var_samples = samples[var_name]
        var_samples = jnp.reshape(var_samples, (-1, var_samples.shape[-1]))
        for i_level, level in enumerate(var_samples):
            lower, upper = hpdi(level, prob=prob)
            center = jnp.median(level)
            name = var_name
            if var_samples.shape[0] != 1:
                name += f"[{i_level}]"
            fig.add_scatter(
                y0=var_name,
                x=[center],
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[upper - center],
                    arrayminus=[center - lower],
                    width=0,
                    thickness=2.5,
                ),
                marker=dict(color=colors[i_level % len(colors)]),
                name=var_name,
                showlegend=False,
                mode="markers",
            )
            lower, upper = hpdi(level, prob=0.5)
            fig.add_scatter(
                y0=var_name,
                x=[center],
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[upper - center],
                    arrayminus=[center - lower],
                    width=0,
                    thickness=5,
                ),
                marker=dict(color=colors[i_level % len(colors)], size=12),
                name=var_name,
                showlegend=False,
                mode="markers",
            )
    fig = fig.update_layout(template="plotly_white", margin=dict(t=20, b=0, l=0, r=0))
    return fig


def plot_ess(mcmc: MCMC | dict, variables: list[str] | None = None):
    px, go, subplots = get_plotly()
    samples = mcmc.get_samples(group_by_chain=True)
    if variables is None:
        variables = get_rvs(mcmc)
    colors = px.colors.qualitative.Dark24
    fig = go.Figure()
    samples = dict(samples)
    for key in samples:
        vals = samples[key]
        vals = jnp.reshape(vals, (vals.shape[0], -1, vals.shape[-1]))
        vals = jnp.transpose(vals, (1, 0, 2))
        samples[key] = vals
    i_color = 0
    for i_variable, var_name in enumerate(variables):
        var_samples = samples[var_name]
        for i_level, level in enumerate(var_samples):
            name = var_name
            if var_samples.shape[0] != 1:
                name += f"[{i_level}]"
            n_draws = level.shape[-1]
            grid = list(range(min(50, n_draws), n_draws, (n_draws - 20) // 20)) + [
                n_draws
            ]
            ess = []
            for upper in grid:
                ess.append(effective_sample_size(level[:, :upper]))
            fig.add_scatter(
                x=grid,
                y=ess,
                mode="lines+markers",
                marker=dict(color=colors[i_color % len(colors)]),
                name=name,
                showlegend=True,
            )
            i_color += 1
    fig = fig.update_layout(
        template="plotly_white",
        margin=dict(t=20, b=20, l=20, r=20),
        xaxis_title="Number of draws from posterior",
        yaxis_title="Effective Sample Size",
    )
    return fig


def plot_predictive_check(
    prior_or_posterior_predictive: dict,
    obs=None,
    obs_name: str = "obs",
    n_grid_points: int = 1000,
):
    px, go, subplots = get_plotly()
    samples = prior_or_posterior_predictive[obs_name]
    lowest = jnp.min(samples)
    highest = jnp.max(samples)
    if obs is not None:
        lowest = min(jnp.min(obs), lowest)
        highest = max(jnp.max(obs), highest)
    grid = jnp.linspace(lowest, highest, n_grid_points)
    fig = go.Figure()
    for i_draw, draw in enumerate(samples):
        draw = jnp.ravel(draw)
        y_draw = gaussian_kde(draw).pdf(grid)
        fig.add_trace(
            go.Scattergl(
                x=grid,
                y=y_draw,
                line=dict(color="#2E91E5"),
                opacity=0.2,
                name="Predictive",
                showlegend=i_draw == 0,
            )
        )
    fig.add_trace(
        go.Scattergl(
            x=grid,
            y=gaussian_kde(jnp.ravel(samples)).pdf(grid),
            line=dict(color="#1616A7", dash="dash", width=3),
            name="Predictive mean",
            showlegend=True,
        )
    )
    if obs is not None:
        fig.add_trace(
            go.Scattergl(
                x=grid,
                y=gaussian_kde(obs).pdf(grid),
                line=dict(color="black", width=3),
                name="Observed",
                showlegend=True,
            )
        )
    fig = fig.update_layout(
        template="plotly_white",
        margin=dict(t=20, b=20, l=20, r=20),
        xaxis_title="Outcome",
        yaxis_title="Density",
    )
    return fig
