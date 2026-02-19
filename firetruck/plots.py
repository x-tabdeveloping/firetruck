from operator import attrgetter

import jax.numpy as jnp
from jax.scipy.stats import gaussian_kde
from numpyro.infer import MCMC


def get_plotly():
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        import plotly.subplots as subplots
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "You can only use plots in NumFire once plotly is installed"
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
        return jnp.max(extra_fields["diverging"], axis=0)
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
        div_ind, *_ = jnp.where(divergences)
        var_range = jnp.max(var_samples) - jnp.min(var_samples)
        fig.add_scatter(
            name="Divergences",
            y=[jnp.min(var_samples) - var_range * 0.1] * len(div_ind),
            x=div_ind,
            marker=dict(color="black", symbol="line-ns-open", size=12),
            mode="markers",
            col=2,
            showlegend=False,
            row=i_variable + 1,
        )
        for i_chain, chain in enumerate(var_samples):
            chain = jnp.reshape(chain, (-1, chain.shape[-1]))
            dash = dashes[i_chain % len(dashes)]
            for i_level, level in enumerate(chain):
                dens = gaussian_kde(level)
                grid = jnp.linspace(jnp.min(level), jnp.max(level), 50)
                color = colors[i_level % len(colors)]
                fig = fig.add_scatter(
                    x=jnp.arange(len(level)),
                    y=level,
                    line=dict(color=color, dash=dash),
                    col=2,
                    row=i_variable + 1,
                    showlegend=False,
                )
                fig = fig.add_scatter(
                    x=grid,
                    y=dens.pdf(grid),
                    row=i_variable + 1,
                    col=1,
                    line=dict(color=color, dash=dash),
                    showlegend=False,
                )
    fig = fig.update_layout(template="plotly_white", margin=dict(t=20, b=0, l=0, r=0))
    return fig
