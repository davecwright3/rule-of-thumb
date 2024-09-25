#!/usr/bin/env python
# coding: utf-8

# In[1]:


import jax
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)
import pathlib
import pickle
from itertools import product

import jax.numpy as jnp
import quadax
import utils


# In[2]:


from models import (
    dw_ds,
    dw_sm,
    igw,
    pt_bubble,
    pt_sound,
    sigw_box,
    sigw_delta,
    sigw_gauss,
)

model_dict = {
    "PT Sound": {
        "model": pt_sound,
        "l_bounds": jnp.array([-2, -4, -3, 3, 2, 3]),
        "u_bounds": jnp.array([1, 4, 0, 5, 4, 5]),
        "l_bounds_post": jnp.array([-0.18, -2.06, -1.00, 3, 2, 3.85]),
        "u_bounds_post": jnp.array([0.70, -1.30, -0.38, 3.74, 3.17, 5]),
        "map": jnp.array([-0.01, -1.75, -0.82, 3, 2, 5]),
        "model_params": ["log10_alpha", "log10_T_star", "log10_H_R", "a", "b", "c"],
    },
    "PT Bubble": {
        "model": pt_bubble,
        "l_bounds": jnp.array([-2, -4, -3, 1, 1, 1]),
        "u_bounds": jnp.array([1, 4, 0, 3, 3, 3]),
        "l_bounds_post": jnp.array([0.03, -1.33, -0.56, 1.49, 1, 1.69]),
        "u_bounds_post": jnp.array([1, -0.39, 0, 2.54, 2.32, 3]),
        "map": jnp.array([1, -0.90, 0, 1.97, 1, 3]),
        "model_params": ["log10_alpha", "log10_T_star", "log10_H_R", "a", "b", "c"],
    },
    "DW Dark Radiation": {
        "model": dw_ds,
        "l_bounds": jnp.array([-3, -4, 0.5, 0.3]),
        "u_bounds": jnp.array([jnp.log10(0.39), 4, 1, 3]),
        "l_bounds_post": jnp.array([-0.49, -1.10, 0.5, 1.62]),
        "u_bounds_post": jnp.array([-0.41, -0.82, 0.97, 3]),
        "map": jnp.array([-0.41, -0.94, 0.5, 3]),
        "model_params": ["log10_N_eff", "log10_T_star", "b", "c"],
    },
    "DW SM Decay": {
        "model": dw_sm,
        "l_bounds": jnp.array([-3, -4, 0.5, 0.3]),
        "u_bounds": jnp.array([0, 4, 1, 3]),
        "l_bounds_post": jnp.array([-1.10, -0.96, 0.5, 1.72]),
        "u_bounds_post": jnp.array([-0.71, -0.56, 0.83, 3]),
        "map": jnp.array([-0.92, -0.79, 0.5, 3]),
        "model_params": ["log10_alpha", "log10_T_star", "b", "c"],
    },
    "SIGW Delta": {
        "model": sigw_delta,
        "l_bounds": jnp.array([-3, -11]),
        "u_bounds": jnp.array([1, -5]),
        "l_bounds_post": jnp.array([-1.00, -6.17]),
        "u_bounds_post": jnp.array([-0.01, -5]),
        "map": jnp.array([-0.14, -5]),
        "model_params": ["log10_A", "log10_f_peak"],
    },
}

n = 96721  # Number of draws, has to be prime
# n = 9
# Set frequency space. Most of the interpolated models in the 15yr only go up
# to 10**-5 Hz, be aware that this affects integrated energy densities (Neff),
# but it doesn't impact the rule of thumb calculations. Here I've omitted the models
# that are interpolated and don't have tabulated values about a frequency limit.
freqs = jnp.logspace(-12, 2, 1000)


# In[3]:


results = []
results_post = []

for _, model in enumerate(model_dict):
    print(model)

    sub_dict = model_dict[model]

    d = (
        len(sub_dict["l_bounds"])
        if (len(sub_dict["l_bounds"]) == len(sub_dict["u_bounds"]))
        else None
    )
    if not d:
        raise Exception(f"{len(sub_dict['l_bounds'])=} != {len(sub_dict['u_bounds'])=}")

    sampler = utils.create_sampler(d)
    samples = jnp.array(
        utils.get_samples(sampler, n, sub_dict["l_bounds"], sub_dict["u_bounds"])
    )

    func = utils.create_vmap_integrator(
        quadax.trapezoid,
        sub_dict["model"].spectrum,
        freqs,
        chunk_size=1000,
        integrator_kwargs={"x": freqs},
    )

    results.append(func(samples))

    samples = jnp.array(
        utils.get_samples(
            sampler, n, sub_dict["l_bounds_post"], sub_dict["u_bounds_post"]
        )
    )

    # Add samples at the edges.
    samples = jnp.vstack(
        (
            samples,
            jnp.array(
                list(
                    product(*zip(sub_dict["l_bounds_post"], sub_dict["u_bounds_post"]))
                )
            ),
        )
    )

    results_post.append(func(samples))


# In[4]:


plt.rcParams.update(
    {
        "axes.linewidth": 0.5,
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "cm",
        "font.size": 10,
    }
)
for i, model in enumerate(model_dict):
    fig, ax = utils.plot_peak_omega_gw_hist(
        results[i],
        model,
        labels=[f"Prior"],
        save=False,
        is_int=True,
        neff=(2.99 + 0.34) - 3.046,
    )

    func = utils.create_vmap_integrator(
        quadax.trapezoid,
        model_dict[model]["model"].spectrum,
        freqs,
        chunk_size=1000,
        integrator_kwargs={"x": freqs},
    )

    # Find the min and max over the 68% CI
    min_peak, max_peak = jnp.log10(
        jnp.array([results_post[i].min(), results_post[i].max()])
    )

    # Plot it and the MAP value peak
    ax.axvspan(min_peak, max_peak, alpha=0.2, color="grey", label="NG15 68\% CI")

    # We already have the function, but it expects two dimensions. Just calculate
    # twice and toss the extra
    ax.axvline(
        jnp.log10(func(model_dict[model]["map"][None, ...].repeat(2, axis=0))[0]),
        color="black",
        alpha=0.5,
        label="NG15 MAP",
        linewidth=2,
    )
    ax.legend(bbox_to_anchor=(0.5, 1.30), loc="upper center", ncol=2)
    fig.tight_layout()

    fig_dir = pathlib.Path().cwd() / "figs"
    fig_dir.mkdir(exist_ok=True)
    fig.savefig(
        fig_dir / f"{model.replace(' ','-').lower()}-int-omega-hist.pdf",
        bbox_inches="tight",
    )

