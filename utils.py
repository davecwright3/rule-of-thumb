"""Utilities for rule-of-thumb and Neff calculations."""

import jax

jax.config.update("jax_enable_x64", True)
import pathlib
from collections.abc import Callable
from typing import Sequence

import jax.numpy as jnp
import matplotlib.pyplot as plt
import netket
from jax.tree_util import Partial
from matplotlib import cm
from matplotlib.tri import Triangulation
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from scipy.stats import qmc


def wrapper(
    function: Callable, domain: jnp.ndarray, func_args: Sequence
) -> jnp.ndarray:
    """
    Wrapper function to apply the given function to the domain and return the maximum along the first axis.

    Parameters
    ----------
    function : Callable
        The function to be applied to the domain.
    domain : jnp.ndarray
        The input domain to be passed to the function.
    func_args : Sequence
        Additional arguments to be passed to the function.

    Returns
    -------
    jnp.ndarray
        The maximum value of the function output along the first axis.
    """
    data = function(domain, *func_args)
    return jnp.max(data, axis=0)


def create_vmap_function(
    function: Callable, domain: jnp.ndarray, chunk_size: float = jnp.inf
) -> Callable:
    """
    Create a vectorized and JIT-compiled version of the given function using netket.jax.vmap_chunked.

    Parameters
    ----------
    function : Callable
        The function to be vectorized and JIT-compiled.
    domain : jnp.ndarray
        The input domain to be passed to the function.
    chunk_size : float, optional
        The chunk size for vectorization. Default is jnp.inf (no chunking).

    Returns
    -------
    Callable
        The vectorized and JIT-compiled version of the input function.
    """
    return jax.jit(
        netket.jax.vmap_chunked(
            Partial(wrapper, function, domain), in_axes=0, chunk_size=chunk_size
        )
    )


def create_sampler(d: int = 3) -> qmc.LatinHypercube:
    """
    Create a Latin Hypercube sampler with the specified dimension.

    Parameters
    ----------
    d : int, optional
        The dimension of the sampler. Default is 3.

    Returns
    -------
    qmc.LatinHypercube
        The Latin Hypercube sampler instance.
    """
    return qmc.LatinHypercube(d=d, strength=2)


def get_samples(
    sampler: qmc.LatinHypercube, n: int, l_bounds: Sequence, u_bounds: Sequence
) -> jnp.ndarray:
    """
    Generate samples from the given Latin Hypercube sampler and scale them to the specified bounds.

    Parameters
    ----------
    sampler : qmc.LatinHypercube
        The Latin Hypercube sampler instance.
    n : int
        The number of samples to generate.
    l_bounds : Sequence
        The lower bounds for scaling the samples.
    u_bounds : Sequence
        The upper bounds for scaling the samples.

    Returns
    -------
    jnp.ndarray
        The generated and scaled samples.
    """
    sample = sampler.random(n=n)
    return qmc.scale(sample, l_bounds, u_bounds)


def integrate(
    integrator: Callable,
    function: Callable,
    domain: jnp.ndarray,
    func_args: Sequence,
    **kwargs,
) -> float:
    """
    Integrate a function over the given ln(domain) using a specified integrator.

    Parameters
    ----------
    integrator : Callable
        The integration function to be used (e.g., quadax.scipy.quad).
    function : Callable
        The function to be integrated.
    domain : jnp.ndarray
        The domain over which to integrate the function.
    func_args : Sequence
        Additional arguments to be passed to the function.
    **kwargs
        Additional keyword arguments to be passed to the integrator.

    Returns
    -------
    float
        The integral of the function over the given domain.
    """
    integrand = function(domain, *func_args) / domain
    return integrator(integrand, **kwargs)


def create_vmap_integrator(
    integrator: Callable,
    function: Callable,
    domain: jnp.ndarray,
    chunk_size: float = jnp.inf,
    integrator_kwargs = {},
) -> Callable:
    """
    Create a vectorized and JIT-compiled version of the integrate function using netket.jax.vmap_chunked.

    Parameters
    ----------
    integrator : Callable
        The integration function to be used (e.g., quadax.scipy.quad).
    function : Callable
        The function to be integrated.
    domain : jnp.ndarray
        The domain over which to integrate the function.
    chunk_size : float, optional
        The chunk size for vectorization. Default is jnp.inf (no chunking).

    Returns
    -------
    Callable
        The vectorized and JIT-compiled version of the integrate function.
    """
    return jax.jit(
        netket.jax.vmap_chunked(
            Partial(integrate, integrator, function, domain, **integrator_kwargs),
            in_axes=0,
            chunk_size=chunk_size,
        )
    )


def plot_peak_omega_gw_hist(res, model_name, show_rot=True, is_int=False, labels=None, neff = None, save=True):
    """
    Plot the peak gravitational wave energy density spectrum and save the figure.

    Parameters
    ----------
    res : jax.numpy.ndarray
        The array containing the gravitational wave energy density values.
    model_name : str
        The name of the model to be used in the plot title and file name.

    Returns
    -------
    None

    Notes
    -----
    This function creates a histogram plot of the logarithm (base 10) of the
    non-zero values in the `res` array, divided by (0.674 ** 2). The plot also
    includes vertical lines representing "Optimistic", "Realistic", and
    "Pessimistic" values. The plot is saved as a PNG file in the "figs"
    directory within the current working directory, with the filename
    constructed using the `model_name`.
    """
    if isinstance(res, list):
        log10 = [jnp.log10(result[jnp.nonzero(result)] / (0.674**2)) for result in res]
        mplib_array = [element.__array__() for element in log10]
    else:
        log10 = jnp.log10(res[jnp.nonzero(res)] / (0.674**2))
        mplib_array = log10.__array__()

    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    _ = ax.hist(mplib_array, bins=50, density=True, histtype="step", label=labels, linewidth=2)


    fig_dir = pathlib.Path().cwd() / "figs"
    fig_dir.mkdir(exist_ok=True)
    if is_int:
        #ax.set_title(f"{model_name} Integrated $\Omega_{{GW}}$")
        ax.axvline(x=jnp.log10(5.6e-6 * neff * 0.674**-2), label=r"$N_{eff}$ Bound", c="r", linestyle="dashed")
        #ax.legend(loc="best")
        ax.set_xlabel(r"$\log_{10}\Omega_{GW}$")
        if save:
            fig.savefig(
                fig_dir / f"{model_name.replace(' ','-').lower()}-int-omega-hist.png",
                bbox_inches="tight",
            )
    else:
        #ax.set_title(f"{model_name} Peak $\Omega_{{GW}}(f)$")

        ax.axvline(x=jnp.log10(4.97e-10), label="Optimistic", color="C2", linestyle="dotted", linewidth=2)
        ax.axvline(x=jnp.log10(1.49e-12), label="Realistic", color="C1", linestyle="dashdot", linewidth=2)
        ax.axvline(x=jnp.log10(9.93e-15), label="Pessimistic", color="C3", linestyle="dashed", linewidth=2)
        #ax.legend(loc="best")
        ax.set_xlabel(r"$\log_{10}\Omega_{GW}(f)$")
        if save:
            fig.savefig(
                fig_dir / f"{model_name.replace(' ','-').lower()}-peak-omega-hist.png",
                bbox_inches="tight",
            )
    return fig, ax


def plot_peak_omega_gw_contour(res, model_name, samples):
    """
    Plot the peak gravitational wave energy density spectrum using a contour plot.

    Parameters
    ----------
    samples : jax.numpy.ndarray
        The array containing the sample points (x, y).
    res : jax.numpy.ndarray
        The array containing the gravitational wave energy density values.
    model_name : str
        The name of the model to be used in the plot title.

    Returns
    -------
    None

    Notes
    -----
    This function creates a contour plot of the logarithm (base 10) of the
    gravitational wave energy density values, using a triangulation of the
    sample points. The plot also includes a colorbar indicating the peak
    log10(Omega_GW) values. The function assumes that the input arrays
    `samples` and `res` have the same shape.
    """

    fig, ax = plt.subplots()

    x, y = jnp.hsplit(samples, 2)
    triangulation = Triangulation(x.squeeze(), y.squeeze())

    res_filt = res.at[res < 1e-20].set(1e-15)

    contour = ax.tricontourf(
        triangulation,
        jnp.log10(res_filt),
        cmap=cm.viridis,
        levels=[-15, -12, -10, -5, -2, res.max()],
    )
    fig.colorbar(contour, label=r"Peak $\log_{10}\,\Omega_{GW}$")

    ax.set_title(f"{model_name} Peak $\Omega_{{GW}}$")
    ax.set_ylabel("log10_A")
    ax.set_xlabel("log10_f_peak")
    fig.savefig(
        fig_dir / f"{model_name}-peak-omega-2d-contour.png", bbox_inches="tight"
    )
