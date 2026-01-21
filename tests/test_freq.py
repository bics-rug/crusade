import jax.numpy as jnp
import pytest

from crusafe.utils import frequency_bins_generator


@pytest.mark.parametrize("n", [1, 2, 8, 24])
@pytest.mark.parametrize("dist", ["linear", "log", "mel"])
def test_bins(n, dist):
    """Test values of bins and frequencies."""
    fmin = 20.0
    fmax = 20000.0
    frequencies, freq_bins = frequency_bins_generator(
        number_of_bins=n, freq_min=fmin, freq_max=fmax, freq_distribution=dist
    )
    assert freq_bins.shape == (n + 1,)
    assert frequencies.shape == (n,)
    assert jnp.isclose(freq_bins[0], fmin), (
        f"First bin {freq_bins[0]} != freq_min {fmin} for {dist}"
    )
    assert jnp.isclose(freq_bins[-1], fmax), (
        f"Last bin {freq_bins[-1]} != freq_max {fmax} for {dist}"
    )
    # Test that we dont have infinite or NaN values
    assert jnp.isfinite(freq_bins).all()
    assert jnp.isfinite(frequencies).all()
    assert jnp.all(jnp.diff(freq_bins) > 0)
    assert jnp.allclose(frequencies, (freq_bins[:-1] + freq_bins[1:]) / 2)


def test_lin_freq():
    """Test the bins of linear scale."""
    _, freq_bins = frequency_bins_generator(
        number_of_bins=8, freq_min=200, freq_max=2000, freq_distribution="linear"
    )
    expected_bins = jnp.linspace(200, 2000, 9)
    assert jnp.allclose(freq_bins, expected_bins)


def test_log_freq():
    """Test the bins of log scale."""
    _, freq_bins = frequency_bins_generator(
        number_of_bins=8, freq_min=200, freq_max=2000, freq_distribution="log"
    )
    expected_bins = jnp.logspace(jnp.log10(200), jnp.log10(2000), 9)
    assert jnp.allclose(freq_bins, expected_bins)


def test_mel_freq():
    """Test the bins of mel scale."""
    _, freq_bins = frequency_bins_generator(
        number_of_bins=8, freq_min=200, freq_max=2000, freq_distribution="mel"
    )
    mel_bins = jnp.linspace(
        2595 * jnp.log10(1 + 200.0 / 700), 2595 * jnp.log10(1 + 2000.0 / 700), 9
    )
    expected_bins = 700 * (10 ** (mel_bins / 2595) - 1)
    assert jnp.allclose(freq_bins, expected_bins)


@pytest.mark.parametrize("n", [2, 8, 24, 32])
@pytest.mark.parametrize("dist", ["linear", "log", "mel"])
def test_equal_min_max_returns_constant_bins(n, dist):
    """Verify that freq_min=freq_max all bins are equal."""
    f = 440.0
    freqs, bins = frequency_bins_generator(
        number_of_bins=n, freq_min=f, freq_max=f, freq_distribution=dist
    )
    assert jnp.allclose(bins, f)
    assert jnp.allclose(freqs, f)


def test_log_invalid_raises():
    """Verify that freq_min = 0 raise ValueError on log scale."""
    with pytest.raises(ValueError):
        frequency_bins_generator(
            number_of_bins=8, freq_min=0.0, freq_max=2000.0, freq_distribution="log"
        )


@pytest.mark.parametrize("dist", ["linear", "log", "mel"])
def test_freq_min_max(dist):
    """Verify that freq_min > freq_max raise ValueError."""
    with pytest.raises(ValueError):
        frequency_bins_generator(
            number_of_bins=8, freq_min=10.0, freq_max=2.0, freq_distribution=dist
        )


@pytest.mark.parametrize("dist", ["linear", "log", "mel"])
def test_min_bins(dist):
    """Verify that number of bins <= 0 raise ValueError."""
    with pytest.raises(ValueError):
        frequency_bins_generator(
            number_of_bins=0, freq_min=10.0, freq_max=2.0, freq_distribution=dist
        )
    with pytest.raises(ValueError):
        frequency_bins_generator(
            number_of_bins=-1, freq_min=10.0, freq_max=2.0, freq_distribution=dist
        )


@pytest.mark.parametrize("dist", ["linear", "log", "mel"])
def test_frequencies_within_range(dist):
    """Verify that all frequency centers are within [freq_min, freq_max]."""
    fmin, fmax = 50.0, 10000.0
    freqs, _ = frequency_bins_generator(
        number_of_bins=20, freq_min=fmin, freq_max=fmax, freq_distribution=dist
    )
    assert jnp.all(freqs >= fmin), f"Some frequencies < freq_min for {dist}"
    assert jnp.all(freqs <= fmax), f"Some frequencies > freq_max for {dist}"


@pytest.mark.parametrize("dist", ["linear", "log", "mel"])
def test_return_types_are_jax_arrays(dist):
    """Verify that both frequencies and freq_bins are JAX arrays."""
    freqs, bins = frequency_bins_generator(
        number_of_bins=8, freq_min=200, freq_max=2000, freq_distribution=dist
    )
    assert isinstance(freqs, jnp.ndarray), (
        f"frequencies is {type(freqs)}, not jnp.ndarray"
    )
    assert isinstance(bins, jnp.ndarray), f"freq_bins is {type(bins)}, not jnp.ndarray"
