import jax.numpy as jnp
import numpy as np
import pytest
from scipy.signal import chirp

from crusafe.conversion_methods import filterbank_ADM


def make_sine(freq, sr, duration, amplitude=0.5):
    """Generate a sine wave signal.

    Args:
        freq (float): Frequency in Hz.
        sr (float): Sampling rate in Hz.
        duration (float): Duration in seconds.
        amplitude (float): Signal amplitude (0-1).

    Returns:
        JAX array containing the sine wave.
    """
    t = np.arange(int(sr * duration)) / sr
    return jnp.asarray(amplitude * np.sin(2 * np.pi * freq * t))


def make_chirp(sr, duration, f0, f1, method="linear"):
    """Generate a chirp signal (frequency sweep).

    Args:
        sr (float): Sampling rate in Hz.
        duration (float): Duration in seconds.
        f0 (float): Start frequency in Hz.
        f1 (float): End frequency in Hz.
        method (str): Chirp method ('linear', 'quadratic', etc.).

    Returns:
        JAX array containing the chirp signal.
    """
    t = np.arange(int(sr * duration)) / sr
    return jnp.asarray(chirp(t, f0=f0, f1=f1, t1=duration, method=method))


class TestFilterbankADMConstructor:
    """Test filterbank_ADM initialization and basic properties."""

    def test_constructor_with_defaults(self):
        """Verify constructor initializes with default parameters."""
        fb = filterbank_ADM()
        assert fb.sampling_rate == 4410000
        assert fb.num_neurons == 32
        assert fb.delta == 0.02
        assert fb.t_ref == 1e-3

    @pytest.mark.parametrize("num_neurons", [4, 8, 16, 32])
    def test_constructor_num_neurons(self, num_neurons):
        """Verify constructor accepts different number of neurons."""
        fb = filterbank_ADM(num_neurons=num_neurons)
        assert len(fb.frequencies) == num_neurons
        assert len(fb.frequencies_bins) == num_neurons + 1
        assert len(fb.band_pass_window_array) == num_neurons
        assert len(fb.low_pass_window_array) == num_neurons

    @pytest.mark.parametrize("dist", ["linear", "log", "mel"])
    def test_constructor_freq_distribution(self, dist):
        """Verify constructor accepts different frequency distributions."""
        fb = filterbank_ADM(num_neurons=8, freq_distribution=dist)
        assert fb.freq_distribution == dist
        assert len(fb.frequencies) == 8

    def test_frequencies_in_range(self):
        """Verify all frequency centers are within specified range."""
        fb = filterbank_ADM(
            num_neurons=16, freq_min=100, freq_max=5000, freq_distribution="linear"
        )
        assert jnp.all(fb.frequencies >= 100)
        assert jnp.all(fb.frequencies <= 5000)


class TestFilterbankADMOutput:
    """Test filterbank_ADM call output properties."""

    def test_output_types_are_jax_arrays(self):
        """Verify output types are JAX arrays."""
        fb = filterbank_ADM(num_neurons=8)
        audio = jnp.zeros(1000)
        event_time, event_address, event_magnitude = fb(audio)
        assert isinstance(event_time, jnp.ndarray)
        assert isinstance(event_address, jnp.ndarray)
        assert isinstance(event_magnitude, jnp.ndarray)

    def test_output_shapes_match(self):
        """Verify all output arrays have compatible shapes."""
        fb = filterbank_ADM(sampling_rate=44100, num_neurons=8)
        audio = make_chirp(44100, 1.0, f0=100, f1=20000)
        event_time, event_address, event_magnitude = fb(audio)
        # All should be 1-D arrays of the same length
        assert event_time.ndim == 1
        assert event_address.ndim == 1
        assert event_magnitude.ndim == 1
        assert len(event_time) == len(event_address) == len(event_magnitude)

    def test_zero_input_produces_no_spikes(self):
        """Verify that zero input produces no spikes."""
        fb = filterbank_ADM(num_neurons=10)
        audio = jnp.zeros(1024)
        event_time, event_address, event_magnitude = fb(audio)
        assert len(event_time) == 0
        assert len(event_address) == 0
        assert len(event_magnitude) == 0

    def test_event_addresses_valid(self):
        """Verify event addresses are within valid neuron indices."""
        fb = filterbank_ADM(num_neurons=8)
        audio = make_sine(500, 44100, 0.1, amplitude=0.5)
        event_time, event_address, event_magnitude = fb(audio, sampling_rate=44100)
        if len(event_address) > 0:
            assert jnp.all(event_address >= 0)
            assert jnp.all(event_address < fb.num_neurons)

    def test_event_magnitudes_are_signed(self):
        """Verify event magnitudes are ±1 (spike direction indicator)."""
        fb = filterbank_ADM(num_neurons=8)
        audio = make_sine(500, 44100, 0.1, amplitude=1.0)
        event_time, event_address, event_magnitude = fb(audio, sampling_rate=44100)
        if len(event_magnitude) > 0:
            unique_magnitudes = jnp.unique(event_magnitude)
            assert all(m in [-1.0, 1.0] for m in unique_magnitudes)

    def test_event_times_are_ordered(self):
        """Verify event times are monotonically increasing."""
        fb = filterbank_ADM(num_neurons=8)
        audio = make_sine(500, 44100, 0.2, amplitude=0.8)
        event_time, _, _ = fb(audio, sampling_rate=44100)
        if len(event_time) > 1:
            assert jnp.all(jnp.diff(event_time) >= 0)

    def test_event_times_in_valid_range(self):
        """Verify event times are within [0, duration]."""
        sr = 44100
        duration = 0.1
        fb = filterbank_ADM(num_neurons=8)
        audio = make_sine(500, sr, duration, amplitude=0.5)
        event_time, _, _ = fb(audio, sampling_rate=sr)
        if len(event_time) > 0:
            assert jnp.all(event_time >= 0)
            assert jnp.all(event_time <= duration)


class TestFilterbankADMSignalTypes:
    """Test filterbank_ADM response to different signal types."""

    def test_sine_generates_spikes(self):
        """Verify sine wave generates spikes."""
        sr = 44100
        fb = filterbank_ADM(num_neurons=8)
        audio = make_sine(500, sr, 0.1, amplitude=0.5)
        event_time, event_address, event_magnitude = fb(audio, sampling_rate=sr)
        assert len(event_time) > 0

    def test_chirp_generates_spikes_across_bands(self):
        """Verify chirp signal generates spikes from multiple bands."""
        sr = 441000
        fb = filterbank_ADM(
            num_neurons=12, freq_min=100, freq_max=20000, freq_distribution="log"
        )
        audio = make_chirp(sr, 1.0, f0=100, f1=20000, method="linear")
        event_time, event_address, event_magnitude = fb(audio, sampling_rate=sr)
        assert len(event_time) > 0
        # Expect spikes from multiple bands due to frequency sweep
        unique_addresses = jnp.unique(event_address)
        assert len(unique_addresses) == 12

    # TODO: ADM Filter bank should not detect a single tone. Should detect changes
    # def test_tone_in_band_generates_spikes_primarily_in_band(self):
    #     """Verify tone centered in a band generates most spikes in that band."""
    #     sr = 44100
    #     num_neurons = 8
    #     fb = filterbank_ADM(
    #         num_neurons=num_neurons,
    #         freq_min=200,
    #         freq_max=2000,
    #         freq_distribution="log",
    #     )
    #     # Tone at the center frequency of band 4
    #     target_band = num_neurons // 2
    #     target_freq = float(fb.frequencies[target_band])
    #     audio = make_sine(target_freq, sr, 0.2, amplitude=0.8)
    #     event_time, event_address, event_magnitude = fb(audio, sampling_rate=sr)

    #     if len(event_address) > 0:
    #         spikes_in_band = jnp.sum(event_address == target_band)
    #         total_spikes = len(event_address)
    #         fraction = float(spikes_in_band) / total_spikes
    #         # Expect at least 50% of spikes in the target band
    #         assert fraction >= 0.5, (
    #             f"Only {fraction:.1%} spikes in target band; expected >=50%"
    #         )
    #     else:
    #         pytest.fail("No spikes generated; cannot verify band specificity.")


class TestFilterbankADMParameterSensitivity:
    """Test sensitivity to parameter changes."""

    def test_delta_affects_spike_rate(self):
        """Verify larger delta produces fewer spikes."""
        sr = 44100
        audio = make_sine(500, sr, 0.2, amplitude=0.8)

        fb_small = filterbank_ADM(num_neurons=8, delta=0.01)
        t_small, _, _ = fb_small(audio, sampling_rate=sr)

        fb_large = filterbank_ADM(num_neurons=8, delta=0.1)
        t_large, _, _ = fb_large(audio, sampling_rate=sr)

        # Larger delta should produce fewer or equal spikes
        assert len(t_large) <= len(t_small)

    @pytest.mark.parametrize("sr", [44100, 48000, 22050])
    def test_different_sampling_rates_consistent(self, sr):
        """Verify filterbank works with different sampling rates."""
        fb = filterbank_ADM(num_neurons=8, freq_min=100, freq_max=5000)
        audio = make_sine(500, sr, 0.1, amplitude=0.5)
        event_time, event_address, event_magnitude = fb(audio, sampling_rate=sr)

        assert isinstance(event_time, jnp.ndarray)
        assert isinstance(event_address, jnp.ndarray)
        if len(event_address) > 0:
            assert jnp.all(event_address >= 0)
            assert jnp.all(event_address < fb.num_neurons)


class TestFilterbankADMEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_short_audio(self):
        """Verify filterbank handles very short audio."""
        fb = filterbank_ADM(num_neurons=8)
        audio = jnp.zeros(16)
        event_time, event_address, event_magnitude = fb(audio)
        # Should not crash; may produce no spikes
        assert len(event_time) == len(event_address) == len(event_magnitude)

    def test_very_long_audio(self):
        """Verify filterbank handles longer audio."""
        sr = 44100
        fb = filterbank_ADM(num_neurons=8)
        audio = make_sine(50000, sr, 1.0, amplitude=0.5)  # 1 second
        event_time, event_address, event_magnitude = fb(audio, sampling_rate=sr)
        assert len(event_time) == len(event_address) == len(event_magnitude)

    def test_very_small_amplitude(self):
        """Verify filterbank handles very small amplitude signals."""
        sr = 44100
        fb = filterbank_ADM(num_neurons=8)
        audio = make_sine(500, sr, 0.1, amplitude=0.01)  # Very small
        event_time, event_address, event_magnitude = fb(audio, sampling_rate=sr)
        # May produce no spikes or very few
        assert len(event_time) == len(event_address) == len(event_magnitude)

    def test_large_amplitude(self):
        """Verify filterbank handles large amplitude signals."""
        sr = 44100
        fb = filterbank_ADM(num_neurons=8)
        audio = make_sine(500, sr, 0.1, amplitude=1.0)
        event_time, event_address, event_magnitude = fb(audio, sampling_rate=sr)
        assert len(event_time) > 0  # Should generate spikes


class TestFilterbankADMConsistency:
    """Test consistency and reproducibility."""

    def test_repeated_call_same_signal_produces_same_output(self):
        """Verify repeated calls with same input produce same output."""
        sr = 44100
        fb = filterbank_ADM(num_neurons=8)
        audio = make_sine(500, sr, 0.1, amplitude=0.5)

        t1, a1, m1 = fb(audio, sampling_rate=sr)
        t2, a2, m2 = fb(audio, sampling_rate=sr)

        assert jnp.allclose(t1, t2)
        assert jnp.allclose(a1, a2)
        assert jnp.allclose(m1, m2)

    def test_initialization_parameters_preserved(self):
        """Verify initialization parameters are stored correctly."""
        params = {
            "sampling_rate": 48000,
            "num_neurons": 16,
            "freq_min": 50,
            "freq_max": 10000,
            "delta": 0.015,
            "t_ref": 2,
            "freq_distribution": "mel",
        }
        fb = filterbank_ADM(**params)
        assert fb.sampling_rate == params["sampling_rate"]
        assert fb.num_neurons == params["num_neurons"]
        assert fb.delta == params["delta"]
        assert fb.t_ref == params["t_ref"]
        assert fb.freq_distribution == params["freq_distribution"]
