import jax
import jax.numpy as jnp
from datasets import load_dataset
from jaxtyping import Array, Float
from scipy.signal import resample, sosfilt, sosfiltfilt, windows


def download_dataset(name="rfcx/frugalai", streaming=False, token=None, split="train"):
    """Downloads the RFCx FrugalAI dataset using the Hugging Face datasets library.
    to login:

    ```bash
    $ huggingface-cli login
    ```

    Returns:
        DatasetDict: The downloaded dataset containing training, validation, and test splits.
    """
    if token is not None:
        dataset = load_dataset(name, streaming=streaming, token=token, split=split)
    else:
        dataset = load_dataset(name, streaming=streaming, split=split)

    return dataset


def low_pass_filter(modulated_signal, window_size=301, gain=4.65415):
    """Applies a low-pass filter to the modulated signal using a Hann window.

    Args:
        modulated_signal (float): Input modulated signal (e.g., spike train).
        window_size (int): Size of the Hann window for filtering.
        gain (float): Gain factor to scale the filtered signal.

    Returns:
        float: The filtered signal.
    """
    kernel = windows.flattop(window_size) / window_size  # Create a Hann window kernel
    filtered_signal = jax.scipy.signal.convolve(
        modulated_signal * gain, kernel, mode="same"
    )

    return filtered_signal


def audio_resampling_and_scaling(
    audio: Float[Array, "#time"],
    original_frequency: float,
    target_frequency: float,
    scaling_factor=1.0,
) -> Float[Array, "#time"]:
    """Resamples and scales the input audio signal.

    Args:
        audio (Array): Input audio signal.
        original_frequency (float): Original sampling frequency of the audio signal.
        target_frequency (float): Target sampling frequency for resampling.
        scaling_factor (float or str): Scaling factor or method ('normalize') for scaling the audio signal.

    Returns:
        Array: Resampled and scaled audio signal.
    """
    if isinstance(scaling_factor, float):
        if scaling_factor != 1.0:
            audio = audio * scaling_factor
    elif isinstance(scaling_factor, str):
        if scaling_factor == "normalize":
            audio = audio / jnp.max(jnp.abs(audio))

        elif scaling_factor == "mulaw":
            audio = audio / jnp.max(jnp.abs(audio))
            audio = mu_encoding(audio, mu=255)

    if original_frequency != target_frequency:
        number_of_points = len(audio)
        audio = jnp.asarray(
            resample(
                audio, int(number_of_points * (target_frequency // original_frequency))
            )
        )

    return audio


def frequency_bins_generator(
    number_of_bins: int = 24,
    freq_min: float = 20.0,
    freq_max: float = 20000.0,
    freq_distribution: str = "linear",
    bins_superimpose: float = 0.0,
):
    if number_of_bins < 1:
        raise ValueError("number_of_bins must be at least 1")

    if freq_min > freq_max:
        raise ValueError("freq_min must be less or equal than freq_max")

    if freq_distribution == "linear":
        frequencies_bins = jnp.linspace(freq_min, freq_max, number_of_bins + 1)
        frequencies = jnp.array(
            [
                (frequencies_bins[i] + frequencies_bins[i + 1]) / 2
                for i in range(len(frequencies_bins) - 1)
            ]
        )
    elif freq_distribution == "log":
        if freq_min <= 0:
            raise ValueError("freq_min must be greater than 0 for log distribution")

        frequencies_bins = jnp.logspace(
            jnp.log10(freq_min), jnp.log10(freq_max), number_of_bins + 1
        )
        frequencies = jnp.array(
            [
                (frequencies_bins[i] + frequencies_bins[i + 1]) / 2
                for i in range(len(frequencies_bins) - 1)
            ]
        )
    elif freq_distribution == "mel":
        mel_start = 2595 * jnp.log10(1 + freq_min / 700)
        mel_end = 2595 * jnp.log10(1 + freq_max / 700)
        mel_bins = jnp.linspace(mel_start, mel_end, number_of_bins + 1)
        frequencies_bins = 700 * (10 ** (mel_bins / 2595) - 1)
        frequencies = jnp.array(
            [
                (frequencies_bins[i] + frequencies_bins[i + 1]) / 2
                for i in range(len(frequencies_bins) - 1)
            ]
        )
    return frequencies, frequencies_bins


def bandpass_signal(num_bands, audio, band_pass_window_array, low_pass_window_array):
    bands = []
    for i in range(num_bands):
        audio_min_len = 3 * (
            band_pass_window_array[i].shape[0] + 1
        )  # SciPy rule of thumb on ratio between window and signal length
        if audio.size >= audio_min_len:
            filtered_audio = sosfiltfilt(band_pass_window_array[i], audio)
        else:
            filtered_audio = sosfilt(band_pass_window_array[i], audio)
        bands.append(jnp.asarray(filtered_audio, dtype=jnp.float32))

    bands = jnp.abs(jnp.asarray(bands))

    bands_amplitude = []

    for i in range(num_bands):
        bands_min_len = 3 * (
            low_pass_window_array[i].shape[0] + 1
        )  # SciPy rule of thumb on ratio between window and signal length
        if bands[i].size >= bands_min_len:
            filtered_bands = sosfiltfilt(low_pass_window_array[i], bands[i])
        else:
            filtered_bands = sosfilt(low_pass_window_array[i], bands[i])
        bands_amplitude.append(jnp.asarray(filtered_bands, dtype=jnp.float32))

    return jnp.asarray(bands_amplitude)


def aer_to_array(event_time, event_address, event_magnitude, duration, sampling_rate):
    num_timesteps = int(duration * sampling_rate)
    num_neurons = jnp.max(event_address) + 1
    spikes_array = jnp.zeros((num_timesteps, num_neurons))
    for time, address, magnitude in zip(event_time, event_address, event_magnitude):
        timestep = int(time * sampling_rate)
        if timestep < num_timesteps:
            spikes_array = spikes_array.at[timestep, address].set(magnitude)
    return spikes_array


def mu_encoding(signal, mu=255):
    """Applies mu-law encoding to the input signal (it should be pronounced mi).

    Args:
        signal (float): Input audio signal.
        mu (int): Mu parameter for mu-law encoding.

    Returns:
        float: Mu-law encoded signal.
    """
    signal = jnp.clip(signal, -1.0, 1.0)
    encoded_signal = jnp.sign(signal) * jnp.log1p(mu * jnp.abs(signal)) / jnp.log1p(mu)
    return encoded_signal
