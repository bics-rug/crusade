from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int8
from scipy.signal import butter

from . import utils


class sigma_delta_neuron_in_the_loop:
    """Sigma-Delta IF Neuron Model
    Implements an adaptive sigma-delta neuron model that converts continuous audio signals into spike trains.
    The difference from the standard sigma-delta model is that the feed back spikes and not continuous values.
    IT has an integrator in series with a integrate and fire neuron.
    Attributes:
        sampling_rate: The sampling rate of the input audio signal.
        feedback_gain (float): The feedback gain for the neuron model.
        threshold (float): The threshold for spike generation.
    Methods:
        __call__(audio, feedback_gain=None, threshold=None): Converts the input audio signal into spike trains.
    """

    def __init__(
        self,
        sampling_rate: float = 4410000,
        feedback_gain: float = 2.47498e-07,
        threshold: float = 3.22814e-13,
    ):
        self.sampling_rate = sampling_rate
        self.feedback_gain = feedback_gain
        self.threshold = threshold

    def __call__(
        self,
        audio: Float[Array, "#time"],
        feedback_gain: Optional[float] = None,
        threshold: Optional[float] = None,
        sampling_rate: Optional[float] = None,
    ) -> tuple[Float[Array, "*time"], Int8[Array, "*time"]]:
        if feedback_gain is None:
            feedback_gain = self.feedback_gain
        if threshold is None:
            threshold = self.threshold
        if sampling_rate is None:
            sampling_rate = self.sampling_rate

        @jax.jit
        def body_fun(state, jj):
            ii, mem_previous, spike_previous = (
                state  # Unpack integrator and memory state
            )

            ii = (
                ii + jj / sampling_rate - spike_previous * feedback_gain
            )  # First integrator stage (accumulate input)
            mem = (
                mem_previous + ii / sampling_rate
            )  # Membrane potential update using neuron gain

            # Generate spike if membrane potential crosses thresholds aeset membrane potential if spike occurred
            spike = (mem >= threshold).astype(int) - (mem <= -threshold).astype(int)
            mem = (
                1 - jnp.abs(spike)
            ) * mem  # Reset membrane potential if spike occurred

            return (ii, mem, spike), spike

        ii = 0
        _, out_spikes = jax.lax.scan(body_fun, (ii, 0, 0), audio)
        time_ax = jnp.linspace(0, len(audio) / sampling_rate, len(audio))
        event_mask = out_spikes != 0
        event_time = time_ax[event_mask]
        event_address = (out_spikes[event_mask] == 1).astype(jnp.int8)
        return event_time, event_address


class sigma_delta_spiking:
    """Sigma-Delta Model with spiking comparator and feedback
    Implements a sigma-delta neuron model that converts continuous audio signals into spike trains.
    Attributes:
        sampling_rate: The sampling rate of the input audio signal.
        feedback_gain (float): The feedback gain for the neuron model.
        threshold (float): The threshold for spike generation.
    Methods:
        __call__(audio, feedback_gain=None, threshold=None): Converts the input audio signal into spike trains.
    """

    def __init__(
        self,
        sampling_rate: float = 4410000,
        feedback_gain: float = 5.75174e-07,
        threshold: float = 6.867776e-07,
    ):
        self.sampling_rate = sampling_rate
        self.feedback_gain = feedback_gain
        self.threshold = threshold

    def __call__(
        self,
        audio: Float[Array, "#time"],
        feedback_gain: Optional[float] = None,
        threshold: Optional[float] = None,
        sampling_rate: Optional[float] = None,
    ) -> tuple[Float[Array, "*time"], Int8[Array, "*time"]]:
        if feedback_gain is None:
            feedback_gain = self.feedback_gain
        if threshold is None:
            threshold = self.threshold
        if sampling_rate is None:
            sampling_rate = self.sampling_rate

        @jax.jit
        def body_fun(carry, input_val):
            (integrator, mem_p, mem_n) = carry
            quantized = jnp.sign(integrator)

            mem_p = jnp.where(quantized == 1, mem_p + 1 / sampling_rate, mem_p)
            mem_n = jnp.where(quantized == -1, mem_n + 1 / sampling_rate, mem_n)

            spikes_p = jnp.where(mem_p >= threshold, 1, 0)
            spikes_n = jnp.where(mem_n >= threshold, 1, 0)

            mem_p = jnp.where(spikes_p == 1, 0, mem_p)
            mem_n = jnp.where(spikes_n == 1, 0, mem_n)

            spikes = spikes_p - spikes_n

            new_integrator = (
                integrator + input_val / sampling_rate - feedback_gain * spikes
            )

            return (new_integrator, mem_p, mem_n), spikes

        initial_carry = (0.0, 0.0, 0.0)
        _, out_spikes = jax.lax.scan(body_fun, initial_carry, audio)
        time_ax = jnp.linspace(0, len(audio) / sampling_rate, len(audio))
        event_mask = out_spikes != 0
        event_time = time_ax[event_mask]
        event_address = (out_spikes[event_mask] == 1).astype(jnp.int8)
        return event_time, event_address


class sigma_delta_edge_spikes:
    """Sigma-Delta ADC which transmits only spikes generate during transitions.
    Attributes:
        sampling_rate: The sampling rate of the input audio signal.
        feedback_gain (float): The feedback gain for the sigma delta, always 1.
        threshold (float): The threshold for spike generation.
    Methods:
        __call__(audio, feedback_gain=None, threshold=None): Converts the input audio signal into spike trains.
    """

    def __init__(
        self,
        sampling_rate: float = 4410000,
        feedback_gain: float = 1,
        threshold: float = 1.98682e-06,
    ):
        self.sampling_rate = sampling_rate
        self.feedback_gain = feedback_gain
        self.threshold = threshold

    def __call__(
        self,
        audio: Float[Array, "#time"],
        feedback_gain: Optional[float] = None,
        threshold: Optional[float] = None,
        sampling_rate: Optional[float] = None,
    ):
        if feedback_gain is None:
            feedback_gain = self.feedback_gain
        if threshold is None:
            threshold = self.threshold
        if sampling_rate is None:
            sampling_rate = self.sampling_rate

        @jax.jit
        def body_fun(carry, input_val):
            (integrator, prev_quant) = carry
            quantized = jnp.sign(integrator + threshold * prev_quant)
            spk = jnp.where(quantized == prev_quant, 0, quantized)

            new_integrator = (
                integrator + (input_val - feedback_gain * quantized) / sampling_rate
            )

            return (new_integrator, quantized), spk

        initial_carry = (0.0, 0.0)
        _, out_spikes = jax.lax.scan(body_fun, initial_carry, audio)
        time_ax = jnp.linspace(0, len(audio) / sampling_rate, len(audio))
        event_mask = out_spikes != 0
        event_time = time_ax[event_mask]
        event_address = (out_spikes[event_mask] == 1).astype(jnp.int8)

        return event_time, event_address


class amplitude_to_frequency:
    """Amplitude to Frequency Conversion
    Converts the amplitude of the input audio signal into a frequency-modulated spike train.
    Attributes:
        sampling_rate: The sampling rate of the input audio signal.
        max_freq (float): The maximum frequency of the output spike train.
    Methods:
        __call__(audio, sampling_rate=None, max_freq=None): Converts the input audio signal into a frequency-modulated spike train.
    """

    def __init__(self, sampling_rate: float = 4410000, max_freq: float = 2000000):
        self.sampling_rate = sampling_rate
        self.max_freq = max_freq

    def __call__(
        self,
        audio: Float[Array, "#time"],
        sampling_rate: Optional[float] = None,
        max_freq: Optional[float] = None,
    ):
        if sampling_rate is None:
            sampling_rate = self.sampling_rate
        if max_freq is None:
            max_freq = self.max_freq

        random_number = jax.random.uniform(jax.random.PRNGKey(0), shape=audio.shape)
        spikes_p = random_number < (audio * max_freq / sampling_rate)
        spikes_n = random_number < (-audio * max_freq / sampling_rate)
        out_spikes = spikes_p.astype(int) - spikes_n.astype(int)

        time_ax = jnp.linspace(0, len(audio) / sampling_rate, len(audio))
        event_mask = out_spikes != 0

        event_time = time_ax[event_mask]
        event_address = (out_spikes[event_mask] == 1).astype(jnp.int8)

        return event_time, event_address


class resonate_and_fire_bank:
    """Resonate-and-Fire Neuron Bank
    Implements a bank of resonate-and-fire neurons that convert continuous audio signals into spike trains.
    Adapted from Giuseppe Leo code
    Attributes:
        sampling_rate: The sampling rate of the input audio signal.
        num_neurons (int): The number of resonate-and-fire neurons in the bank.
        freq_min (float): The minimum frequency of the resonate-and-fire neurons.
        freq_max (float): The maximum frequency of the resonate-and-fire neurons.
        damping_factor (float): The damping factor for the resonate-and-fire neurons.
        input_gain (float): The gain applied to the input audio signal.
        freq_distribution (str): The distribution of frequencies for the resonate-and-fire neurons ('linear', 'log', 'mel').
        threshold (float): The threshold for spike generation.
    Methods:
        __call__(audio, sampling_rate=None, damping_factor=None, input_gain=None, threshold=None): Converts the input audio signal into spike trains.
    """

    def __init__(
        self,
        sampling_rate: float = 4410000,
        num_neurons: int = 32,
        freq_min: float = 20,
        freq_max: float = 20000,
        damping_factor_scaling: float = 1,
        input_gain: float = 10,
        freq_distribution: str = "linear",
        threshold: float = 1.0,
        debug: bool = False,
    ):
        self.sampling_rate = sampling_rate
        self.num_neurons = num_neurons
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.threshold = threshold
        self.freq_distribution = freq_distribution
        self.damping_factor_scaling = damping_factor_scaling
        self.input_gain = input_gain
        self.debug = debug

        # Create frequency array for the resonate-and-fire neurons
        self.frequencies, self.frequencies_bins = utils.frequency_bins_generator(
            number_of_bins=self.num_neurons,
            freq_min=self.freq_min,
            freq_max=self.freq_max,
            freq_distribution=self.freq_distribution,
            bins_superimpose=0.0,
        )

        delta_freq = jnp.array(
            [
                self.frequencies_bins[i + 1] - self.frequencies_bins[i]
                for i in range(len(self.frequencies_bins) - 1)
            ]
        )
        self.omega = 2 * jnp.pi * self.frequencies
        natural_omega = (
            self.omega / jnp.sqrt(1 - ((delta_freq / self.frequencies) ** 2) / 2)
        )  # here sometimes gives a NAN for the first neuron cause frequencies[1] sould be less than 3*frequencies[0]
        damping_ratio = (2 * jnp.pi * delta_freq) / (2 * self.omega)
        self.damping_factors = (
            -damping_ratio * natural_omega * self.damping_factor_scaling
        )

        # factors for analitical solution
        self.exp_damping = jnp.exp(self.damping_factors / sampling_rate)
        self.cos_omega = jnp.cos(self.omega / sampling_rate)
        self.sin_omega = jnp.sin(self.omega / sampling_rate)

    def __call__(
        self,
        audio: Float[Array, "#time"],
        sampling_rate: Optional[float] = None,
        input_gain: Optional[float] = None,
        threshold: Optional[float] = None,
        damping_factor_scaling: Optional[float] = None,
    ):
        if sampling_rate is None:
            sampling_rate = self.sampling_rate

        if damping_factor_scaling is None:
            damping_factor_scaling = self.damping_factor_scaling

        if (sampling_rate != self.sampling_rate) or (
            damping_factor_scaling != self.damping_factor_scaling
        ):
            self.sampling_rate = sampling_rate
            self.damping_factor_scaling = damping_factor_scaling
            self.exp_damping = jnp.exp(
                self.damping_factors * damping_factor_scaling / sampling_rate
            )
            self.cos_omega = jnp.cos(self.omega / sampling_rate)
            self.sin_omega = jnp.sin(self.omega / sampling_rate)

        if input_gain is None:
            input_gain = self.input_gain
        if threshold is None:
            threshold = self.threshold

        if self.debug:

            @jax.jit
            def body_fun(carry, input_val):
                (x_prev, y_prev) = carry

                # exact dynamics
                x = (
                    self.exp_damping
                    * (x_prev * self.cos_omega - y_prev * self.sin_omega)
                    + input_gain
                    * input_val
                    * self.damping_factors
                    * damping_factor_scaling
                    * 2
                    / sampling_rate
                )
                y = self.exp_damping * (
                    x_prev * self.sin_omega + y_prev * self.cos_omega
                )

                # pike generation and reset
                spikes = (y > threshold).astype(int)
                y = (1 - spikes) * y + spikes * threshold  # reset y if spike occurred
                x = (1 - spikes) * x  # reset x if spike occurred

                return (x, y), (spikes, x, y)

            initial_carry = (
                jnp.zeros((self.num_neurons,)),
                jnp.zeros((self.num_neurons,)),
            )
            _, (out_spikes, x, y) = jax.lax.scan(body_fun, initial_carry, audio)

            time_ax = jnp.linspace(0, len(audio) / sampling_rate, len(audio))
            event_v, address_v = jnp.where(out_spikes != 0)
            event_time = time_ax[event_v]
            event_address = address_v

            return event_time, event_address, x, y

        else:

            @jax.jit
            def body_fun(carry, input_val):
                (x_prev, y_prev) = carry

                # exact dynamics
                x = (
                    self.exp_damping
                    * (x_prev * self.cos_omega - y_prev * self.sin_omega)
                    + input_gain
                    * input_val
                    * self.damping_factors
                    * damping_factor_scaling
                    * 2
                    / sampling_rate
                )
                y = self.exp_damping * (
                    x_prev * self.sin_omega + y_prev * self.cos_omega
                )

                # pike generation and reset
                spikes = (y > threshold).astype(int)
                y = (1 - spikes) * y + spikes * threshold  # reset y if spike occurred
                x = (1 - spikes) * x  # reset x if spike occurred

                return (x, y), spikes

            initial_carry = (
                jnp.zeros((self.num_neurons,)),
                jnp.zeros((self.num_neurons,)),
            )
            _, out_spikes = jax.lax.scan(body_fun, initial_carry, audio)

            time_ax = jnp.linspace(0, len(audio) / sampling_rate, len(audio))
            event_v, address_v = jnp.where(out_spikes != 0)
            event_time = time_ax[event_v]
            event_address = address_v

            return event_time, event_address

    def dynamic_range_test(self):
        max_input = []
        min_input = []
        dynamic_range = []

        # search for all the central freqs of the array
        for freq in self.frequencies:
            # search for max input
            test_time = 2 / freq
            ampl_max = []
            ampl_max.append(10)
            time = jnp.arange(0, test_time, 1 / self.sampling_rate)
            for search_step in jnp.arange(1, 20, 1):
                event_time, event_address = self.__call__(
                    audio=jnp.sin(jnp.pi * time * freq) * ampl_max[-1]
                )
                saturation = 0
                for jj in jnp.arange(0, self.num_neurons, 1):
                    count_x = jnp.count_nonzero(event_address == jj)
                    if count_x >= 1:  # ((test_time/freq)*0.99):
                        saturation = saturation + 1
                if saturation == 0:
                    ampl_max.append(ampl_max[-1] + ampl_max[0] / 2**search_step)
                elif saturation >= 1:
                    ampl_max.append(ampl_max[-1] - ampl_max[0] / 2**search_step)

            # search for min input
            ampl_min = []
            ampl_min.append(10)
            test_time = 50 / freq
            time = jnp.arange(0, test_time, 1 / self.sampling_rate)
            for search_step in jnp.arange(1, 20, 1):
                event_time, event_address = self.__call__(
                    audio=jnp.sin(jnp.pi * time * freq) * ampl_min[-1]
                )

                activation = 0
                for jj in jnp.arange(0, self.num_neurons, 1):
                    count_x = jnp.count_nonzero(event_address == jj)
                    if count_x > 0:
                        activation = activation + 1
                if activation == 0:
                    ampl_min.append(ampl_min[-1] + ampl_min[0] / 2**search_step)
                elif activation >= 1:
                    ampl_min.append(ampl_min[-1] - ampl_min[0] / 2**search_step)

            max_input.append(ampl_max[-1])
            min_input.append(ampl_min[-1])
            dynamic_range.append(jnp.log10(ampl_max[-1] / ampl_min[-1]) * 2)

        return dynamic_range, max_input, min_input


class standard_ADM:
    """Standard Adaptive Delta Modulator (ADM)
    Implements a standard adaptive delta modulator that converts continuous audio signals into spike trains.
    Based on Olympia Gallou code
    Attributes:
        sampling_rate: The sampling rate of the input audio signal.
        threshold (float): The threshold for spike generation.
        t_ref (int): The refractory period in number of samples. (Not implemented)
    Methods:
        __call__(audio, sampling_rate=None, threshold=None, t_ref=None): Converts the input audio signal into spike trains.
    """

    # TODO: Implement t_ref functionality
    def __init__(
        self, sampling_rate: float = 4410000, threshold: float = 0.01, t_ref: int = 1
    ):
        self.sampling_rate = sampling_rate
        self.threshold = threshold
        self.t_ref = t_ref

    def __call__(
        self,
        audio: Float[Array, "#time *N"],
        sampling_rate: Optional[float] = None,
        threshold: Optional[float] = None,
        t_ref: Optional[int] = None,
        return_dense: bool = False,  # Use for vmap
    ):
        if sampling_rate is None:
            sampling_rate = self.sampling_rate
        if threshold is None:
            threshold = self.threshold
        if t_ref is None:
            t_ref = self.t_ref

        @jax.jit
        def body_fun(carry, input_val):
            (thr_DN, thr_UP, count_threshold) = carry

            spikes = (
                (input_val >= thr_UP).astype(jnp.int32)
                - (input_val <= thr_DN).astype(jnp.int32)
            ) * (count_threshold.astype(jnp.int32) >= 1).astype(
                jnp.int32
            )  # Generate spikes based on thresholds and refractory period

            count_threshold = (count_threshold + 1) * (
                1 - jnp.abs(spikes)
            )  # Reset counter if spike occurred
            thr_UP = thr_UP * (1 - jnp.abs(spikes)) + (input_val + threshold) * jnp.abs(
                spikes
            )  # Update upper threshold if spike occurred
            thr_DN = thr_DN * (1 - jnp.abs(spikes)) + (input_val - threshold) * jnp.abs(
                spikes
            )  # Update lower threshold if spike occurred

            return (thr_DN, thr_UP, count_threshold), spikes

        initial_carry = (
            audio[0] - threshold,
            audio[0] + threshold,
            jnp.zeros_like(audio[0], dtype=jnp.int32),
        )

        _, out_spikes = jax.lax.scan(body_fun, initial_carry, audio)

        time_ax = jnp.linspace(0, len(audio) / sampling_rate, len(audio))
        spikes_2d = jnp.atleast_2d(out_spikes.T).T
        N = spikes_2d.shape[1]

        if return_dense:
            max_spikes = spikes_2d.shape[0] * spikes_2d.shape[1]
            time_idx, channel_idx = jnp.where(
                spikes_2d != 0, size=max_spikes, fill_value=-1
            )

            safe_time_idx = jnp.where(time_idx >= 0, time_idx, 0)
            safe_channel_idx = jnp.where(channel_idx >= 0, channel_idx, 0)

            event_time = time_ax[safe_time_idx]
            event_address = jnp.where(
                spikes_2d[safe_time_idx, safe_channel_idx] == 1,
                safe_channel_idx,
                safe_channel_idx + N,
            )

            # Set invalid entries to inf to mark them as padding
            event_time = jnp.where(time_idx >= 0, event_time, jnp.inf)
            event_address = jnp.where(time_idx >= 0, event_address, jnp.inf)
        else:
            time_idx, channel_idx = jnp.where(spikes_2d != 0)
            event_time = time_ax[time_idx]
            event_address = jnp.where(
                spikes_2d[time_idx, channel_idx] == 1, channel_idx, channel_idx + N
            )

        return event_time, event_address


class filterbank_ADM:
    """Filterbank Adaptive Delta Modulator (ADM)
    Implements a filterbank-based adaptive delta modulator that converts continuous audio signals into spike trains.
    Based on Olympia Gallou code.
    Attributes:
        sampling_rate: The sampling rate of the input audio signal.
        num_neurons (int): The number of neurons in the filterbank.
        freq_min (float): The minimum frequency of the filterbank neurons.
        freq_max (float): The maximum frequency of the filterbank neurons.
        freq_distribution (str): The distribution of frequencies for the filterbank neurons ('linear', 'log', 'mel').
        delta (float): The delta value for threshold adaptation.
        t_ref (float): The refractory period in seconds.
    Methods:
        __call__(audio, sampling_rate=None, threshold=None, t_ref=None): Converts the input audio signal into spike trains.
    """

    def __init__(
        self,
        sampling_rate: float = 4410000,
        num_neurons: int = 32,
        freq_min: float = 200,
        freq_max: float = 20000,
        freq_distribution: str = "linear",
        delta: float = 0.02,
        t_ref: float = 1e-3,
    ):
        self.sampling_rate = sampling_rate
        self.num_neurons = num_neurons
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.freq_distribution = freq_distribution
        self.delta = delta
        self.t_ref = t_ref
        self.step_ref = int(t_ref * sampling_rate)

        self.frequencies, self.frequencies_bins = utils.frequency_bins_generator(
            number_of_bins=self.num_neurons,
            freq_min=self.freq_min,
            freq_max=self.freq_max,
            freq_distribution=self.freq_distribution,
            bins_superimpose=0.0,
        )

        self.band_pass_window_array = []
        self.low_pass_window_array = []
        for i in range(self.num_neurons):
            self.band_pass_window_array.append(
                butter(
                    N=2,
                    Wn=[self.frequencies_bins[i], self.frequencies_bins[i + 1]],
                    btype="bandpass",
                    fs=sampling_rate,
                    output="sos",
                )
            )
            self.low_pass_window_array.append(
                butter(N=2, Wn=100, btype="lowpass", fs=sampling_rate, output="sos")
            )

    def __call__(
        self,
        audio: Float[Array, "#time"],
        sampling_rate: Optional[float] = None,
        delta: Optional[float] = None,
        t_ref: Optional[float] = None,
    ):
        if sampling_rate is None:
            sampling_rate = self.sampling_rate
        if delta is None:
            delta = self.delta
        if t_ref is None:
            t_ref = self.t_ref

        self.t_ref = t_ref
        self.step_ref = int(t_ref * sampling_rate)

        if sampling_rate != self.sampling_rate:
            self.sampling_rate = sampling_rate
            self.band_pass_window_array = []
            self.low_pass_window_array = []
            for i in range(self.num_neurons):
                self.band_pass_window_array.append(
                    butter(
                        N=2,
                        Wn=[self.frequencies_bins[i], self.frequencies_bins[i + 1]],
                        btype="bandpass",
                        fs=sampling_rate,
                        output="sos",
                    )
                )
                self.low_pass_window_array.append(
                    butter(N=2, Wn=100, btype="lowpass", fs=sampling_rate, output="sos")
                )

        audio_bands = utils.bandpass_signal(
            num_bands=self.num_neurons,
            audio=audio,
            band_pass_window_array=self.band_pass_window_array,
            low_pass_window_array=self.low_pass_window_array,
        )

        # for audio_band in audio_bands:
        #     plt.plot(audio_band[::1000])
        # plt.legend()
        # plt.show()

        @jax.jit
        def body_fun(carry, input_val):
            (thr_DN, thr_UP, count_threshold) = carry

            spikes = (
                (input_val >= thr_UP).astype(int) - (input_val <= thr_DN).astype(int)
            ) * (count_threshold.astype(int) >= self.step_ref).astype(
                int
            )  # Generate spikes based on thresholds and refractory period (in steps)

            count_threshold = (count_threshold + 1) * (
                1 - jnp.abs(spikes)
            )  # Reset counter if spike occurred
            thr_UP = thr_UP * (1 - jnp.abs(spikes)) + (input_val + delta) * jnp.abs(
                spikes
            )  # Update upper threshold if spike occurred
            thr_DN = thr_DN * (1 - jnp.abs(spikes)) + (input_val - delta) * jnp.abs(
                spikes
            )  # Update lower threshold if spike occurred

            return (thr_DN, thr_UP, count_threshold), spikes

        initial_carry = (
            audio_bands[:, 0] - delta,
            audio_bands[:, 0] + delta,
            jnp.zeros((self.num_neurons,)),
        )
        _, out_spikes = jax.lax.scan(body_fun, initial_carry, audio_bands.T)

        time_ax = jnp.linspace(0, len(audio) / sampling_rate, len(audio))
        event_v, address_v = jnp.where(out_spikes != 0)

        event_time = time_ax[event_v]
        event_address = address_v
        event_magnitude = out_spikes[event_v, address_v]

        return event_time, event_address, event_magnitude

    def dynamic_range_test(self):
        max_input = []
        min_input = []
        dynamic_range = []

        test_time = 0.1
        time = jnp.arange(0, test_time, 1 / self.sampling_rate)
        for freq in self.frequencies:
            ampl_max = []
            ampl_max.append(10)
            for search_step in jnp.arange(1, 20, 1):
                _, event_address, _ = self.__call__(
                    audio=jnp.sin(jnp.pi * time * freq)
                    * (
                        jnp.abs(
                            2 * (time / test_time - jnp.floor(time / test_time + 0.5))
                        )
                    )
                    * ampl_max[-1]
                )

                saturation = 0
                for jj in jnp.arange(0, self.num_neurons, 1):
                    count_x = jnp.count_nonzero(event_address == jj)
                    # print(count_x)
                    if count_x >= ((test_time / self.t_ref) * 0.95):
                        saturation = saturation + 1
                if saturation == 0:
                    ampl_max.append(ampl_max[-1] + ampl_max[0] / 2**search_step)
                elif saturation >= 1:
                    ampl_max.append(ampl_max[-1] - ampl_max[0] / 2**search_step)

            ampl_min = []
            ampl_min.append(10)
            for search_step in jnp.arange(1, 20, 1):
                _, event_address, _ = self.__call__(
                    audio=jnp.sin(jnp.pi * time * freq)
                    * (
                        jnp.abs(
                            2 * (time / test_time - jnp.floor(time / test_time + 0.5))
                        )
                    )
                    * ampl_min[-1]
                )

                activation = 0
                for jj in jnp.arange(0, self.num_neurons, 1):
                    count_x = jnp.count_nonzero(event_address == jj)
                    if count_x > 0:
                        activation = activation + 1
                if activation == 0:
                    ampl_min.append(ampl_min[-1] + ampl_min[0] / 2**search_step)
                elif activation >= 1:
                    ampl_min.append(ampl_min[-1] - ampl_min[0] / 2**search_step)

            dynamic_range.append(2 * jnp.log10(ampl_max[-1] / ampl_min[-1]))
            max_input.append(ampl_max[-1])
            min_input.append(ampl_min[-1])

        return dynamic_range, max_input, min_input


class filterbank_sync_phase:
    """Based on the code of Patrick Boesch and Qi Shen"""

    def __init__(
        self,
        sampling_rate: float = 4410000,
        num_neurons: int = 32,
        freq_min: float = 200,
        freq_max: float = 20000,
        freq_distribution: str = "linear",
        lif_variability: float = 0.01,
        lif_exc_idc: float = 2.0,
        lif_exc_tau_membrane: float = 20e-3,
        lif_exc_tau_synapse: float = 20e-3,
        lif_exc_threshold: float = 1.0,
        lif_exc_tau_ref: float = 5e-3,
        lif_inh_idc: float = 0.0,
        lif_inh_tau_membrane: float = 20e-3,
        lif_inh_tau_synapse: float = 5e-3,
        lif_inh_threshold: float = 1.0,
        lif_inh_tau_ref: float = 5e-3,
        weight_ee: float = 2.0,
        weight_ei: float = 10.0,
        weight_ie: float = 9.0,
        weight_ii: float = 2.0,
        weight_in: float = 0.02,
    ):
        self.sampling_rate = sampling_rate
        self.num_neurons = num_neurons
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.freq_distribution = freq_distribution
        self.lif_variability = lif_variability
        self.lif_exc_idc = lif_exc_idc
        self.lif_exc_tau_membrane = lif_exc_tau_membrane
        self.lif_exc_tau_synapse = lif_exc_tau_synapse
        self.lif_exc_threshold = lif_exc_threshold
        self.lif_exc_tau_ref = lif_exc_tau_ref
        self.lif_inh_idc = lif_inh_idc
        self.lif_inh_tau_membrane = lif_inh_tau_membrane
        self.lif_inh_tau_synapse = lif_inh_tau_synapse
        self.lif_inh_threshold = lif_inh_threshold
        self.lif_inh_tau_ref = lif_inh_tau_ref
        self.weight_ee = weight_ee
        self.weight_ei = weight_ei
        self.weight_ie = weight_ie
        self.weight_ii = weight_ii
        self.weight_in = weight_in

        self.step_ref_exc = int(self.lif_exc_tau_ref * self.sampling_rate)
        self.step_ref_inh = int(self.lif_inh_tau_ref * self.sampling_rate)

        self.frequencies, self.frequencies_bins = utils.frequency_bins_generator(
            number_of_bins=self.num_neurons,
            freq_min=self.freq_min,
            freq_max=self.freq_max,
            freq_distribution=self.freq_distribution,
            bins_superimpose=0.0,
        )

        self.band_pass_window_array = []
        self.low_pass_window_array = []
        for i in range(self.num_neurons):
            self.band_pass_window_array.append(
                butter(
                    N=2,
                    Wn=[self.frequencies_bins[i], self.frequencies_bins[i + 1]],
                    btype="bandpass",
                    fs=sampling_rate,
                    output="sos",
                )
            )
            self.low_pass_window_array.append(
                butter(N=2, Wn=100, btype="lowpass", fs=sampling_rate, output="sos")
            )

    def __call__(
        self,
        audio: Float[Array, "#time"],
        sampling_rate: Optional[float] = None,
    ):
        if sampling_rate is None:
            sampling_rate = self.sampling_rate

        if sampling_rate != self.sampling_rate:
            self.sampling_rate = sampling_rate

            self.step_ref_exc = int(self.lif_exc_tau_ref * self.sampling_rate)
            self.step_ref_inh = int(self.lif_inh_tau_ref * self.sampling_rate)

            self.band_pass_window_array = []
            self.low_pass_window_array = []
            for i in range(self.num_neurons):
                self.band_pass_window_array.append(
                    butter(
                        N=2,
                        Wn=[self.frequencies_bins[i], self.frequencies_bins[i + 1]],
                        btype="bandpass",
                        fs=sampling_rate,
                        output="sos",
                    )
                )
                self.low_pass_window_array.append(
                    butter(N=2, Wn=100, btype="lowpass", fs=sampling_rate, output="sos")
                )

        audio_bands = utils.bandpass_signal(
            num_bands=self.num_neurons,
            audio=audio,
            band_pass_window_array=self.band_pass_window_array,
            low_pass_window_array=self.low_pass_window_array,
        )

        @jax.jit
        def body_fun(carry, input_val):
            (
                lif_mem_exc,
                count_threshold_exc,
                lif_mem_inh,
                count_threshold_inh,
                I_exc,
                I_inh,
            ) = carry

            spikes_exc = ((lif_mem_exc >= self.lif_exc_threshold).astype(int)) * (
                count_threshold_exc.astype(int) >= self.step_ref_exc
            ).astype(
                int
            )  # Generate spikes based on thresholds and refractory period (in steps)

            count_threshold_exc = (count_threshold_exc + 1) * (
                1 - spikes_exc
            )  # Reset counter if spike occurred

            spikes_inh = ((lif_mem_inh >= self.lif_inh_threshold).astype(int)) * (
                count_threshold_inh.astype(int) >= self.step_ref_inh
            ).astype(
                int
            )  # Generate spikes based on thresholds and refractory period (in steps)

            count_threshold_inh = (count_threshold_inh + 1) * (
                1 - spikes_inh
            )  # Reset counter if spike occurred

            I_exc = I_exc * (
                1 - 1 / (self.lif_exc_tau_synapse * self.sampling_rate)
            ) + (
                input_val * self.weight_in
                + self.weight_ee * jnp.sum(spikes_exc)
                - self.weight_ei * jnp.sum(spikes_inh)
            )

            I_inh = I_inh * (
                1 - 1 / (self.lif_inh_tau_synapse * self.sampling_rate)
            ) + (
                0
                - self.weight_ii * jnp.sum(spikes_inh)
                + self.weight_ie * jnp.sum(spikes_exc)
            )

            lif_mem_exc = lif_mem_exc + (
                (-lif_mem_exc / self.lif_exc_tau_membrane + I_exc + self.lif_exc_idc)
                / self.sampling_rate
            )
            lif_mem_inh = lif_mem_inh + (
                (-lif_mem_inh / self.lif_inh_tau_membrane + I_inh + self.lif_inh_idc)
                / self.sampling_rate
            )

            return (
                lif_mem_exc,
                count_threshold_exc,
                lif_mem_inh,
                count_threshold_inh,
                I_exc,
                I_inh,
            ), (spikes_exc, spikes_inh)

        initial_carry = (
            jnp.zeros((self.num_neurons,)),
            jnp.zeros((self.num_neurons,)),
            jnp.zeros((self.num_neurons,)),
            jnp.zeros((self.num_neurons,)),
            jnp.zeros((self.num_neurons,)),
            jnp.zeros((self.num_neurons,)),
        )
        _, out_spikes = jax.lax.scan(body_fun, initial_carry, audio_bands.T)

        time_ax = jnp.linspace(0, len(audio) / sampling_rate, len(audio))
        event_v, address_v = jnp.where(out_spikes != 0)
        event_time = time_ax[event_v]
        event_address = address_v

        return event_time, event_address
