import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
from jaxtyping import Array, Float
from scipy.signal import butter, lfilter, windows

SAMPLING_RATE = 44100 * 100  # Default sampling rate


def adaptive_sigma_delta_spikes(
    audio: float, feedback_gain: float, threshold: float
) -> int:
    @jax.jit
    def run(state, jj):
        ii, mem_previous, spike_previous = state  # Unpack integrator and memory state

        ii = (
            ii + jj / SAMPLING_RATE - spike_previous * feedback_gain
        )  # First integrator stage (accumulate input)
        mem = (
            mem_previous + ii / SAMPLING_RATE
        )  # Membrane potential update using neuron gain

        # Generate spike if membrane potential crosses thresholds aeset membrane potential if spike occurred
        spike = (mem >= threshold).astype(int) - (mem <= -threshold).astype(int)
        mem = (1 - jnp.abs(spike)) * mem  # Reset membrane potential if spike occurred

        return (ii, mem, spike), spike

    ii = 0
    _, out_spikes = jax.lax.scan(run, (ii, 0, 0), audio)

    return out_spikes


def constant_sigma_delta_spike(input_signal, threshold=10, feedback_gain=1.0):
    def body_fun(carry, input_val):
        (integrator, mem_p, mem_n) = carry
        quantized = jnp.sign(integrator)

        mem_p = jnp.where(quantized == 1, mem_p + 1 / SAMPLING_RATE, mem_p)
        mem_n = jnp.where(quantized == -1, mem_n + 1 / SAMPLING_RATE, mem_n)

        spikes_p = jnp.where(mem_p >= threshold, 1, 0)
        spikes_n = jnp.where(mem_n >= threshold, 1, 0)

        mem_p = jnp.where(spikes_p == 1, 0, mem_p)
        mem_n = jnp.where(spikes_n == 1, 0, mem_n)

        spikes = spikes_p - spikes_n

        new_integrator = integrator + input_val / SAMPLING_RATE - feedback_gain * spikes

        return (new_integrator, mem_p, mem_n), spikes

    initial_carry = (0.0, 0.0, 0.0)
    _, modulated_signal = jax.lax.scan(body_fun, initial_carry, input_signal)
    return modulated_signal


def edge_sigma_delta(input_signal, feedback_gain=1.0, threshold=0.1):
    def body_fun(carry, input_val):
        (integrator, prev_quant) = carry

        # non_linear = jnp.tanh(integrator)
        # quantized = jnp.sign(integrator)
        quantized = jnp.sign(integrator + threshold * prev_quant)
        spk = jnp.where(quantized == prev_quant, 0, quantized)

        new_integrator = (
            integrator + (input_val - feedback_gain * quantized) / SAMPLING_RATE
        )

        return (new_integrator, quantized), spk

    initial_carry = (0.0, 0.0)
    _, modulated_signal = jax.lax.scan(body_fun, initial_carry, input_signal)
    return modulated_signal


def edge_sigma_delta_second(input_signal, feedback_gain=1.0, threshold=0.1):
    def body_fun(carry, input_val):
        (state, prev_quant) = carry
        (first_integrator, second_integrator) = state
        quantized = jnp.sign(second_integrator + threshold * prev_quant)
        spk = jnp.where(quantized == prev_quant, 0, quantized)

        new_first_integrator = (
            first_integrator + (input_val - quantized) / SAMPLING_RATE
        )
        new_second_integrator = (
            second_integrator
            + (new_first_integrator * feedback_gain - quantized) / SAMPLING_RATE
        )
        new_state = (new_first_integrator, new_second_integrator)
        return (new_state, quantized), spk

    initial_carry = ((0.0, 0.0), 0.0)
    _, modulated_signal = jax.lax.scan(body_fun, initial_carry, input_signal)
    return modulated_signal


def event_reconstruction(modulated_signal):
    def body_fun(signal, input_val):
        signal = jnp.where(input_val == 0, signal, input_val)

        return signal, signal

    initial_carry = modulated_signal[0]
    _, reconstructed_signal = jax.lax.scan(body_fun, initial_carry, modulated_signal)
    return reconstructed_signal


def async_sigma_delta(input_signal, feedback_gain=1.0, threshold=0.1):
    def body_fun(carry, input_val):
        integrator, prev_quant = carry
        quantized = jnp.sign(integrator + threshold * prev_quant)

        new_integrator = (
            integrator + (input_val - feedback_gain * quantized) / SAMPLING_RATE
        )

        return (new_integrator, quantized), quantized

    initial_carry = (0.0, 0.0)
    _, modulated_signal = jax.lax.scan(body_fun, initial_carry, input_signal)
    return modulated_signal


def amplitude_to_frequency(
    audio: Float[Array, "#time"], max_frequency: float = 1e3
) -> Float[Array, "#time"]:
    # data_normalized = data_interpolated / jnp.max(jnp.abs(data_interpolated))  # Normalize the audio signal
    random_number = jax.random.uniform(jax.random.PRNGKey(0), shape=audio.shape)
    spikes_p = random_number < (audio * max_frequency / SAMPLING_RATE)
    spikes_n = random_number < (-audio * max_frequency / SAMPLING_RATE)
    spikes = spikes_p.astype(int) - spikes_n.astype(int)

    return spikes


# Low-pass filter to reconstruct the signal
def low_pass_filter(modulated_signal, window_size=301, gain=4.65415):
    kernel = windows.flattop(window_size) / window_size  # Create a Hann window kernel
    filtered_signal = jax.scipy.signal.convolve(
        modulated_signal * gain, kernel, mode="same"
    )

    return filtered_signal


def butter_lowpass_filter(modulated_signal, cutoff=30e-3, fs=SAMPLING_RATE, order=4):
    b, a = butter(order, Wn=cutoff, fs=fs, btype="low", analog=False)
    filtered_signal = lfilter(b, a, modulated_signal)
    return filtered_signal


def getSignals(
    threshold=1,
    window_size=301,
    feedback_gain=1,
    max_frequency=1e3,
    lowpass_gain=4.65415,
    total_time=1,
    sampling_rate=44100 * 100,
    mode="spike",
):
    # Example usage
    n_samples = int(sampling_rate * total_time)

    t = jnp.linspace(0, total_time, n_samples, endpoint=False)
    input_signal = 0.5 * jnp.sin(2 * jnp.pi * 7 * t) + 0.5 * jnp.sin(
        2 * jnp.pi * 10e3 * t
    )
    # input_signal = jnp.sin(2 * jnp.pi * 5 * t)
    # input_signal = jnp.where(t < 0.01, 0.5, 0.0)  # Step function for testing
    # input_signal = 1 * scipy.signal.chirp(t, 1, total_time, 20e3, method="logarithmic")

    if mode == "adaptive":
        modulated_signal = adaptive_sigma_delta_spikes(
            input_signal, feedback_gain=feedback_gain, threshold=threshold
        )
        filtered_signal = low_pass_filter(
            modulated_signal, window_size=window_size, gain=lowpass_gain
        )
        # filtered_signal = butter_lowpass_filter(modulated_signal, cutoff=30e-3, fs=SAMPLING_RATE, order=4)
    elif mode == "constantine":
        modulated_signal = constant_sigma_delta_spike(
            input_signal, feedback_gain=feedback_gain, threshold=threshold
        )
        filtered_signal = low_pass_filter(
            modulated_signal, window_size=window_size, gain=lowpass_gain
        )
        # filtered_signal = butter_lowpass_filter(modulated_signal, cutoff=30e-3, fs=SAMPLING_RATE, order=4)
    elif mode == "edge":
        modulated_signal = edge_sigma_delta(
            input_signal, feedback_gain=feedback_gain, threshold=threshold
        )
        reconstructed_signal = event_reconstruction(modulated_signal)
        filtered_signal = low_pass_filter(
            reconstructed_signal, window_size=window_size, gain=lowpass_gain
        )
        # filtered_signal = butter_lowpass_filter(reconstructed_signal, cutoff=30e-3, fs=SAMPLING_RATE, order=4)
    elif mode == "async":
        modulated_signal = async_sigma_delta(
            input_signal, feedback_gain=feedback_gain, threshold=threshold
        )
        filtered_signal = low_pass_filter(
            modulated_signal, window_size=window_size, gain=lowpass_gain
        )
        # filtered_signal = butter_lowpass_filter(modulated_signal, cutoff=30e-3, fs=SAMPLING_RATE, order=4)
    elif mode == "freq":
        modulated_signal = amplitude_to_frequency(
            input_signal, max_frequency=max_frequency
        )
        filtered_signal = low_pass_filter(
            modulated_signal, window_size=window_size, gain=lowpass_gain
        )
        # filtered_signal = butter_lowpass_filter(modulated_signal, cutoff=30e-3, fs=SAMPLING_RATE, order=4)
    elif mode == "second_order_edge":
        modulated_signal = edge_sigma_delta_second(
            input_signal, feedback_gain=feedback_gain, threshold=threshold
        )
        reconstructed_signal = event_reconstruction(modulated_signal)
        filtered_signal = low_pass_filter(
            reconstructed_signal, window_size=window_size, gain=lowpass_gain
        )

    return input_signal, modulated_signal, filtered_signal


def getSNR(input_signal, filtered_signal, return_noise=False):
    # input_signal = input_signal / jnp.max(jnp.abs(input_signal))  # Normalize the input signal
    # filtered_signal = filtered_signal / jnp.max(jnp.abs(filtered_signal))  # Normalize
    noise = input_signal - filtered_signal
    power_original = jnp.mean(input_signal**2)
    power_noise = jnp.mean(noise**2)

    # Add a small epsilon to avoid division by zero if power_noise is zero
    epsilon = 1e-15
    snr = 10 * jnp.log10(power_original / (power_noise + epsilon))

    if return_noise:
        return snr, noise
    else:
        return snr


def objective(trial, mode="constantine", window_size=301, lowpass_gain=4.65415):
    threshold = None
    window_size = None
    feedback_gain = None
    max_frequency = None
    lowpass_gain = None

    if mode == "freq":
        max_frequency = trial.suggest_float("max_frequency", 1e3, 10e6, log=True)
    elif (mode == "edge") or (mode == "async"):
        feedback_gain = 1  # trial.suggest_float('feedback_gain', 0.8, 2, log=True)
        threshold = trial.suggest_float("threshold", 1e-12, 1e-4, log=True)
    elif mode == "adaptive":
        # feedback_gain = trial.suggest_float('feedback_gain', 1e-7, 1e-6, log=True)
        # threshold = trial.suggest_float('threshold', 1e-18, 1e-9, log=True)
        feedback_gain = trial.suggest_float("feedback_gain", 1e-12, 1, log=True)
        threshold = trial.suggest_float("threshold", 1e-18, 1, log=True)
    elif mode == "constantine":
        # feedback_gain = trial.suggest_float('feedback_gain', 1e-7, 1e-6, log=True)
        # threshold = trial.suggest_float('threshold', 1e-18, 1e-12, log=True)
        feedback_gain = trial.suggest_float("feedback_gain", 1e-7, 1e-6, log=True)
        threshold = trial.suggest_float("threshold", 1e-18, 1e-4, log=True)
    elif mode == "second_order_edge":
        feedback_gain = trial.suggest_float("feedback_gain", 1e3, 1e12, log=True)
        threshold = trial.suggest_float("threshold", 1e-18, 1e-6, log=True)
    else:
        feedback_gain = trial.suggest_float("feedback_gain", 1e-16, 1e-3, log=True)
        threshold = trial.suggest_float("threshold", 1e-8, 1e-4, log=True)

    window_size = 301  # trial.suggest_int('window_size', 10, 1e3)
    lowpass_gain = 4.65415  # 1#trial.suggest_float('lowpass_gain', 1e-3, 1e3)

    input_signal, modulated_signal, filtered_signal = getSignals(
        threshold=threshold,
        window_size=window_size,
        feedback_gain=feedback_gain,
        max_frequency=max_frequency,
        lowpass_gain=lowpass_gain,
        total_time=1,
        sampling_rate=SAMPLING_RATE,
        mode=mode,
    )
    snr = getSNR(input_signal, filtered_signal, return_noise=False)

    if mode == "async":
        return snr
    else:
        firing_frequency = jnp.mean(jnp.abs(modulated_signal)) * SAMPLING_RATE
        return snr, firing_frequency


def getParetoReferencePoint(study, margin=0.1):
    values = [trial.values for trial in study.best_trials if trial.values is not None]
    objectives = jnp.array(values)
    reference_point = []
    for i, direction in enumerate(study.directions):
        if direction == optuna.study.StudyDirection.MINIMIZE:
            worst = jnp.max(objectives[:, i])
            reference_point.append(worst * (1 + margin))
        else:  # MAXIMIZE
            worst = jnp.min(objectives[:, i])
            reference_point.append(worst * (1 - margin))
    return reference_point


def _get_pareto_front_2d(info_spike, info_event):
    # Set up the graph style.
    plt.style.use("ggplot")  # Use ggplot style sheet for similar outputs to plotly.
    _, ax = plt.subplots()
    ax.set_title("Pareto-front Plot")
    cmap = plt.get_cmap("tab10")  # Use tab10 colormap for similar outputs to plotly.

    ax.set_xlabel(info_spike.target_names[info_spike.axis_order[0]])
    ax.set_ylabel(info_spike.target_names[info_spike.axis_order[1]])

    if len(info_spike.best_trials_with_values) > 0:
        ax.scatter(
            x=[
                values[info_spike.axis_order[0]]
                for _, values in info_spike.best_trials_with_values
            ],
            y=[
                values[info_spike.axis_order[1]]
                for _, values in info_spike.best_trials_with_values
            ],
            color=cmap(1),
            label="Best Trial Spike",
        )

    if len(info_event.best_trials_with_values) > 0:
        ax.scatter(
            x=[
                values[info_event.axis_order[0]]
                for _, values in info_event.best_trials_with_values
            ],
            y=[
                values[info_event.axis_order[1]]
                for _, values in info_event.best_trials_with_values
            ],
            color=cmap(2),
            label="Best Trial Event",
        )

    if ax.has_data():
        ax.legend()

    return ax


def _get_pareto_front_plot(info_spike, info_event):
    from optuna.visualization._pareto_front import _make_scatter_object
    from optuna.visualization._plotly_imports import go

    data = [
        _make_scatter_object(
            info_spike.n_targets,
            info_spike.axis_order,
            info_spike.include_dominated_trials,
            info_spike.best_trials_with_values,
            hovertemplate="%{text}<extra>Best Trial Spike</extra>",
            dominated_trials=False,
        ),
        _make_scatter_object(
            info_event.n_targets,
            info_event.axis_order,
            info_event.include_dominated_trials,
            info_event.best_trials_with_values,
            hovertemplate="%{text}<extra>Best Trial Event</extra>",
            dominated_trials=False,
        ),
    ]

    layout = go.Layout(
        title="Pareto-front Plot",
        xaxis_title=info_spike.target_names[info_spike.axis_order[0]],
        yaxis_title=info_spike.target_names[info_spike.axis_order[1]],
    )
    return go.Figure(data=data, layout=layout)


if __name__ == "__main__":
    import argparse
    from functools import partial

    import optuna
    from optuna.visualization._pareto_front import _get_pareto_front_info
    from plotly.io import show

    parser = argparse.ArgumentParser(
        prog="ProgramName",
        description="Sigma-Delta Modulation Parameter Selection",
        epilog="Choose one mode: optimize, use best, or custom parameters.",
    )
    parser.add_argument(
        "--study_name", type=str, default="sigma_delta", help="Name of the Optuna study"
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///sigma_delta_study.db",
        help="Storage for the Optuna study",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=100,
        help="Number of trials for the Optuna study",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--optimize", action="store_true", help="Optimize the parameters using Optuna"
    )
    group.add_argument(
        "--use-best", action="store_true", help="Use best parameters from Optuna study"
    )
    group.add_argument(
        "--comparison",
        action="store_true",
        help="Comparison between edge, constantine, and adaptive-based sigma-delta modulation",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "adaptive",
            "constantine",
            "edge",
            "async",
            "freq",
            "second_order_edge",
        ],
        default="constantine",
        help="Mode of sigma-delta modulation (spike, event or continuous)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-3,
        help="Threshold for sigma-delta modulation",
    )
    parser.add_argument(
        "--window-size", type=int, default=301, help="Window size for low-pass filter"
    )
    parser.add_argument(
        "--max-frequency",
        type=float,
        default=1000,
        help="Maximum frequency for frequency-based modulation",
    )
    parser.add_argument(
        "--feedback-gain",
        type=float,
        default=1,
        help="Feedback gain for sigma-delta event-based modulation",
    )
    parser.add_argument(
        "--lowpass-gain",
        type=float,
        default=4.65415,
        help="Feedback gain for sigma-delta event-based modulation",
    )
    args = parser.parse_args()

    if args.optimize:
        if args.mode == "async":
            study = optuna.create_study(
                direction="maximize",
                study_name=args.study_name + "_" + args.mode,
                storage=args.storage,
                load_if_exists=True,
            )
            study.optimize(partial(objective, mode=args.mode), n_trials=args.n_trials)
            optuna.visualization.plot_param_importances(study).show()
        else:
            study = optuna.create_study(
                directions=["maximize", "minimize"],
                study_name=args.study_name + "_" + args.mode,
                storage=args.storage,
                load_if_exists=True,
            )
            study.optimize(partial(objective, mode=args.mode), n_trials=args.n_trials)
            reference_point = getParetoReferencePoint(study)
            print(f"Number of trials on the Pareto front: {len(study.best_trials)}")
            print(f"Pareto reference point: {reference_point}")
            optuna.visualization.plot_param_importances(study).show()
            optuna.visualization.plot_pareto_front(
                study, target_names=["SNR", "Firing rate"]
            ).show()
            fig = optuna.visualization.plot_hypervolume_history(study, reference_point)
            show(fig)

        # best_params = study.best_params
        # threshold = best_params['threshold']
        # window_size = best_params['window_size']
        # print(f"Best parameters found: threshold={threshold}, window_size={window_size}")
    elif args.use_best:
        if args.mode != "async":
            study = optuna.load_study(
                study_name=args.study_name + "_" + args.mode, storage=args.storage
            )
            reference_point = getParetoReferencePoint(study)
            print(f"Number of trials on the Pareto front: {len(study.best_trials)}")
            print(f"Pareto reference point: {reference_point}")
            optuna.visualization.plot_param_importances(study).show()
            optuna.visualization.plot_pareto_front(
                study, target_names=["SNR", "Firing rate"]
            ).show()
            optuna.visualization.plot_hypervolume_history(study, reference_point).show()
        # best_params = study.best_params
        # threshold = best_params['threshold']
        # window_size = best_params['window_size']
        # print(f"Loaded best parameters: threshold={threshold}, window_size={window_size}")

    elif args.comparison:
        study_spike = optuna.load_study(
            study_name=args.study_name + "_" + "spike", storage=args.storage
        )
        study_event = optuna.load_study(
            study_name=args.study_name + "_" + "event", storage=args.storage
        )

        info_spike = _get_pareto_front_info(
            study_spike, target_names=["SNR", "Firing rate"]
        )
        info_event = _get_pareto_front_info(
            study_event, target_names=["SNR", "Firing rate"]
        )
        _get_pareto_front_2d(info_spike, info_event)
        plt.show()
        # _get_pareto_front_plot(info_spike, info_event).show()

    else:
        threshold = args.threshold
        window_size = args.window_size
        feedback_gain = args.feedback_gain
        max_frequency = args.max_frequency
        lowpass_gain = args.lowpass_gain
        print(
            f"Using custom/default parameters: threshold={threshold}, window_size={window_size}"
        )

        input_signal, modulated_signal, filtered_signal = getSignals(
            threshold=threshold,
            window_size=window_size,
            feedback_gain=feedback_gain,
            max_frequency=max_frequency,
            lowpass_gain=lowpass_gain,
            total_time=1,
            sampling_rate=SAMPLING_RATE,
            mode=args.mode,
        )

        snr, noise = getSNR(input_signal, filtered_signal, return_noise=True)
        print(f"Signal-to-Noise Ratio (SNR): {snr:.2f} dB")
        if args.mode != "async":
            firing_frequency = jnp.mean(jnp.abs(modulated_signal)) * SAMPLING_RATE
            print(f"Firing Frequency: {firing_frequency:.2f} Hz")

        # Plotting
        with mpl.style.context("boilerplot.ieeetran"):
            plt.figure(figsize=[6.45, 4.3], dpi=200, constrained_layout=True)
            plt.plot(input_signal, label="Original Signal", alpha=1.0)

            plt.plot(noise, label="Noise", alpha=0.3)

            if args.mode == "async":
                plt.plot(
                    modulated_signal, label="Modulated Signal", linewidth=1, alpha=0.5
                )
            else:
                modulated_signal_p = jnp.argwhere(modulated_signal == 1)
                modulated_signal_n = jnp.argwhere(modulated_signal == -1)
                plt.plot(
                    modulated_signal_p,
                    jnp.ones_like(modulated_signal_p),
                    "|",
                    label="Modulated Signal Postive",
                    alpha=1.0,
                    markersize=10,
                )
                plt.plot(
                    modulated_signal_n,
                    -jnp.ones_like(modulated_signal_n),
                    "|",
                    label="Modulated Signal Negative",
                    alpha=1.0,
                    markersize=10,
                )

            plt.plot(
                filtered_signal, label="Reconstructed Signal", linewidth=2, alpha=0.3
            )
            plt.ylim(-1.5, 1.5)
            plt.title("Sigma-Delta Modulation and Signal Reconstruction")
            plt.legend()
            plt.xlabel("Sample Index")
            plt.ylabel("Amplitude")
            plt.show()
