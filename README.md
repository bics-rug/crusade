# CRUSADE
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18480360.svg)](https://doi.org/10.5281/zenodo.18480360)

Conversion of Raw-audio Using Spikes Analog Digital Encoders

This repository contains implementations of different analogue-to-spike converters for raw audio signal.

## Features

This project contains Python implementations of various methods and neural models used to transform audio signals into spike/event trains, using [JAX](https://github.com/google/jax). It includes:
- Standalone ADM.
- Filterbank with ADM.
- Filterbank with Resonate and Fire neurons.
- Filterbank with phase encoding (under development).

## Installation
Recommended version for installation is using [uv](https://docs.astral.sh/uv/)
```bash
uv sync --frozen --all-extras
```

Or with pip

```bash
pip install ".[dev]"
```

## Example

Example for use `filterbank_ADM`:

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.signal import chirp

from crusafe.conversion_methods import filterbank_ADM

sr = 44100
duration = 0.1
t = jnp.arange(int(sr * duration)) / sr
audio = chirp(t, f0=200, f1=2000, t1=duration, method="linear")

fb = filterbank_ADM(
    num_neurons=16, freq_min=200, freq_max=2000, freq_distribution="linear"
)
event_time, event_address, event_magnitude = fb(audio, sampling_rate=sr)

plt.figure()
plt.scatter(x=event_time, y=event_address, c=event_magnitude, cmap="bwr")
plt.show()
```

## Tests
- On pc

To check if the models are working:
```bash
uv run pytest
```
or if you are in the envirnoment

```bash
pytest
```

To check before commit:

```bash
uv run pre-commit run --all-files
```
to install the  pre-commit (only forst time):
```bash
uv run pre-commit install
```
once installed it runs automatically for every commit

- On git

It does automatically runs all the tests
