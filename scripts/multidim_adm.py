import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.signal import chirp

from crusafe.conversion_methods import standard_ADM

sr = 44100
duration = 0.1
t = jnp.arange(int(sr * duration)) / sr
audio1 = chirp(t, f0=20, f1=200, t1=duration, method="linear")
audio2 = chirp(t, f0=30, f1=200, t1=duration, method="linear")
audio3 = audio1 + audio2
audio = jnp.stack([audio1, audio2, audio3], axis=1)

fb = standard_ADM(sampling_rate=sr, threshold=0.8)
event_time, event_address = fb(audio)

fig, ax = plt.subplots(2, 1, figsize=(3.45, 2.3), dpi=200, sharex=True)
ax[0].plot(t, audio)
ax[1].scatter(x=event_time, y=event_address, s=6)
ax[1].set_ylim([-1, 6])
plt.show()
