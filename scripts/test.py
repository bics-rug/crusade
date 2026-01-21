import matplotlib.pyplot as plt

from crusade import conversion_methods, utils

# source .venv/bin/activate
# module load FFmpeg/7.1.1-GCCcore-14.2.0
# module load Python/3.9.6-GCCcore-11.2.0


train_dataset = utils.download_dataset()
frequency = 4410000
sample = 0
# region_dataset = (train_dataset[sample]["audio"]["path"]).split("_")[0]
# label_dataset = train_dataset[sample]["label"]
# print(f"Region: {region_dataset}, Label: {label_dataset}")
scaled_audio = utils.audio_resampling_and_scaling(
    train_dataset[sample]["audio"]["array"],
    original_frequency=train_dataset[sample]["audio"]["sampling_rate"],
    target_frequency=frequency,
)  # , scaling_factor="normalize")

# plt.plot(scaled_audio)
# plt.show()
# plt.savefig("scaled_audio.png")
# plt.clf()

neuron_model = conversion_methods.filterbank_ADM(frequency)
event_time, event_address = neuron_model(scaled_audio, frequency)
# event_time, event_address = neuron_model(scaled_audio, sampling_rate= frequncy)
plt.plot(event_time, event_address, ".")
plt.show()
# plt.savefig("event_plot.png")
plt.clf()
plt.plot(event_time[0:100], event_address[0:100], ".")
plt.show()
# plt.savefig("event_plot_100.png")
# print(event_time)
# print(event_address)
