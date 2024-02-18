import pickle
import os
import logging
# Scientific libraries
import numpy as np
import mne
# Plot library
import matplotlib.pyplot as plt
# PyEEG
from pyeeg.io import eeglab2mne
# Custom files
from utils import TTestResult, TRFEstimator

# Set logging level
logging.getLogger().setLevel("ERROR")
mne.set_log_level("ERROR")

# paths
PATH_TO_RESULTS = os.path.join(os.path.split(__file__)[0], "../results/")
PATH_TO_DATA = os.path.join(os.path.split(__file__)[0], "../../trf_gpt2/sample_data/")


def plot_topomap(data: np.ndarray, info, ax, sensor_names=False, cbar_label=None):
    """
    Plots topomap and with a colorbar

    :param data: The values to plot
    :param info: raw.info file for sensor names and positions
    :param ax: Axes to plot on
    :param sensor_names: Whether to plot names
    :param cbar_label: label for the colorbar
    """
    if sensor_names:
        image, _ = mne.viz.plot_topomap(data, info, sensors=False, contours=0, axes=ax, show=False, show_names=True,
                                        names=info["ch_names"])
    else:
        image, _ = mne.viz.plot_topomap(data, info, contours=0, axes=ax, show=False)
    plt.colorbar(image, ax=ax, label=cbar_label)


def plot_trf_topomap(trf: TRFEstimator, info, ax, sensor_names=False, max_span=None):
    """
    Plot TRF results of the maximum peak as a topomap

    :param trf: the TRFEstimator object containing the results
    :param info: info from a raw file for channel positions
    :param ax: axes to plot the topomap
    :param sensor_names: if True, channel names are plotted, if False, dots at sensor positions are used
    :param max_span: time span in which the maximum peak for the topomap plot is searched
    """
    for feat in range(trf.n_feats_):
        # max index
        zero_offset = np.argwhere(trf.times >= 0)[0, 0]  # index closest to zero

        if max_span is not None:
            assert len(max_span) == 2, "Search span for maximum has to be an interval: 2 values have to be provided!"
        max_span_start = zero_offset if max_span is None else max(zero_offset,
                                                                  zero_offset + int(max_span[0] * info["sfreq"]))
        max_span_end = len(trf.times) if max_span is None else min(len(trf.times),
                                                                   zero_offset + int(max_span[1] * info["sfreq"]))

        abs_mean = np.mean(np.abs(trf.coef_[:, feat, :]), axis=1)
        idx = zero_offset + abs_mean[max_span_start:max_span_end].argmax()

        # region where the absolute amplitude is greater than mean + std
        indices = np.where(abs_mean > (np.mean(abs_mean) + np.std(abs_mean)))

        # find neighbor indices and estimating spans
        ind = indices[0]
        spans = []
        start = 0
        end = 0
        for i in range(1, len(ind)):
            if ind[i] == ind[end] + 1:
                end = i
            else:
                spans.append((ind[start], ind[end]))
                start = i
                end = i
        spans.append((ind[start], ind[end]))

        # estimate the span where the maximum lies in
        span_to_plot = [s for s in spans if idx in np.arange(s[0], s[1])][0]

        # use it for topomap because of greatest activity
        data_to_plot = np.mean(trf.coef_[span_to_plot[0]:span_to_plot[1]][feat, :], axis=0)

        plot = ax[0, feat] if len(trf.feat_names_) > 1 else ax[feat]

        # plot the absolute mean
        plot.plot(trf.times, abs_mean, c="black", lw=2.5)

        # highlight the spans
        for s, e in spans:
            # grey region highlight
            plot.axvspan(trf.times[s], trf.times[e], color="gray", alpha=0.3)
        # red maximum line
        plot.axvspan(trf.times[idx], trf.times[idx], color="red", alpha=0.3)
        # green plot region
        # plot.axvspan(trf.times[span_to_plot[0]], trf.times[span_to_plot[1]], color="green", alpha=0.2)

        # plot the topomap
        plot_topomap(data=data_to_plot,
                     info=info,
                     ax=ax[1, feat] if len(trf.feat_names_) > 1 else ax[feat + 1],
                     sensor_names=sensor_names,
                     cbar_label="TRF coefficients")


def main():
    # TODO: define computed feature name => has to be in the results folder (computed by main.py)
    filename_to_plot = "gpt2_L0_d477"

    # ------------------------------------------------------------------------------------------------------------------
    # SETUP ------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # Check if file exists
    assert filename_to_plot + ".pkl" in os.listdir(PATH_TO_RESULTS), \
        f"Feature {filename_to_plot} has to be computed first!"

    # with open("C:/Users/Simon/PycharmProjects/gpt2features_trf/data/reference.pkl", "rb") as f:
    with open(os.path.join(PATH_TO_RESULTS, filename_to_plot + ".pkl"), "rb") as f:
        data: TTestResult = pickle.load(f)

    # trf = data
    trf = data.trf_result

    # Loading any raw file for info
    raw = eeglab2mne(os.path.join(PATH_TO_DATA, "P14_21032017",
                                  "Fs-125-AllChannels-interp_bad-BP-0.3-65-ICA-Blink_pruned-P14_21032017_.set"))
    raw = raw.pick_types(eeg=True)
    raw = raw.resample(sfreq=trf.trf_estimator.srate, n_jobs=-1)
    # Filter EEG data in delta band
    raw = raw.filter(None, 4, n_jobs=-1)

    raw.plot_sensors(kind="topomap", ch_type="eeg", show_names=True, show=True)

    # ------------------------------------------------------------------------------------------------------------------
    # PLOT TRF BUTTERFLY AND TOPOMAP PLOT ------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    max_span = None

    f, axes = plt.subplots(2, trf.trf_estimator.n_feats_, figsize=(5 * trf.trf_estimator.n_feats_, 6))
    trf.trf_estimator.plot(spatial_colors=True, info=raw.info, ax=axes[0])
    for i in axes[0]:
        i.set_xlabel("Time in s")
        i.set_ylabel("TRF coefficients")
    plot_trf_topomap(trf=trf.trf_estimator, info=raw.info, ax=axes, sensor_names=False, max_span=max_span)
    plt.subplots_adjust(wspace=0.5, hspace=0.3)
    plt.show(block=True)

    # ------------------------------------------------------------------------------------------------------------------
    # PLOT CORRELATION AND T-TEST RESULTS ------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots()
    plot_topomap(data=trf.channel_corr, info=raw.info, ax=ax, sensor_names=True, cbar_label="Correlation coefficient")
    plt.title(f"Correlation across channels\n"
              f"Total correlation: {trf.total_avg_corr:.3} (increment: {data.inc}, p-value: {data.ttest_result[1]:.3})",
              fontsize=10)
    plt.show(block=True)


if __name__ == '__main__':
    main()
