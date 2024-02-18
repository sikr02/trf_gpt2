import os
import pickle
from functools import reduce
# Scientific libraries
import numpy as np
from scipy.stats import zscore, ttest_rel
# PyEEG
from pyeeg.io import AlignedSpeech, WordLevelFeatures, loadmat, eeglab2mne
from pyeeg.models import TRFEstimator


class TRFResult:
    """
    Class to hold the correlation and the fitted TRF model
    """
    def __init__(self, feat_names: list, total_avg_corr: float, subj_corr: np.ndarray, channel_corr: np.ndarray,
                 trf_estimator: TRFEstimator):
        assert trf_estimator.fitted is True, \
            f"Cannot create result for feature {feat_names[-1]}: Model has to be fitted first!"
        assert len(feat_names) == trf_estimator.n_feats_, f"Cannot create result for feature {feat_names[-1]}: " \
                                                          f"Number of features doesn't match number of computed " \
                                                          f"coefficients! -> {len(feat_names)} != " \
                                                          f"{trf_estimator.n_feats_}"
        self.feat_names: list = feat_names
        self.total_avg_corr: float = total_avg_corr
        self.subj_corr: np.ndarray = subj_corr
        self.channel_corr: np.ndarray = channel_corr
        self.trf_estimator: TRFEstimator = trf_estimator

    def __repr__(self) -> str:
        return f"Result for {self.feat_names[-1]}"


class TTestResult:
    """
    Class to hold increment indicator and p-value of the paired t-test; additionally stores the TRFResult object
    """
    def __init__(self, trf_result: TRFResult, subj_ref: np.ndarray):
        self.trf_result = trf_result

        subj_new = trf_result.subj_corr
        self.inc = np.mean(subj_ref) < np.mean(subj_new)
        self.ttest_result = ttest_rel(subj_ref, subj_new)

    def __repr__(self) -> str:
        return f"Paired t-test result for feature {self.trf_result.feat_names[-1]}:\nIncrease: {self.inc} " \
               f"(p-value: {self.ttest_result[1]:.3})"


def preload_eeg(path_to_data: str,
                path_to_eeg: str,
                list_subjects: list,
                list_stories: list,
                onsets: dict,
                audio_path,
                sfreq: float):
    """
    Preload the EEG data. Will be stored in <path_to_data> folder as pickle file

    :param path_to_data: location of the data cache
    :param path_to_eeg: location of the eeg files
    :param list_subjects: list of the subjects to load
    :param list_stories: list of the stories to concatenate the data correctly
    :param onsets: list of onset times for the aligning of the story parts
    :param audio_path: path to the audio files to extract the duration
    :param sfreq: sampling frequency of the input data => correct alignment
    """
    Y = []  # will contain list of EEG data from each subject

    for subj_id, subject in enumerate(list_subjects):
        print(f"Loading EEG: subject {subject}")
        eeg_filename = [f for f in os.listdir(os.path.join(path_to_eeg, subject)) if f.endswith(".set")][0]
        raw = eeglab2mne(os.path.join(path_to_eeg, subject, eeg_filename))

        # Process data
        # raw = raw.set_eeg_reference() # already re-referenced to average
        raw = raw.pick_types(eeg=True)
        # resample because of memory usage
        raw = raw.resample(sfreq=sfreq, n_jobs=-1)
        # filter EEG data in delta band
        raw = raw.filter(None, 4, n_jobs=-1)

        # get indices of EEG samples relative to alphabetically ordered stories
        indices = reduce(lambda x, y: x + y,
                         [AlignedSpeech(onset=onsets[subj_id, k], srate=sfreq, path_audio=audio_path(story))
                          for k, story in enumerate(list_stories)]).indices
        Y.append(zscore(raw.get_data()[:, indices].T))
    Y = np.asarray(Y)

    # store preload
    with open(os.path.join(path_to_data, "eeg_data_preload.pkl"), "wb") as f:
        pickle.dump(Y, f)


def load_features(path_to_eeg: str, sfreq: float, new_feat_name: str, new_feat_values: list) -> tuple:
    """
    Load and combine speech features of all stories

    :param path_to_eeg: path to the eeg data
    :param sfreq: sampling frequency of the eeg data
    :param new_feat_name: name of the new feature
    :param new_feat_values: values of the new feature
    :return: tuple containing feature names as a list and a list of AlignedSpeech objects with the feature values
    """
    # ------------------------------------------------------------------------------------------------------------------
    # PATHS ------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # EEG
    list_subjects = [subj for subj in os.listdir(path_to_eeg) if subj.startswith("P")]

    # Audio files
    audio_path = lambda s: os.path.join(path_to_eeg, "stories", "story_parts", "alignment_data", s, s + ".wav")

    # Speech transcript
    path_transcripts = os.path.join(path_to_eeg, "stories", "story_parts", "transcripts")
    list_stories = [s.rstrip(".txt") for s in os.listdir(path_transcripts) if s.endswith(".txt")]
    # Surprisal
    path_surprisal = os.path.join(path_to_eeg, "stories", "story_parts", "surprisal")
    list_surprisal_files = [s for s in os.listdir(path_surprisal) if s.endswith("probs3.txt")]
    # Word frequency
    path_wordfreq = os.path.join(path_to_eeg, "stories", "story_parts", "word_frequencies")
    list_wordfreq_files = [s for s in os.listdir(path_wordfreq) if s.endswith("timed.csv")]

    # Onsets of story parts for each participant
    onsets = loadmat(os.path.join(path_to_eeg, "onsets.mat"))["onsets"]

    # Sort all lists (alphabetical order):
    for unsorted_list in [list_subjects, list_stories, list_surprisal_files, list_wordfreq_files]:
        unsorted_list.sort()

    # ------------------------------------------------------------------------------------------------------------------
    # LINGUISTIC FEATURES ----------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    all_speech = []
    feature_names_for_estimator = []

    for k, story in enumerate(list_stories):
        speech = AlignedSpeech(onset=onsets[0, k], srate=sfreq, path_audio=audio_path(story))
        speech.add_word_level_features(WordLevelFeatures(
            path_audio=audio_path(story),
            path_surprisal=os.path.join(path_surprisal, list_surprisal_files[k]),
            path_wordfrequency=os.path.join(path_wordfreq, list_wordfreq_files[k]),
            path_wordonsets=os.path.join(path_wordfreq, list_wordfreq_files[k]),
            path_transcript=os.path.join(path_transcripts, story + ".txt"),
        ), use_wordonsets=True)

        # add new feature
        values = np.zeros((len(speech),))
        values[np.argwhere(speech.feats.wordonsets)[:, 0]] = new_feat_values[k]
        speech.add_feature(values, new_feat_name)

        # add to list
        all_speech.append(speech)

        # save names for the estimator
        feature_names_for_estimator = speech.feats.columns.tolist()

    return feature_names_for_estimator, all_speech


def remaining_time(remaining_iters: int, time_per_iter: float) -> tuple:
    """
    Computes and formats the remaining time of computations

    :param remaining_iters: number of remaining computations
    :param time_per_iter: time of one iteration
    :return: tuple containing formatted time (hours, minutes, seconds)
    """
    rem_time = remaining_iters * time_per_iter
    return int(rem_time // 3600), int((rem_time % 3600) // 60), int(rem_time % 60)
