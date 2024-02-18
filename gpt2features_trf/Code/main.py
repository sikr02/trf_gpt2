import datetime
import logging
import os
import pickle
from functools import reduce
# MP
import time
from multiprocessing.pool import Pool
# Scientific libraries
from scipy.stats import zscore
import mne
import numpy as np
# PyEEG
from pyeeg.io import loadmat
from pyeeg.models import TRFEstimator, mem_check
# Custom files
from utils import TRFResult, TTestResult, preload_eeg, load_features, remaining_time

# Set logging level
logging.getLogger().setLevel('ERROR')
mne.set_log_level('ERROR')

# resampling to 62.5 Hz because of memory usage
S_FREQ = 62.5

# paths
PATH_TO_EEG = os.path.join(os.path.split(__file__)[0], "../sample_data/")
PATH_TO_DATA = os.path.join(os.path.split(__file__)[0], "../data/")
PATH_TO_RESULTS = os.path.join(os.path.split(__file__)[0], "../results/")
PATH_TO_LOG = os.path.join(os.path.split(__file__)[0], "../log/")


# Exception for handling keyboard interrupt
class KeyboardInterruptWorkerException(Exception):
    """
    Convert interrupt to exception, so it can be handled in the main process
    """
    pass


def init_trf():
    """
    Preload EEG data and set up the TRFEstimator
    """

    # ------------------------------------------------------------------------------------------------------------------
    # EEG DATA ---------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    path_to_preload = os.path.join(PATH_TO_DATA, "eeg_data_preload.pkl")
    # Contains list of EEG data from each subject
    with open(path_to_preload, "rb") as f:
        Y = pickle.load(f)

    Y = np.asarray(Y)

    # ------------------------------------------------------------------------------------------------------------------
    # SETUP ------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    trf = TRFEstimator(alpha=0., tmin=-.4, tmax=1.5, srate=S_FREQ)

    # Global tuple containing data that will be needed in the worker process later
    global data
    data = (Y, trf)


def compute_trf(feat: tuple) -> tuple:
    """
    Executes a feature computation with TRF

    :param feat: tuple containing the name of the new feature and the feature values of shape (15, #words, 1)
    :return: tuple with a TRFResult object and the duration of the computation
    """
    # Try-catch block to catch keyboard interrupts
    try:
        feat_name, feature = feat
        print(f"Starting feature {feat_name}", flush=True)

        # Measuring time
        start_time = time.time()

        global data
        Y, trf = data

        # Load features word onset, surprisal, wordfrequency and a new feature defined in "feat"
        feature_names_for_estimator, all_speech = load_features(path_to_eeg=PATH_TO_EEG,
                                                                sfreq=S_FREQ,
                                                                new_feat_name=feat_name,
                                                                new_feat_values=feature)

        # Concatenate all and extract numpy array
        X = reduce(lambda x, y: x + y, all_speech).feats.get_values()

        # Normalize features except of word onset
        X[X[:, 0] != 0, 1:] = zscore(X[X[:, 0] != 0, 1:])

        corr_coefs = []

        # Coefficients for each subject
        corr_coefs_lists = [[] for _ in range(len(Y))]

        # --------------------------------------------------------------------------------------------------------------
        # CROSS-VALIDATION ---------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # Split dataset in train and test
        split = int(0.2 * len(X))
        for i in range(5):  # Cross-validation
            # Test data: 20% of dataset
            Xtest = X[i * split:(i + 1) * split]
            Ytest = Y[:, i * split:(i + 1) * split]

            # Train data: 80% of dataset
            # Combine the parts to a coherent array
            Xtrain = np.concatenate((X[0:i * split], X[(i + 1) * split:]))
            Ytrain = np.concatenate((Y[:, 0:i * split], Y[:, (i + 1) * split:]), axis=1)

            # ----------------------------------------------------------------------------------------------------------
            # COMPUTING TRF --------------------------------------------------------------------------------------------
            # ----------------------------------------------------------------------------------------------------------
            print(f"{feat_name}, cross-validation round {i + 1}", flush=True)
            trf.fit(feat_names=feature_names_for_estimator, drop=True, X=Xtrain, y=Ytrain)

            for s, subj_y in enumerate(Ytest):
                c = trf.score(Xtest, subj_y)
                corr_coefs.append(c)

                corr_coefs_lists[s].append(c)

        # --------------------------------------------------------------------------------------------------------------
        # CORRELATION --------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # Total correlation over all subjects
        corr_coefs = np.asarray(corr_coefs)
        corr_avg = np.mean(corr_coefs, axis=0)
        total_correlation = np.mean(corr_avg)

        # Correlation for subjects
        corr_coefs_crossvalidation = np.asarray(corr_coefs_lists)
        corr_coefs_channels = np.mean(corr_coefs_crossvalidation, axis=1)
        corr_coefs_subjects = np.mean(corr_coefs_channels, axis=1)

        print(f"Completed feature {feat_name}", flush=True)

        # --------------------------------------------------------------------------------------------------------------
        # RETURN CORRELATION AND TIME ----------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # Measuring time
        time_passed = time.time() - start_time
        m, s = int((time_passed % 3600) // 60), int(time_passed % 60)

        # Return tuple containing feature names, resulting correlation, correlation for each subject and the passed time
        result = TRFResult(feat_names=feature_names_for_estimator,
                           total_avg_corr=float(total_correlation),
                           subj_corr=corr_coefs_subjects,
                           channel_corr=corr_avg,
                           trf_estimator=trf)
        return result, m, s

    # ------------------------------------------------------------------------------------------------------------------
    # EXCEPTION --------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # Convert interrupt to exception
    except KeyboardInterrupt:
        raise KeyboardInterruptWorkerException()


def main():
    # obligatory files for the computation
    assert "gpt2_feature_list.pkl" in os.listdir(PATH_TO_DATA), "Generate features first with gpt2features_generate.py!"
    assert "reference.pkl" in os.listdir(PATH_TO_DATA), "No reference values for statistical t-test!"

    # Create directories if necessary
    for d in [PATH_TO_LOG, PATH_TO_RESULTS]:
        if not os.path.isdir(d):
            os.makedirs(d)

    # Reference
    with open(os.path.join(PATH_TO_DATA, "reference.pkl"), "rb") as f:
        ref = pickle.load(f)
    subj_ref = ref.subj_corr

    # ------------------------------------------------------------------------------------------------------------------
    # PATHS ------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # EEG
    list_subjects = [subj for subj in os.listdir(PATH_TO_EEG) if subj.startswith("P")]

    # Audio files
    audio_path = lambda s: os.path.join(PATH_TO_EEG, "stories", "story_parts", "alignment_data", s, s + ".wav")

    # Speech transcript
    path_transcripts = os.path.join(PATH_TO_EEG, "stories", "story_parts", "transcripts")
    list_stories = [s.rstrip(".txt") for s in os.listdir(path_transcripts) if s.endswith("txt")]

    # Onsets of story parts for each participant
    onsets = loadmat(os.path.join(PATH_TO_EEG, "onsets.mat"))["onsets"]

    # Sort all lists (alphabetical order):
    for unsorted_list in [list_subjects, list_stories]:
        unsorted_list.sort()

    # ------------------------------------------------------------------------------------------------------------------
    # EEG --------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # check for preloaded EEG
    if "eeg_data_preload.pkl" not in os.listdir(PATH_TO_DATA):
        print("No preload found. Loading data...")
        preload_eeg(path_to_data=PATH_TO_DATA,
                    path_to_eeg=PATH_TO_EEG,
                    list_subjects=list_subjects,
                    list_stories=list_stories,
                    onsets=onsets,
                    audio_path=audio_path,
                    sfreq=S_FREQ)

    # Multiprocessing
    n_processes = mem_check() // 4.5

    with open(os.path.join(PATH_TO_DATA, "gpt2_feature_list.pkl"), "rb") as f:
        tasks = pickle.load(f)

    computed_tasks = [t.rstrip(".pkl") for t in os.listdir(PATH_TO_RESULTS)]
    tasks = [t for t in tasks if t[0] not in computed_tasks]

    print(f"Computing {len(tasks)} features with {n_processes} processes:")
    completed_tasks = 0

    with Pool(initializer=init_trf, processes=n_processes) as pool:
        try:
            for res in pool.imap_unordered(func=compute_trf, iterable=tasks):
                result_obj, passed_min, passed_sec = res

                completed_tasks += 1
                h, m, s = remaining_time(remaining_iters=(len(tasks) - completed_tasks) // n_processes,
                                         time_per_iter=passed_min * 60 + passed_sec)

                ttest_res = TTestResult(trf_result=result_obj,
                                        subj_ref=subj_ref)
                with open(os.path.join(PATH_TO_RESULTS, f"{result_obj.feat_names[-1]}.pkl"), "wb") as f:
                    pickle.dump(ttest_res, f)

                with open(os.path.join(PATH_TO_LOG, f"{datetime.date.today()}.txt"), "a") as f:
                    f.write(f"{result_obj.feat_names[-1]}\t{passed_min:02}:{passed_sec:02}\t"
                            f"(Estimated runtime: {h:02}:{m:02}:{s:02})\n")
            pool.close()

        # Handle Ctrl+C interrupt
        except (KeyboardInterruptWorkerException, KeyboardInterrupt):
            print("Computation aborted. Terminating the pool...", flush=True)
            pool.terminate()
        finally:
            pool.join()


if __name__ == "__main__":
    main()
