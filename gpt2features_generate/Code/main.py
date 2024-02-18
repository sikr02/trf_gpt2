import pickle
import os

from gpt2_utils import load_stories, extract_hidden_states
from preprocess import *


# transcript path
PATH_TO_TRANSCRIPTS = "../../gpt2features_trf/sample_data/stories/story_parts/alignment_data/"
# feature path
PATH_TO_FEATURES = "../../gpt2features_trf/data/"


def main():
    titles, stories = load_stories(os.path.split(__file__)[0], PATH_TO_TRANSCRIPTS)

    print("Load hidden states: ", end="")
    try:
        with open(os.path.join(os.path.split(__file__)[0], "hidden_states_preload.pkl"), "rb") as f:
            print("Using preloaded hidden states...")
            hidden_states = pickle.load(f)
    except (FileNotFoundError, OSError):
        print("Preload hidden states...")
        hidden_states = extract_hidden_states(stories, titles)
        with open(os.path.join(os.path.split(__file__)[0], "hidden_states_preload.pkl"), "wb") as f:
            pickle.dump(hidden_states, f)

    # TODO: Pre-process hidden states using a function which returns a list of GPT2Feature objects
    # feature_list = convert_hidden_states_to_list(hidden_states=hidden_states, layers=[0], dims=[*range(470, 480)])
    feature_list = pca_hidden_states_to_list(hidden_states, stories, pca_components=3)
    # feature_list = kmeans_pca_hidden_states_to_list(hidden_states, stories, pca_components=3, n_cluster=3)

    # store the feature list at the defined location
    with open(os.path.join(os.path.split(__file__)[0], PATH_TO_FEATURES, "gpt2_feature_list.pkl"), "wb") as f:
        pickle.dump(feature_list, f)

    # TODO: compute trf with batch script


if __name__ == '__main__':
    main()
