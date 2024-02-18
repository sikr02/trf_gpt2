import os
import torch  # for GPU usage, use a precompiled version of pytorch (see: https://pytorch.org/get-started/locally/)
from tqdm import tqdm
from minicons import cwe
import numpy as np
import time
import sys


def remaining_time(remaining_iters: int, time_per_iter: float) -> tuple[int, int, int]:
    """
    Computes and formats the remaining time of computations

    :param remaining_iters: number of remaining computations
    :param time_per_iter: time of one iteration
    :return: tuple containing formatted time (hours, minutes, seconds)
    """
    rem_time = remaining_iters * time_per_iter
    return int(rem_time // 3600), int((rem_time % 3600) // 60), int(rem_time % 60)


def load_stories(path_to_transcripts: str) -> tuple[list[str], list[str]]:
    """
    Load titles and stories

    :param path_to_transcripts: path to where the files are located
    :return: a list of the title names and a list of the story contents
    """
    filenames = []
    stories = []

    # read in stories
    for story in os.listdir(path_to_transcripts):
        filenames.append(story)
        file = os.path.join(path_to_transcripts, story, "clean_" + story + "_35K.txt")
        with open(file, "r") as f:
            text = f.read()

            # PREPROCESS TEXT
            # remove line breaks
            text = text.replace("\n", " ")
            # remove meaningless apostrophes (=> so that number of words is matching to number of onsets)
            text = text.replace("\' ", "")
            # store in list
            stories.append(text)

    return filenames, stories


def extract_hidden_states(stories: list[str], filenames: list[str]) -> list[np.ndarray]:
    """
    Extract hidden states of GPT-2 for the listed stories,
    shape: #stories arrays of shape (#words, layers + embedding (13), embedding size (768))

    :param stories: list of the story texts
    :param filenames: list of the story names
    :return: list of numpy arrays containing hidden states
    """
    # clear cache
    torch.cuda.empty_cache()

    # load small gpt2 model from minicons cwe
    lm = cwe.CWE(model_name="gpt2", device="cuda:0" if torch.cuda.is_available() else "cpu")
    # add unknown token "<unk>" and end-of-sequence token (or start token) "<s>"
    lm.tokenizer.add_special_tokens({"unk_token": "<unk>", "eos_token": "<s>"})
    lm.model.resize_token_embeddings(len(lm.tokenizer))

    # textwise and character indexing
    word_count = [len(s.split()) for s in stories]

    # list which will be of size (stories, words, layers, 1, hidden_size)
    results = [[] for _ in stories]
    for i, filename in enumerate(filenames):
        stimuli = []
        # Split into character spans
        start_idx = 0
        for idx, c in enumerate(stories[i]):
            if c == " ":
                # Add start token "<s> " to prevent function extract_representation from returning NaN-values
                stimuli.append(["<s> " + stories[i][:-1], (start_idx + len("<s> "), idx + len("<s> "))])
                start_idx = idx + 1

        # measuring time
        start_time = time.time_ns()

        # compute word for word
        for batch in tqdm(stimuli, desc=f"Story \"{filename}\" ({i + 1}/{len(stories)})", file=sys.stdout):
            reps = lm.extract_representation(batch, layer=[*range(13)])
            results[i].append(reps)

        # measuring time
        time_per_word = (time.time_ns() - start_time) / word_count[i] * 1e-9
        remaining_words = sum(word_count[(i + 1):])

        # remaining time
        h, m, s = remaining_time(remaining_words, time_per_word)
        print(f"Completed story \"{filename}\" (time per word: {time_per_word:.2f}s) => remaining words: "
              f"{remaining_words} (estimated remaining time: {h:02}h {m:02}m {s:02}s)")

    hidden_states_np_list = []
    for story in results:
        feat_array = np.asarray(
            [[[t.numpy() for t in layer]
              for layer in word]
             for word in story]
        )

        hidden_states_np_list.append(feat_array[:, :, 0, :])  # remove unnecessary batch dimension

    return hidden_states_np_list
