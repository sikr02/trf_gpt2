# trf_gpt2

Repository for the Bachelor Thesis "Features of GPT-2 for Improving Speech-Brain-Models".

## Setup
1. Download the EEG and speech data set from https://doi.org/10.6084/m9.figshare.9033983.v1 (CC0, Creative Commons) and move it to `./gpt2features_trf/sample_data/`
2. Install `conda` and make sure you have added conda to PATH.
3. Because of version dependencies two virtual environments are needed:
   1. Create an environment with python version 3.10: 
      ```bash
      conda create -n "gpt2features_generate_env" python=3.10.0
      ```
      Navigate to the `./gpt2features_generate` folder and install the packages from requirements:
      ```bash
      conda activate gpt2features_generate_env
      pip install -r requirements.txt
      ```
   2. Create an environment with python version 3.6:
      ```bash
      conda create -n "gpt2features_trf_env" python=3.6.13
      ```
4. Download the pyEEG library from https://doi.org/10.6084/m9.figshare.9034481.v3 (licensed under GNU GENERAL PUBLIC LICENSE) and extract `pyEEG.zip` to `./gpt2features_trf/Code/pyEEG`
5. The project structure should look like this:
   ```
   ./
   ├── gpt2features_generate
   │   ├── Code
   │   │   ├── main.py
   │   │   ├── gpt2_utils.py
   │   │   └── preprocess.py
   │   │
   │   └── requirements.txt
   │
   ├── gpt2features_trf
   │   ├── Code
   │   │   ├── pyEEG
   │   │   │   ├── pyeeg
   │   │   │   ├── LICENSE
   │   │   │   ├── README.md
   │   │   │   ├── requirements.txt
   │   │   │   └── setup.py
   │   │   │
   │   │   ├── main.py
   │   │   ├── plot.py
   │   │   └── utils.py
   │   │
   │   └── data
   │       └── reference.pkl
   │
   ├── sample_data
   │   ├── P01_bis
   │   │   ├── config.txt
   │   │   ├── Fs-125..._bis.set
   │   │   └── P01_bis.mat
   │   │ 
   │   ├── ...
   │   ├── P14_21032017
   │   ├── stories/story_parts
   │   │   ├── alignment_data
   │   │   ├── surprisal
   │   │   ├── transcripts
   │   │   └── word_freqencies
   │   │
   │   └── onsets.mat
   │
   ├── .gitignore
   ├── LICENSE
   └── README.md
   ```
6. Navigate to the `./gpt2features_trf/Code/` folder and activate the environment:
   ```bash
   conda activate gpt2features_trf_env
   ```
   Follow the steps from `./gpt2features_trf/Code/pyEEG/README.md` to install the pyEEG library to the environment. 
   Don't create another environment, as they propose, use the `gpt2features_trf_env` environment instead.
   
## Run
1. **Edit** main function in `./gpt2features_generate/Code/main.py` and generate the features which should be computed in the TRF.
2. Activate the `gpt2features_generate_env` environment and **run** the `./gpt2features_generate/Code/main.py` file.
3. Check if `gpt2_feature_list.pkl` appears in the `./gpt2features_trf/data/` folder.
4. Activate the `gpt2features_trf_env` environment and **run** `./gpt2features_trf/Code/main.py`. This may take some time depending on the number of features you have generated before. 
The remaining time can be checked in the generated log files.
5. The results are stored as `<feature_name>.pkl`. They contain the TRF coefficients, the correlation scores for the subjects, 
the average correlation and the results from a paired t-test.
6. Select a feature by **editing** the variable `filename_to_plot` in the main function of `./gpt2features_trf/Code/plot.py`.
7. Plot the TRF results of the feature as butterfly and topomap plot by **running** `./gpt2features_trf/Code/main.py` in the `gpt2features_trf_env` environment.

Author: Simon Kramer, 19.02.2024
