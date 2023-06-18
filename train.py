# %% [markdown]
# # Fine-tuning ECAPA-TDNN on  [CryCeleb2023](https://huggingface.co/spaces/competitions/CryCeleb2023) using [SpeechBrain](https://speechbrain.readthedocs.io)
# 
# This notebook should help you get started training your own models for CryCeleb2023 challenge.
# 
# Note that it is provides basic example for simplicity and speed.
# 
# Author: David Budaghyan (Ubenwa)
# 

# %% [markdown]
# ### Imports

# %%
# For Colab - uncomment and run the following to set up the repo
# !pip install speechbrain
# !git clone https://github.com/Ubenwa/cryceleb2023.git
# %cd cryceleb2023

# %%

import pathlib
import random

import numpy as np
import pandas as pd
import seaborn as sns
import speechbrain as sb
import torch
from huggingface_hub import hf_hub_download
from hyperpyyaml import load_hyperpyyaml
from IPython.display import display
from speechbrain.dataio.dataio import read_audio, write_audio
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.encoder import CategoricalEncoder
from speechbrain.lobes.augment import TimeDomainSpecAugment

from crybrain import CryBrain, download_data
from tqdm import tqdm
from functools import partialmethod

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

dataset_path = "data"

# %% [markdown]
# ### Download data
# 
# You need to log in to HuggingFace to be able to download the dataset

# %%
from huggingface_hub import notebook_login

notebook_login()

# %%
download_data(dataset_path)

# %%
# read metadata
metadata = pd.read_csv(
    f"{dataset_path}/metadata.csv", dtype={"baby_id": str, "chronological_index": str}
)
train_metadata = metadata.loc[metadata["split"] != "test"].copy()
display(
    train_metadata.head()
    .style.set_caption("train_metadata")
    .set_table_styles([{"selector": "caption", "props": [("font-size", "20px")]}])
)
display(train_metadata.describe())

# %%
dev_babies = set(train_metadata.loc[train_metadata["split"] == "dev", "baby_id"])
print(dev_babies)

# %% [markdown]
# ### Concatenate cry sounds
# 
# We are given short cry sounds for each baby. Here we simply concatenate them. 

# %%
# read the segments
train_metadata["cry"] = train_metadata.apply(
    lambda row: read_audio(f'{dataset_path}/{row["file_name"]}').numpy(), axis=1
)
# concatenate all segments for each (baby_id, period) group
manifest_df = pd.DataFrame(
    train_metadata.groupby(["baby_id", "period"])["cry"].agg(lambda x: np.concatenate(x.values)),
    columns=["cry"],
).reset_index()
# all files have 16000 sampling rate
manifest_df["duration"] = manifest_df["cry"].apply(len) / 16000
pathlib.Path(f"{dataset_path}/concatenated_audio_train").mkdir(exist_ok=True)
manifest_df["file_path"] = manifest_df.apply(
    lambda row: f"{dataset_path}/concatenated_audio_train/{row['baby_id']}_{row['period']}.wav",
    axis=1,
)
manifest_df.apply(
    lambda row: write_audio(
        filepath=f'{row["file_path"]}', audio=torch.tensor(row["cry"]), samplerate=16000
    ),
    axis=1,
)
manifest_df = manifest_df.drop(columns=["cry"])
manifest_df["train_split"] = manifest_df.apply(lambda row: "dev" if row["baby_id"] in dev_babies else "train", axis=1)
display(manifest_df)
ax = sns.histplot(manifest_df, x="duration")
ax.set_title("Histogram of Concatenated Cry Sound Lengths")

# %% [markdown]
# During training, we will extract random cuts of 3-5 seconds from concatenated audio

# %%
def create_cut_length_interval(row, cut_length_interval):
    """cut_length_interval is a tuple indicating the range of lengths we want our chunks to be.
    this function computes the valid range of chunk lengths for each audio file
    """
    # the lengths are in seconds, convert them to frames
    cut_length_interval = [round(length * 16000) for length in cut_length_interval]
    cry_length = round(row["duration"] * 16000)
    # make the interval valid for the specific sound file
    min_cut_length, max_cut_length = cut_length_interval
    # if min_cut_length is greater than length of cry, don't cut
    if min_cut_length >= cry_length:
        cut_length_interval = (cry_length, cry_length)
    # if max_cut_length is greater than length of cry, take a cut of length between min_cut_length and full length of cry
    elif max_cut_length >= cry_length:
        cut_length_interval = (min_cut_length, cry_length)
    return cut_length_interval


cut_length_interval = (3, 5)
manifest_df["cut_length_interval_in_frames"] = manifest_df.apply(
    lambda row: create_cut_length_interval(row, cut_length_interval=cut_length_interval), axis=1
)

# %% [markdown]
# ### Split into train and val
# 
# For training a classfier, we can split the data into train/val in any way, as long as val does not contain new classes
# 
# One way to split is to split by period: train on birth recordings and validate on discharge

# %%
# we can train on any subset of babies (e.g. to reduce the number of classes, only keep babies with long enough cries, etc)
def get_babies_with_both_recordings(manifest_df):
    count_of_periods_per_baby = manifest_df.groupby("baby_id")["period"].count()
    baby_ids_with_recording_from_both_periods = count_of_periods_per_baby[
        count_of_periods_per_baby >0
    ].index
    return baby_ids_with_recording_from_both_periods


# def get_babies_with_a_birth_recording(manifest_df):
#   bool_series = manifest_df.groupby('baby_id')['period'].unique().apply(set(['B']).issubset)
#   baby_ids_with_a_recordings_from_birth = bool_series[bool_series].index
#   return baby_ids_with_a_recordings_from_birth


def split_by_period(row, included_baby_ids):
    if row["baby_id"] in included_baby_ids:
        if row["period"] == "D" and row["train_split"] == "dev":
            return "val"
        else:
            return "train"
    else:
        return "not_used"


babies_with_both_recordings = get_babies_with_both_recordings(manifest_df)
manifest_df["split"] = manifest_df.apply(
    lambda row: split_by_period(row, included_baby_ids=babies_with_both_recordings), axis=1
)

# each instance will be identified with a unique id
manifest_df["id"] = manifest_df["baby_id"] + "_" + manifest_df["period"]
display(manifest_df)
display(
    manifest_df["split"]
    .value_counts()
    .rename("use_babies_with_both_recordings_and_split_by_period")
)
manifest_df.set_index("id").to_json("manifest.json", orient="index")

# %% [markdown]
# ### Create dynamic datasets
# 
# See SpeechBrain documentation to understand details

# %%
# create a dynamic dataset from the csv, only used to create train and val datasets
dataset = DynamicItemDataset.from_json("manifest.json")
baby_id_encoder = CategoricalEncoder()
datasets = {}
# create a dataset for each split
for split in ["train", "val"]:
    # retrieve the desired slice (train or val) and sort by length to minimize amount of padding
    datasets[split] = dataset.filtered_sorted(
        key_test={"split": lambda value: value == split}, sort_key="duration"
    )  # select_n=100
    # create the baby_id_encoded field
    datasets[split].add_dynamic_item(
        baby_id_encoder.encode_label_torch, takes="baby_id", provides="baby_id_encoded"
    )
    # set visible fields
    datasets[split].set_output_keys(["id", "baby_id", "baby_id_encoded", "sig"])


# create the signal field for the val split (no chunking)
datasets["val"].add_dynamic_item(sb.dataio.dataio.read_audio, takes="file_path", provides="sig")

# the label encoder will map the baby_ids to target classes 0, 1, 2, ...
# only use the classes which appear in `train`,
baby_id_encoder.update_from_didataset(datasets["train"], "baby_id")


# for reading the train split, we add chunking
def audio_pipeline(file_path, cut_length_interval_in_frames):
    """Load the signal, and pass it and its length to the corruption class.
    This is done on the CPU in the `collate_fn`."""
    sig = sb.dataio.dataio.read_audio(file_path)
    if cut_length_interval_in_frames is not None:
        cut_length = random.randint(*cut_length_interval_in_frames)
        # pick the start index of the cut
        left_index = random.randint(0, len(sig) - cut_length)
        # cut the signal
        sig = sig[left_index : left_index + cut_length]
    return sig


# create the signal field (with chunking)
datasets["train"].add_dynamic_item(
    audio_pipeline, takes=["file_path", "cut_length_interval_in_frames"], provides="sig"
)

print(datasets["train"][0])

# %% [markdown]
# ### Fine-tune the classifier
# 
# Here we use a very basic example that just trains for 5 epochs

# %%
config_filename = "hparams/ecapa_voxceleb_basic.yaml"
overrides = {
    "seed": 3011,
    "n_classes": len(baby_id_encoder),
    "experiment_name": "ecapa_voxceleb_ft_basic",
    "bs": 32,
    "n_epochs": 1000,
}
device = "cuda"
run_opts = {"device": device}
###########################################
# Load hyperparameters file with command-line overrides.
with open(config_filename) as fin:
    hparams = load_hyperpyyaml(fin, overrides)

#hparams['augment_pipeline'] = [TimeDomainSpecAugment(perturb_prob = 0.5,
#                                                                     drop_freq_prob=0.5,
#                                                                     drop_chunk_prob=0.5)]

#hparams['concat_augment'] = False
# Create experiment directory
sb.create_experiment_directory(
    experiment_directory=hparams["experiment_dir"],
    hyperparams_to_save=config_filename,
    overrides=overrides,
)

# Initialize the Brain object to prepare for training.
crybrain = CryBrain(
    modules=hparams["modules"],
    opt_class=hparams["opt_class"],
    hparams=hparams,
    run_opts=run_opts,
    checkpointer=hparams["checkpointer"],
)

# if a pretrained model is specified, load it
if "pretrained_embedding_model" in hparams:
    sb.utils.distributed.run_on_main(hparams["pretrained_embedding_model"].collect_files)
    hparams["pretrained_embedding_model"].load_collected(device=device)

crybrain.fit(
    epoch_counter=crybrain.hparams.epoch_counter,
    train_set=datasets["train"],
    valid_set=datasets["val"],
    train_loader_kwargs=hparams["train_dataloader_options"],
    valid_loader_kwargs=hparams["val_dataloader_options"],
)

# %% [markdown]
# You can now use embedding_model.ckpt from this recipe and use it in evaluate.ipynb to verify pairs of cries and submit your results!


