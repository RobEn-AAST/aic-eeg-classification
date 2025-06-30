# %%
# for data info http://moabb.neurotechx.com/docs/dataset_summary.html
import moabb
from moabb.datasets import PhysionetMI, Cho2017, BNCI2014_001, Weibo2014, Lee2019_MI  # mi
from moabb.datasets import Wang2016, Nakanishi2015, MAMEM3, MAMEM2, MAMEM1, Lee2019_SSVEP, Kalunga2016  # ssvep
from moabb.paradigms import LeftRightImagery, SSVEP
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from mne import create_info
from mne.io import RawArray
from moabb.datasets.base import BaseDataset
import mne
import os


# %%
data_path = "./data/mtcaic3"



class CompetitionDataset(BaseDataset):
    def __init__(self, split="train", paradigm_type="imagery"):
        if paradigm_type == "ssvep":
            events = {"7": 1, "8": 2, "10": 3, "13": 4}
        else:
            events = {"left_hand": 1, "right_hand": 2}

        super().__init__(
            subjects=list(range(1, 31)),
            sessions_per_subject=1,
            events=events,
            code="CompetitionDataset",
            interval=[0, 4],
            paradigm=paradigm_type,
            doi=None,
        )
        self.split = split
        self.paradigm_type = paradigm_type

    def _get_single_subject_data(self, subject):
        """
        Return data for one subject - THIS IS THE KEY METHOD
        tips for Motor Imagery:
            include C3, CZ, C4
            may include FZ, PZ
            don't include PO7, PO8, Oz

        """
        ch_names = ["FZ", "C3", "CZ", "C4", "PZ", "PO7", "OZ", "PO8"]
        moabb_channels = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
        if self.paradigm == "ssvep":
            task = "SSVEP"
        elif self.paradigm == "imagery":
            task = "MI"
        else:
            raise ValueError(f"got unexpected paradigm {self.paradigm}")

        # they competition forced us to do this...
        subject_row = subject + 30 if self.split == "validation" else subject
        # Load labels for this subject
        labels_df = pd.read_csv(os.path.join(data_path, f"{self.split}.csv"), usecols=["subject_id", "trial_session", "trial", "task", "label"])
        task_df = labels_df.query(f"task=='{task}' and subject_id=='S{subject_row}'")

        if task_df.empty:
            print(f"\n\n\nWARNING TASK DF EMPTY {subject} AT ROW {subject_row} AT SPLIT {self.split}\n\n\n")
            return {"0": {"0": None}}  # No data for this subject

        sfreq = 250
        ch_types = ["eeg"] * len(ch_names)

        # Process each session
        sessions = {}
        for session_id in task_df["trial_session"].unique():
            session_trials = task_df[task_df["trial_session"] == session_id]

            # Load EEG data for this session
            fp = os.path.join(data_path, task, self.split, f"S{subject_row}", str(session_id), "EEGdata.csv")
            if not os.path.exists(fp):
                continue

            # Load the full session data
            eeg_data = pd.read_csv(fp, usecols=ch_names).values
            total_samples = eeg_data.shape[0]
            trial_length = total_samples // 10  # 10 trials per session

            # Create continuous data and events
            all_trial_data = []
            events_list = []
            current_sample = 0

            for _, trial_row in session_trials.iterrows():
                trial_num = int(trial_row["trial"])

                # Extract trial data (trial numbers are 1-indexed)
                start_idx = (trial_num - 1) * trial_length
                end_idx = trial_num * trial_length
                trial_data = eeg_data[start_idx:end_idx]

                all_trial_data.append(trial_data)

                # Create event at trial start
                if self.paradigm == "imagery":
                    if trial_row.label == "Left":
                        label = "left_hand"
                    elif trial_row.label == "Right":
                        label = "right_hand"
                    else:
                        raise ValueError(f"unkown label for motor imagery {trial_row.label}")
                elif self.paradigm == "ssvep":
                    if trial_row.label == "Left":
                        label = "10"
                    elif trial_row.label == "Right":
                        label = "13"
                    elif trial_row.label == "Forward":
                        label = "7"
                    elif trial_row.label == "Backward":
                        label = "8"
                    else:
                        raise ValueError(f"Got unkown label for SSVEP {trial_row.label}")
                else:
                    raise ValueError(f"Unkwon paradigm {self.paradigm}")

                events_list.append([current_sample, 0, self.event_id[label]])
                current_sample += trial_data.shape[0]

            if not all_trial_data:
                continue

            # Concatenate all trials for this session
            continuous_data = np.vstack(all_trial_data).T  # Shape: (channels, samples)

            # Create MNE info object
            info = create_info(moabb_channels, sfreq, ch_types)

            # Create Raw object (convert to microvolts)
            raw = RawArray(continuous_data * 1e-6, info, verbose=False)

            # Add events as annotations
            if events_list:
                events_array = np.array(events_list)
                event_desc = {v: k for k, v in self.event_id.items()}
                annotations = mne.annotations_from_events(events_array, sfreq=sfreq, event_desc=event_desc)
                raw.set_annotations(annotations)

            sessions[str(session_id)] = {"0": raw}

        if sessions is None:
            print(f"\n\n\nWARNING TASK DF EMPTY {subject} AT ROW {subject_row} AT SPLIT {self.split}\n\n\n")
            return {"0": {"0": None}}
        else:
            return sessions
        # Return in required format: {"session_id": {"run_id": raw}}

    def data_path(self, subject, path=None, force_update=False, update_path=None, verbose=None):
        """Return file paths for the subject's data"""
        subject_paths = []

        # Get all session directories for this subject
        task = "mi" if self.paradigm == "imagery" else "ssvep"
        subject_dir = os.path.join(data_path, task, self.split, f"S{subject}")
        if os.path.exists(subject_dir):
            for session in os.listdir(subject_dir):
                session_path = os.path.join(subject_dir, session)
                eeg_file = os.path.join(session_path, "EEGdata.csv")
                if os.path.exists(eeg_file):
                    subject_paths.append(eeg_file)

        return subject_paths



def load_combined_moabb_data(datasets, task="mi", paradigm_config=None, subjects_per_dataset=None):
    """
    Load and combine multiple MOABB datasets for DANN training.

    Args:
        datasets: List of MOABB dataset instances
        paradigm_config: Dict with paradigm parameters (channels, tmin, tmax, resample)
        subjects_per_dataset: Dict mapping dataset names to subject lists, or None for all

    Returns:
        X: Combined feature array
        class_labels: Binary class labels (0/1)
        domain_labels: Dataset-specific subject IDs (continuous across datasets)
        dataset_info: Metadata about each dataset
    """
    if task.upper() == "MI":
        if paradigm_config is None:
            paradigm_config = {
                "channels": ["Fz", "C3", "Cz", "C4", "Pz"],
                "tmin": 0.0,
                "tmax": 4.0,
                "resample": 250,
                "fmin": 8,
                "fmax": 32,
            }
        paradigm = LeftRightImagery(**paradigm_config)
    elif task.upper() == "SSVEP":
        if paradigm_config is None:
            paradigm_config = {
                "events": ["7", "8", "10", "13"],  # Your specific frequencies
                "n_classes": 4,
                "tmin": 0.0,
                "tmax": 4.0,  # Standard SSVEP window
                "channels": ["Oz", "POz", "PO8", "Pz"],  # Occipital channels
                "resample": 250,
                "fmin": 7,  # Covers your lowest frequency
                "fmax": 45,  # Standard SSVEP upper bound
            }
        paradigm = SSVEP(**paradigm_config)
    else:
        raise ValueError(f"task {task} is not a valid code")

    all_X = []
    all_class_labels = []
    all_domain_labels = []
    dataset_info = {}

    current_subject_offset = 0

    for dataset_idx, dataset in enumerate(datasets):
        dataset_name = dataset.__class__.__name__
        print(f"\nProcessing dataset: {dataset_name}")

        # Get subjects for this dataset
        if subjects_per_dataset and dataset_name in subjects_per_dataset:
            subjects = subjects_per_dataset[dataset_name]
        else:
            subjects = dataset.subject_list

        print(f"Original subject range: {min(subjects)} to {max(subjects)}")

        # Load data for this dataset
        X, labels, metadata = paradigm.get_data(dataset, subjects=subjects)

        # Convert string labels to binary
        class_labels = []
        for label in labels:
            if label == "left_hand":
                class_labels.append(0)
            elif label == "right_hand":
                class_labels.append(1)
            else:
                raise ValueError(f"Unexpected label {label}")

        # Create domain labels with offset to avoid conflicts
        domain_labels = []
        for i in range(len(labels)):
            original_subject = metadata.iloc[i]["subject"]
            adjusted_subject = original_subject + current_subject_offset
            domain_labels.append(adjusted_subject)

        # Update offset for next dataset
        max_subject_in_dataset = max(metadata["subject"])
        next_offset = current_subject_offset + max_subject_in_dataset

        # Store dataset info
        dataset_info[dataset_name] = {
            "original_subject_range": (min(subjects), max(subjects)),
            "adjusted_subject_range": (current_subject_offset + min(subjects), current_subject_offset + max_subject_in_dataset),
            "n_trials": len(X),
            "n_subjects": len(set(metadata["subject"])),
            "subject_offset": current_subject_offset,
        }

        print(f"Adjusted subject range: {dataset_info[dataset_name]['adjusted_subject_range']}")
        print(f"Number of trials: {len(X)}")
        print(f"Number of subjects: {len(set(metadata['subject']))}")

        # Accumulate data
        all_X.append(X)
        all_class_labels.extend(class_labels)
        all_domain_labels.extend(domain_labels)

        current_subject_offset = next_offset

    # drop to match
    tmin = paradigm_config["tmin"]
    tmax = paradigm_config["tmax"]
    sfreq = paradigm_config["resample"]
    max_possible_value = int((tmax - tmin) * sfreq)

    for i, x in enumerate(all_X):
        all_X[i] = x[:, :, :max_possible_value]

    # Combine all data
    combined_X = np.concatenate(all_X, axis=0)
    combined_class_labels = np.array(all_class_labels)
    combined_domain_labels = np.array(all_domain_labels)

    print(f"\n=== COMBINED DATASET SUMMARY ===")
    print(f"Total trials: {len(combined_X)}")
    print(f"Feature shape: {combined_X.shape}")
    print(f"Class distribution: {np.bincount(combined_class_labels)}")
    print(f"Subject range: {min(combined_domain_labels)} to {max(combined_domain_labels)}")
    print(f"Total unique subjects: {len(np.unique(combined_domain_labels))}")

    return combined_X, combined_class_labels, combined_domain_labels, dataset_info


def analyze_dataset_channels(datasets):
    """
    Analyze channels and event labels across multiple MOABB datasets and generate a comprehensive report.
    Only loads a single subject per dataset to minimize download requirements.
    """
    from collections import Counter

    dataset_channels = {}
    dataset_labels = {}
    all_channels = []
    all_labels = []
    dataset_channel_strings = []
    dataset_label_strings = []

    for dataset in datasets:
        dataset_name = type(dataset).__name__

        try:
            # Get data for first subject only - this calls _get_single_subject_data internally
            first_subject = dataset.subject_list[0]
            subject_data = dataset._get_single_subject_data(first_subject)

            first_session = list(subject_data.keys())[0]
            first_run = list(subject_data[first_session].keys())[0]
            raw = subject_data[first_session][first_run]

            # Extract EEG channels only
            raw.pick_types(eeg=True)
            channels = [ch for ch in raw.info["ch_names"] if ch.upper().find("EEG") == -1]

            # Extract event labels from dataset.events
            labels = list(dataset.event_id.keys())

            dataset_channels[dataset_name] = channels
            dataset_labels[dataset_name] = labels
            all_channels.extend(channels)
            all_labels.extend(labels)

            # Store the formatted strings
            dataset_channel_strings.append(f"{dataset_name} - Channels: {', '.join(sorted(channels))}")
            dataset_label_strings.append(f"{dataset_name} - Labels: {', '.join(sorted(labels))}")

        except Exception as e:
            dataset_channels[dataset_name] = []
            dataset_labels[dataset_name] = []
            dataset_channel_strings.append(f"{dataset_name} - Channels: Error: {e}")
            dataset_label_strings.append(f"{dataset_name} - Labels: Error: {e}")

    # Print results
    print("=== CHANNELS ===")
    for dataset_string in dataset_channel_strings:
        print(dataset_string)

    print("\n=== LABELS ===")
    for dataset_string in dataset_label_strings:
        print(dataset_string)

    print("\n=== CHANNEL FREQUENCY ===")
    channel_counts = Counter(all_channels)
    for channel, count in channel_counts.most_common():
        print(f"{channel} {count}/{len(datasets)}")

    print("\n=== LABEL FREQUENCY ===")
    label_counts = Counter(all_labels)
    for label, count in label_counts.most_common():
        print(f"{label} {count}/{len(datasets)}")

    return {
        "dataset_channels": dataset_channels,
        "dataset_labels": dataset_labels,
        "channel_frequencies": dict(channel_counts),
        "label_frequencies": dict(label_counts),
        "common_channels": [ch for ch, count in channel_counts.items() if count == len(datasets)],
        "common_labels": [label for label, count in label_counts.items() if count == len(datasets)],
    }


def main():
    # Example usage:
    datasets = [
        # Wang2016(),
        # Nakanishi2015(),
        # MAMEM3(),
        # MAMEM2(),
        # MAMEM1(),
        # Lee2019_SSVEP(),
        # Kalunga2016(),
        CompetitionDataset(paradigm_type="ssvep"),
    ]

    analyze_dataset_channels(datasets)

    return
    # Load combined data
    X, class_labels, domain_labels, info = load_combined_moabb_data(
        datasets=datasets,
        task="ssvep",
        subjects_per_dataset={
            "CompetitionDataset": [1, 2, 3],
        },
    )

    # Create combined labels for DANN
    combined_labels = np.column_stack([class_labels, domain_labels])
    print(f"Combined labels shape: {combined_labels.shape}")
    print(f"Sample combined labels: {combined_labels[:5]}")


if __name__ == "__main__":
    main()
