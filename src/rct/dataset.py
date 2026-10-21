import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def _open_image_with_retry(open_fn, idx: int, length: int, max_retries: int = 10):
    """Try open_fn(idx), retrying at (idx+1)%length up to max_retries times on failure.
    Returns (image, resolved_idx) so callers can fetch the matching label row.
    """
    for attempt in range(max_retries):
        try:
            return open_fn(idx), idx
        except Exception:
            idx = (idx + 1) % length
    raise RuntimeError(f"Failed to load an image after {max_retries} retries starting from a bad index")


class MIMICCXRDataset(Dataset):
    """
    MIMIC-CXR-JPG Dataset

    Directory structure:
        root/
            files/
                p{group}/
                    p{subject_id}/
                        s{study_id}/
                            *.jpg
            mimic-cxr-2.0.0-chexpert.csv.gz
            mimic-cxr-2.0.0-split.csv.gz
            mimic-cxr-2.0.0-metadata.csv.gz
    """

    CHEXPERT_LABELS = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
        "Lung Opacity", "No Finding", "Pleural Effusion",
        "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
    ]

    def __init__(
        self,
        root: str,
        split: str = "train",          # "train" | "validate" | "test"
        label_csv: str = "chexpert",   # "chexpert" | "negbio"
        transform=None,
        frontal_only: bool = True,     # PA/AP view만 사용
        uncertainty: str = "zero",     # "zero" | "one" | "ignore"
    ):
        self.root = root
        self.transform = transform
        self.uncertainty = uncertainty

        split_df = pd.read_csv(
            os.path.join(root, "mimic-cxr-2.0.0-split.csv.gz")
        )
        split_df = split_df[split_df["split"] == split].reset_index(drop=True)

        if frontal_only:
            meta_df = pd.read_csv(
                os.path.join(root, "mimic-cxr-2.0.0-metadata.csv.gz")
            )
            frontal_views = meta_df[
                meta_df["ViewPosition"].isin(["PA", "AP"])
            ][["dicom_id"]]
            split_df = split_df.merge(frontal_views, on="dicom_id")

        label_file = f"mimic-cxr-2.0.0-{label_csv}.csv.gz"
        label_df = pd.read_csv(os.path.join(root, label_file))

        self.df = split_df.merge(label_df, on=["subject_id", "study_id"], how="left")
        self.df = self.df.reset_index(drop=True)

        for col in self.CHEXPERT_LABELS:
            if col not in self.df.columns:
                self.df[col] = 0.0
                continue
            if uncertainty == "zero":
                self.df[col] = self.df[col].replace(-1, 0)
            elif uncertainty == "one":
                self.df[col] = self.df[col].replace(-1, 1)
            self.df[col] = self.df[col].fillna(0)

    def __len__(self):
        return len(self.df)

    def _get_image_path(self, row) -> str:
        subject_id = str(int(row["subject_id"]))
        study_id   = str(int(row["study_id"]))
        dicom_id   = str(row["dicom_id"])

        group = "p" + subject_id[:2]

        path = os.path.join(
            self.root, "files",
            group,
            f"p{subject_id}",
            f"s{study_id}",
            f"{dicom_id}.jpg"
        )
        return path

    def _load_image(self, idx):
        row = self.df.iloc[idx]
        npy_path = self._get_image_path(row)[:-4] + ".npy"
        arr = np.load(npy_path)                      # (256, 256) uint8 grayscale
        return Image.fromarray(arr, mode="L")

    def __getitem__(self, idx):
        image, idx = _open_image_with_retry(self._load_image, idx, len(self))
        row = self.df.iloc[idx]

        if self.transform:
            image = self.transform(image)

        labels = row[self.CHEXPERT_LABELS].values.astype("float32")

        return image, labels


class NIHChestXray14Dataset(Dataset):
    """
    NIH ChestX-ray14 Dataset

    Directory structure:
        root/
            images-224/
                images-224/
                    *.png
            Data_Entry_2017.csv
            train_val_list_NIH.txt
            test_list_NIH.txt

    Only two official splits exist (train_val, test) — "validate" and "test"
    both map to test_list_NIH.txt.
    """

    NIH_LABELS = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
        "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
        "No Finding", "Nodule", "Pleural_Thickening", "Pneumonia",
        "Pneumothorax",
    ]

    def __init__(
        self,
        root: str,
        split: str = "train",          # "train" | "validate" | "test"
        transform=None,
        frontal_only: bool = True,     # PA/AP만 사용 (NIH엔 원래 lateral이 없어 사실상 no-op)
    ):
        self.root = root
        self.transform = transform

        list_file = "train_val_list_NIH.txt" if split == "train" else "test_list_NIH.txt"
        with open(os.path.join(root, list_file)) as f:
            image_names = {line.strip() for line in f if line.strip()}

        meta_df = pd.read_csv(os.path.join(root, "Data_Entry_2017.csv"))
        self.df = meta_df[meta_df["Image Index"].isin(image_names)].reset_index(drop=True)

        if frontal_only:
            self.df = self.df[self.df["View Position"].isin(["PA", "AP"])].reset_index(drop=True)

        for label in self.NIH_LABELS:
            self.df[label] = self.df["Finding Labels"].apply(
                lambda s, label=label: float(label in s.split("|"))
            )

    def __len__(self):
        return len(self.df)

    def _load_image(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root, "images-224", "images-224", row["Image Index"])
        return Image.open(img_path).convert("L")

    def __getitem__(self, idx):
        image, idx = _open_image_with_retry(self._load_image, idx, len(self))
        row = self.df.iloc[idx]

        if self.transform:
            image = self.transform(image)

        labels = row[self.NIH_LABELS].values.astype("float32")
        return image, labels


class CheXpertDataset(Dataset):
    """
    CheXpert Dataset

    Directory structure:
        root/
            train/
                patient.../study.../*.jpg
            valid/
                patient.../study.../*.jpg
            train.csv
            valid.csv

    train.csv / valid.csv "Path" column is prefixed with
    "CheXpert-v1.0-small/", which is stripped before joining with root.
    Public CheXpert has no held-out test labels, so split="test" is invalid.
    """

    CHEXPERT_LABELS = MIMICCXRDataset.CHEXPERT_LABELS

    def __init__(
        self,
        root: str,
        split: str = "train",          # "train" | "validate"
        transform=None,
        frontal_only: bool = True,
        uncertainty: str = "zero",     # "zero" | "one" | "ignore"
    ):
        if split == "train":
            csv_name = "train.csv"
        elif split == "validate":
            csv_name = "valid.csv"
        else:
            raise ValueError(f"CheXpertDataset has no '{split}' split (only 'train'/'validate')")

        self.root = root
        self.transform = transform
        self.uncertainty = uncertainty

        self.df = pd.read_csv(os.path.join(root, csv_name))

        if frontal_only:
            self.df = self.df[self.df["Frontal/Lateral"] == "Frontal"].reset_index(drop=True)

        for col in self.CHEXPERT_LABELS:
            if col not in self.df.columns:
                self.df[col] = 0.0
                continue
            if uncertainty == "zero":
                self.df[col] = self.df[col].replace(-1, 0)
            elif uncertainty == "one":
                self.df[col] = self.df[col].replace(-1, 1)
            self.df[col] = self.df[col].fillna(0)

    def __len__(self):
        return len(self.df)

    def _get_image_path(self, row) -> str:
        rel_path = row["Path"].removeprefix("CheXpert-v1.0-small/")
        return os.path.join(self.root, rel_path)

    def _load_image(self, idx):
        row = self.df.iloc[idx]
        return Image.open(self._get_image_path(row)).convert("L")

    def __getitem__(self, idx):
        image, idx = _open_image_with_retry(self._load_image, idx, len(self))
        row = self.df.iloc[idx]

        if self.transform:
            image = self.transform(image)

        labels = row[self.CHEXPERT_LABELS].values.astype("float32")
        return image, labels
