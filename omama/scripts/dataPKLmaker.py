#!/usr/bin/env python3

import os
import pandas as pd
from pathlib import Path
import pydicom
import pickle
import argparse
from tqdm import tqdm


def create_datalist(data_dir, csv_path, output_path):
    df = pd.read_csv(csv_path)

    datalist = []

    for d in tqdm(sorted(os.listdir(data_dir)), desc="Processing studies"):
        # Initialize the dictionary for the study
        study_dict = {
            "L-MLO": [],
            "R-MLO": [],
            "L-CC": [],
            "R-CC": [],
            "horizontal_flip": "NO",
            "cancer_label": {
                "left_malignant": 0,
                "right_malignant": 0,
                "left_benign": 0,
                "right_benign": 0,
            },
        }

        study_dir = os.path.join(data_dir, d)

        # Find all files for the study, excluding .png files, and assign to correct laterality/views
        for path in Path(study_dir).iterdir():
            if path.suffix == ".png":
                continue  # Skip .png files

            ds = pydicom.dcmread(path.absolute())

            # print(f"DEBUG: ds.ViewCodeSequence[0].CodeMeaning = {ds.ViewCodeSequence[0].CodeMeaning}")  # Add this for debugging

            # Figure out mammo view (MLO versus CC)
            if (
                ds.ViewCodeSequence[0].CodeMeaning == "medio-lateral oblique"
                or ds.ViewCodeSequence[0].CodeMeaning == "MLO"
            ):
                mammo_view = "MLO"
            elif (
                ds.ViewCodeSequence[0].CodeMeaning == "cranio-caudal"
                or ds.ViewCodeSequence[0].CodeMeaning == "CC"
                or ds.ViewCodeSequence[0].CodeMeaning
                == "cranio-caudal exaggerated laterally"
            ):
                mammo_view = "CC"
            else:
                print(f"DEBUG: mammo_view = {mammo_view}")  # Add this for debugging
                raise ValueError(f"Unsupported mammo view {mammo_view}")

            lat_and_view = f"{ds.ImageLaterality}-{mammo_view}"
            study_dict[lat_and_view].append(os.path.join(d, path.name))

        # Find correct labels
        study_df = df[df.StudyInstanceUID == d]
        for _, row in study_df.iterrows():
            if row.Label == "IndexCancer":
                if row.CancerLaterality == "L":
                    study_dict["cancer_label"]["left_malignant"] = 1
                elif row.CancerLaterality == "R":
                    study_dict["cancer_label"]["right_malignant"] = 1
                else:
                    raise ValueError(
                        f"Unexpected value in CancerLaterality for IndexCancer label: {row.CancerLaterality}"
                    )

            elif row.Label == "NonCancer":
                if pd.isna(
                    row.CancerLaterality
                ):  # For NonCancer cases where CancerLaterality is None
                    laterality_from_dicom = ds.ImageLaterality
                    if laterality_from_dicom == "L":
                        study_dict["cancer_label"]["left_benign"] = 1
                    elif laterality_from_dicom == "R":
                        study_dict["cancer_label"]["right_benign"] = 1
                    else:
                        raise ValueError(
                            f"Unexpected value in DICOM ImageLaterality: {laterality_from_dicom}"
                        )
            else:
                raise ValueError(f"Unexpected label: {row.Label}")

        datalist.append(study_dict)

    with open(output_path, "wb") as f:
        pickle.dump(datalist, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate datalist.pkl for mammography dataset."
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        required=True,
        help="Path to directory containing study folders.",
    )
    parser.add_argument(
        "-c", "--csv_path", required=True, help="Path to CSV file with study labels."
    )
    parser.add_argument(
        "-o",
        "--output_path",
        default="datalist.pkl",
        help="Output path for generated .pkl file (default: datalist.pkl).",
    )

    args = parser.parse_args()

    create_datalist(args.data_dir, args.csv_path, args.output_path)
