import glob
import os
import pandas as pd
import numpy as np
from multiprocessing import Pool, Process, Queue
from itertools import repeat
import time
import pydicom
from pathlib import Path
from collections import defaultdict


class DataSelector:
    def __init__(self, path_database=None, path_labels=None, directories=None):
        self.files_2D = []
        self.files_3D = []
        self.df_labels = None
        self.df_omama = pd.DataFrame()
        self.path_database = path_database
        self.subsetting_fields = [
            "FilePath",
            "StudyInstanceUID",
            "image",
            "Imagelaterality",
            "label",
            "PatientAge",
            "PatientName",
            "PatientBirthDate",
            "PatientSex",
            "PatientID",
            "Shape",
            "Manufacturer",
            "Modality",
            "SeriesInstanceUID",
            "SOPInstanceUID",
        ]
        if self.path_database is None:
            self.path_database = "/raid/data01/deephealth"
        self.path_labels = path_labels
        if self.path_labels is None:
            self.path_labels = "/raid/data01/deephealth/labels/"
        self.directories = directories
        if self.directories is None:
            self.directories = ["dh_dh2", "dh_dh0new", "dh_dcm_ast"]

    def scrape_and_store_database(self, tasks=24, timeit=False):
        if not os.path.isdir(os.path.join(os.getcwd(), "df_omama.plk")):
            directories_paths_tuples = [
                (directory, os.path.join(self.path_database, directory))
                for directory in self.directories
            ]
            result_dfs_holder = []
            queue = Queue()
            for tuple in directories_paths_tuples:
                print(f"Loading {tuple[0]} directory contents into memory")
                self.folder = tuple[0]
                self.df_labels = pd.read_csv(f"{self.path_labels}{tuple[0]}_labels.csv")
                self.df_labels = self.df_labels.drop_duplicates()
                p1 = Process(
                    target=self._generate_files,
                    args=(1, str(tuple[1]) + "/*/DXm*", queue),
                )
                p2 = Process(
                    target=self._generate_files,
                    args=(2, str(tuple[1]) + "/*/BT*", queue),
                )
                p1.start()
                p2.start()
                if timeit:
                    t0 = time.time()
                while True:
                    msg = queue.get()
                    if msg[0] == 1:
                        self.files_2D = msg[1]
                        df_2D = self._create_dataset(tasks, 1)
                    else:
                        self.files_3D = msg[1]
                        df_3D = self._create_dataset(tasks, 2)
                    if self.files_2D and self.files_3D:
                        result_df = pd.concat([df_2D, df_3D], ignore_index=True)
                        result_df.to_pickle(
                            os.path.join(os.getcwd(), f"{tuple[0]}.pkl")
                        )
                        if timeit:
                            print(
                                f"Completed loading {tuple[0]} directory contents into memory ...took ",
                                time.time() - t0,
                                "seconds ",
                            )
                        else:
                            print(
                                f"Completed loading {tuple[0]} directory contents into memory"
                            )
                        result_dfs_holder.append(result_df)
                        break
                # flush out the queue
                if not queue.empty():
                    r = queue.get()
                p1.join()
                p2.join()
            for df in result_dfs_holder:
                self.df_omama = pd.concat([self.df_omama, df], ignore_index=True)
            self.df_omama.to_pickle(os.path.join(os.getcwd(), "df_omama.pkl"))
            print("Contents of all directories loaded into memory")
        else:
            print("Omama Dataframe pickle file detected, no need to scrape database")

    def _create_dataset(self, task_num, image_type_flag):
        def create_dataset_aux(method, list_elems, flag, task_num):
            with Pool(task_num) as pool:
                df = pool.starmap(method, zip(list_elems, repeat(flag)))
                result_df = pd.concat(df)
                return result_df

        if image_type_flag == 1:
            return create_dataset_aux(
                self._build_dataframe, self.files_2D, image_type_flag, task_num
            )
        return create_dataset_aux(
            self._build_dataframe, self.files_3D, image_type_flag, task_num
        )

    def _generate_files(self, ind, path_to_hospital, q):
        list = sorted([f for f in glob.glob(path_to_hospital)])
        q.put((ind, list))

    def _build_dataframe(self, file, image_flag):
        def getLabel(studyInstanceUID, imageLaterality):
            value = self.df_labels[
                (self.df_labels["StudyInstanceUID"] == str(studyInstanceUID))
            ]
            if value.shape[0] != 0:
                # required when same study instance UID has two cancer laterality information stored in the csv file
                value = value[value["CancerLaterality"] == str(imageLaterality)]
                if value.shape[0] != 0:
                    return str(value["Label"].values[0])
                else:
                    return "NonCancer"
            elif value.shape[0] == 0:
                return "No Label information"
            else:
                print(value)
                print("error")

        db = []
        ds = pydicom.filereader.dcmread(file, stop_before_pixels=True, force=False)
        if image_flag == 1:
            studyInstanceUID = ds.get("StudyInstanceUID")
            image_type = "2D"
            imageLaterality = ds.get("ImageLaterality")
            shape = (ds.Rows, ds.Columns)
        else:
            studyInstanceUID = ds.get("StudyInstanceUID")
            image_type = "3D"
            imageLaterality = (
                ds.SharedFunctionalGroupsSequence[0]
                .FrameAnatomySequence[0]
                .FrameLaterality
            )
            shape = (ds.Rows, ds.Columns, ds.NumberOfFrames)

        h = {
            "filePath": file,
            "image": image_type,
            "StudyInstanceUID": str(studyInstanceUID),
            "Imagelaterality": str(imageLaterality),
            "label": str(getLabel(str(studyInstanceUID), str(imageLaterality))),
            "PatientAge": str(ds.get("PatientAge")),
            "PatientName": str(ds.get("PatientName")),
            "PatientBirthDate": str(ds.get("PatientBirthDate")),
            "PatientSex": str(ds.get("Patient'sSex")),
            "PatientID": str(ds.get("PatientID")),
            "Shape": shape,
            "Manufacturer": str(ds.get("Manufacturer")),
            "Modality": str(ds.get("Modality")),
            "SeriesInstanceUID": str(ds.get("SeriesInstanceUID")),
            "SOPInstanceUID": str(ds.get("SOPInstanceUID")),
        }
        for k in ds.keys():
            # Skip images
            if (k.group, k.elem) == (0x7FE0, 0x0010):
                continue
            # Checks in case underlying int is >32 bits, DICOM does not allow this
            #             if (not -2**31 <= k < 2**31):
            #                 continue
            try:
                v = ds[k]
            except:
                continue
            key = (
                v.name.replace(" ", "")
                .replace("'", "")
                .replace("/", "")
                .replace("[", "")
                .replace("]", "")
                .replace("(", "")
                .replace(")", "")
                .replace("-", "")
            )

            value = v.value  # Do I need to encode to bytes?
            if value is None:
                continue
            # Conversions that are necessary
            if (k.group, k.elem) in [
                (0x20, 0x1041),  # group and elem for Slice location
                (0x18, 0x50),  # group and elem for Slice thickness
            ]:
                value = float(value)
            elif (k.group, k.elem) in [
                (0x28, 0x30),  # Pixel spacing is a pair value
            ]:
                value = tuple(float(f) for f in v.value)

            elif not ("\\" in str(value) or "[" in str(value)):
                if v.VR in ["IS", "SL", "US"]:
                    try:
                        value = int(value)
                    except ValueError:
                        continue
                elif isinstance(value, pydicom.valuerep.DSfloat):
                    value = float(value)
                elif isinstance(value, bytes):
                    value = value.decode("utf-8")
                else:
                    value = str(value)

            # coerce to string if prev conditions unmet. Important for pickling
            if not type(value) in [str, list, dict, tuple, int, float]:
                value = str(value)
            # Turn lists to tuples to make immutable
            if type(value) == list:
                value = tuple(value)
            h[key] = value
            h["X%04x_%04x" % (k.group, k.elem)] = h[
                key
            ]  # Use both name and group,element syntax
        db += [h]
        df = pd.DataFrame(db)
        return df

    def load_dataset(self, filepath="df_omama.pkl"):
        """Load database file from disk"""
        return pd.read_pickle(filepath)

    def fill_na_dataset(self, df):
        df_omama = df
        df_omama.drop(
            ["PatientsName", "PatientsAge", "PatientsSex", "PatientsBirthDate"],
            axis=1,
            inplace=True,
        )
        # extract Patient Age numeric value and convert to float
        df_omama["PatientAge"] = df_omama["PatientAge"].str.extract(
            "(\d+)", expand=False
        )
        df_omama["PatientAge"] = df_omama["PatientAge"].apply(lambda x: float(x) // 12)
        df_omama = df_omama.sort_values("PatientID")
        # replace empty string with nan
        df_omama.replace(r"^\s*$", np.NaN, regex=True, inplace=True)
        # replace None value with nan
        df_omama.fillna(value=np.NaN, inplace=True)
        df_omama.replace("None", np.NaN, inplace=True)
        return df_omama

    def check_percent_missing_values_by_field(self, df):
        df_omama = df
        # check percentage missing values across columns
        percent_missing = df_omama.isnull().sum() * 100 / len(df_omama)
        # missing_value_df = pd.DataFrame({'column_name': df_omama.columns,
        #                                  'percent_missing': percent_missing})
        missing_value_df = pd.DataFrame({"percent_missing": percent_missing})
        # missing_value_df.sort_values('column_name', inplace=True)
        missing_value_df.sort_index(axis=0, inplace=True)
        print(missing_value_df)

    def subset_by_fields(self, df):
        # subset dataframe by fields of interest and sort on SOP ID
        df_omama_reduced = df[self.subsetting_fields]
        df_omama_reduced = df_omama_reduced.sort_values("SOPInstanceUID")
        return df_omama_reduced

    def store_dataset(self, df, path_to_store=None):
        if path_to_store is None:
            path_to_store = os.path.join(os.getcwd())
        file_path = os.path.join(path_to_store, "df_omama_reduced.pkl")
        df.to_pickle(file_path)
        assert Path(file_path).is_file()
        return df

    def subset_pipeline_to_manufacturer_step(self, df=None, filepath="df_omama.pkl"):
        df_omama = df
        if df is None:
            df_omama = self.load_dataset(filepath)
        # Removal of No Label and Both values subset
        df_omama = df_omama[df_omama.label != "No Label information"]
        df_omama = df_omama[df_omama.Imagelaterality != "B"]
        # Shape subsetting steps
        df_omama["Shape"] = df_omama["Shape"].astype(str)
        df_omama["Shape"] = df_omama["Shape"].str.replace(
            "(2457, 1890,.*)", "2457, 1890, x)"
        )
        df_omama["Shape"] = df_omama["Shape"].str.replace(
            "(2457, 1996,.*)", "2457, 1996, x)"
        )
        df_omama["Shape"] = df_omama["Shape"].str.replace(
            "(2059, 652,.*)", "2059, 652, x)"
        )
        df_omama["Shape"] = df_omama["Shape"].str.replace(
            "(2023, 918,.*)", "2023, 918, x)"
        )
        # remove shape less than (1024, 1024)
        pattern = "(2059, 652,.*)|(2023, 918,.*)|(512, 512)|(707, 989)|(794, 990)"
        df_omama = df_omama[~df_omama["Shape"].str.contains(pattern)]
        # Non-dominant Manufacturers removal
        df_omama["Manufacturer"] = df_omama["Manufacturer"].replace(
            [
                "HOLOGIC, Inc.",
                "Hologic, Inc.",
                "HOLOGIC, INC.",
                "LORAD",
                "Lorad",
                "LORAD.",
                "Lorad, A Hologic Company",
            ],
            "HOLOGIC, INC.",
        )
        manu_keep = ["GE MEDICAL SYSTEMS", "HOLOGIC, INC."]
        df_omama = df_omama.loc[df_omama["Manufacturer"].isin(manu_keep)]
        return df_omama

    def subset_pipeline_post_manufacturer_step(
        self, df=None, by="StudyInstanceUID", filepath="df_omama.pkl"
    ):
        if df is not None:
            df_omama = df
        else:
            df_omama = self.subset_pipeline_to_manufacturer_step(filepath=filepath)
        print("After Manufacturer Subsetting Step: ")
        self.check_metrics(df_omama)
        df_omama_copy = df_omama.copy()
        df_3d = df_omama_copy[df_omama_copy["image"] == "3D"]
        df_2d = df_omama_copy[df_omama_copy["image"] == "2D"]
        dfCancer = dfnonCancer = df_omama_copy
        dfCancer = dfCancer[dfCancer["label"] != "NonCancer"]
        dfnonCancer = dfnonCancer[dfnonCancer["label"] == "NonCancer"]

        df_3d_cancer = dfCancer[dfCancer["image"] == "3D"]
        df_2d_cancer = dfCancer[dfCancer["image"] == "2D"]
        df_2d_cancer_pre = df_2d_cancer[df_2d_cancer["label"] == "PreIndexCancer"]
        df_2d_cancer_index = df_2d_cancer[df_2d_cancer["label"] == "IndexCancer"]

        df_3d_noncancer = dfnonCancer[dfnonCancer["image"] == "3D"]
        df_2d_noncancer = dfnonCancer[dfnonCancer["image"] == "2D"]

        df_2d_noncancer_dropped = df_2d_noncancer.drop_duplicates([by])
        df_2d_noncancer_no_overlap = df_2d_noncancer_dropped[
            ~df_2d_noncancer_dropped[by].isin(df_3d[by].values)
        ]
        merged_noncancer_dfs_no_overlap = pd.concat(
            [df_2d_noncancer_no_overlap, df_3d_noncancer]
        )
        df_omama = pd.concat([dfCancer, merged_noncancer_dfs_no_overlap])
        print(f"After {by} Duplicate Removal for 2D NonCancer Subsetting Step: ")
        self.check_metrics(df_omama)
        return df_omama

    def make_caselist(self, df):
        path = "/tmp/test/"
        filename = "whitelist.txt"
        with open(path + filename, "w") as f:
            for file_path in df["FilePath"]:
                f.write(file_path + "\n")
        return path + filename

    def get_study_to_sop_dict(self, df):
        sop_to_study_dict = pd.Series(
            df.StudyInstanceUID.values, index=df.SOPInstanceUID
        ).to_dict()
        swapped = defaultdict(set)
        for k, v in sop_to_study_dict.items():
            swapped[v].add(k)
        return swapped

    def check_metrics(self, df):
        df_omama_copy = df
        print(f"Total cases count: {len(df_omama_copy.index)}")
        print(f"Total studies count: {df_omama_copy.StudyInstanceUID.nunique()}")
        print(f"Total patients count: {df_omama_copy.PatientID.nunique()}\n")

        df_3d = df_omama_copy[df_omama_copy["image"] == "3D"]
        df_2d = df_omama_copy[df_omama_copy["image"] == "2D"]

        print(f"Total 2D cases count: {len(df_2d.index)}")
        print(f"Total 2D studies count: {df_2d.StudyInstanceUID.nunique()}\n")
        print(f"Total 2D patients count: {df_2d.PatientID.nunique()}\n")

        print(f"Total 3D cases count: {len(df_3d.index)}")
        print(f"Total 3D studies count: {df_3d.StudyInstanceUID.nunique()}")
        print(f"Total 3D patient count: {df_3d.PatientID.nunique()}")
        merged = df_3d.merge(df_2d, on="StudyInstanceUID", how="inner")
        print(
            f"number of 3D studies that match with those of 2D: {merged.StudyInstanceUID.nunique()}"
        )
        merged = df_3d.merge(df_2d, on="PatientID", how="inner")
        print(
            f"number of 3D patients that match with those of 2D: {merged.PatientID.nunique()}\n"
        )
        dfCancer = dfnonCancer = df_omama_copy
        dfCancer = dfCancer[dfCancer["label"] != "NonCancer"]
        dfnonCancer = dfnonCancer[dfnonCancer["label"] == "NonCancer"]

        df_2d_cancer = dfCancer[dfCancer["image"] == "2D"]
        df_2d_cancer_cases_counts = len(df_2d_cancer.index)
        df_2d_cancer_unique_counts = df_2d_cancer["StudyInstanceUID"].nunique()
        df_2d_cancer_unique_patients = df_2d_cancer["PatientID"].nunique()
        df_2d_cancer_pre = df_2d_cancer[df_2d_cancer["label"] == "PreIndexCancer"]
        df_2d_cancer_index = df_2d_cancer[df_2d_cancer["label"] == "IndexCancer"]
        df_3d_cancer = dfCancer[dfCancer["image"] == "3D"]
        df_2d_noncancer = dfnonCancer[dfnonCancer["image"] == "2D"]
        df_3d_noncancer = dfnonCancer[dfnonCancer["image"] == "3D"]

        df_2d_cancer_pre_cases_counts = len(df_2d_cancer_pre.index)
        df_2d_cancer_pre_unique_counts = df_2d_cancer_pre["StudyInstanceUID"].nunique()
        df_2d_cancer_pre_unique_patients = df_2d_cancer_pre["PatientID"].nunique()
        df_2d_cancer_pre_nonunique_counts = (
            df_2d_cancer_pre_cases_counts - df_2d_cancer_pre_unique_counts
        )

        df_2d_cancer_index_cases_counts = len(df_2d_cancer_index.index)
        df_2d_cancer_index_unique_counts = df_2d_cancer_index[
            "StudyInstanceUID"
        ].nunique()
        df_2d_cancer_index_unique_patients = df_2d_cancer_index["PatientID"].nunique()
        df_2d_cancer_index_nonunique_counts = (
            df_2d_cancer_index_cases_counts - df_2d_cancer_index_unique_counts
        )

        print(f"2d Cancer cases count: {df_2d_cancer_cases_counts}")
        print(f"Unique 2D Cancer Study instances count: {df_2d_cancer_unique_counts}")
        print(
            f"Unique 2D Cancer Patient instances count: {df_2d_cancer_unique_patients}"
        )
        merged = df_2d_cancer.merge(df_2d_noncancer, on="StudyInstanceUID", how="inner")
        print(
            f"number of 2D cancer studies that match with those of 2D noncancer: {merged.StudyInstanceUID.nunique()}"
        )
        merged = df_2d_cancer.merge(df_2d_noncancer, on="PatientID", how="inner")
        print(
            f"number of 2D cancer patients that match with those of 2D noncancer: {merged.PatientID.nunique()}\n"
        )

        print(f"2d Cancer -Pre- cases count: {df_2d_cancer_pre_cases_counts}")
        print(
            f"Unique 2D Cancer -Pre- Study instances count: {df_2d_cancer_pre_unique_counts}"
        )
        print(
            f"Unique 2D Cancer -Pre- Patient instances count: {df_2d_cancer_pre_unique_patients}"
        )
        print(
            f"Nonunique 2D Cancer -Pre- Study instances count: {df_2d_cancer_pre_nonunique_counts}"
        )
        merged = df_2d_cancer_pre.merge(
            df_2d_noncancer, on="StudyInstanceUID", how="inner"
        )
        print(
            f"number of 2D pre cancer studies that match with those of 2D noncancer: {merged.StudyInstanceUID.nunique()}"
        )
        merged = df_2d_cancer_pre.merge(df_2d_noncancer, on="PatientID", how="inner")
        print(
            f"number of 2D pre cancer patients that match with those of 2D noncancer: {merged.PatientID.nunique()}\n"
        )

        print(f"2d Cancer -Index- cases count: {df_2d_cancer_index_cases_counts}")
        print(
            f"Unique 2D Cancer -Index- Study instances count: {df_2d_cancer_index_unique_counts}"
        )
        print(
            f"Unique 2D Cancer -Index- Patient instances count: {df_2d_cancer_index_unique_patients}"
        )
        print(
            f"Nonunique 2D Cancer -Index- Study instances count: {df_2d_cancer_index_nonunique_counts}"
        )
        merged = df_2d_cancer_index.merge(
            df_2d_noncancer, on="StudyInstanceUID", how="inner"
        )
        print(
            f"number of 2D index cancer studies that match with those of 2D noncancer: {merged.StudyInstanceUID.nunique()}"
        )
        merged = df_2d_cancer_index.merge(df_2d_noncancer, on="PatientID", how="inner")
        print(
            f"number of 2D index cancer patients that match with those of 2D noncancer: {merged.PatientID.nunique()}\n"
        )

        merged = df_2d_cancer_pre.merge(
            df_2d_cancer_index, on="StudyInstanceUID", how="inner"
        )
        print(
            f"number of 2D precancer studies that match with those of 2D indexcancer cases: {merged.StudyInstanceUID.nunique()}"
        )
        merged = df_2d_cancer_pre.merge(df_2d_cancer_index, on="PatientID", how="inner")
        print(
            f"number of 2D precancer patients that match with those of 2D indexcancer cases: {merged.PatientID.nunique()}\n"
        )

        df_3d_cancer = dfCancer[dfCancer["image"] == "3D"]
        print(f"3d Cancer -Index- cases count: {len(df_3d_cancer.index)}")
        print(
            f"unique 3D Cancer Study instances count {df_3d_cancer['StudyInstanceUID'].nunique()}"
        )
        print(
            f"unique 3D Cancer Patient instances count {df_3d_cancer['PatientID'].nunique()}"
        )
        print(
            f"Nonunique 3D Cancer study instances count {len(df_3d_cancer.index) - df_3d_cancer['StudyInstanceUID'].nunique()}\n"
        )

        merged = df_3d_cancer.merge(df_3d_noncancer, on="StudyInstanceUID", how="inner")
        print(
            f"number of 3D cancer Studies that match with 3D noncancer: {merged.StudyInstanceUID.nunique()}"
        )
        merged = df_3d_cancer.merge(df_2d_noncancer, on="StudyInstanceUID", how="inner")
        print(
            f"number of 3D cancer Studies that match with 2D noncancer:"
            f" {merged.StudyInstanceUID.nunique()}"
        )
        merged = df_3d_cancer.merge(df_2d_cancer, on="StudyInstanceUID", how="inner")
        print(
            f"number of 3D cancer Studies that match with 2D Cancer:"
            f" {merged.StudyInstanceUID.nunique()}"
        )
        merged = df_3d_cancer.merge(df_2d, on="StudyInstanceUID", how="inner")
        print(
            f"number of 3D cancer Studies that match with 2D:"
            f" {merged.StudyInstanceUID.nunique()}"
        )
        merged = df_3d_cancer.merge(df_3d_noncancer, on="PatientID", how="inner")
        print(
            f"number of 3D cancer Patients that match with 3D noncancer: {merged.PatientID.nunique()}"
        )

        merged = df_3d_cancer.merge(df_2d_noncancer, on="PatientID", how="inner")
        print(
            f"number of 3D cancer Patients that match with 2D noncancer:"
            f" {merged.PatientID.nunique()}"
        )
        merged = df_3d_cancer.merge(df_2d_cancer, on="PatientID", how="inner")
        print(
            f"number of 3D cancer Patients that match with 2D Cancer:"
            f" {merged.PatientID.nunique()}"
        )
        merged = df_3d_cancer.merge(df_2d, on="PatientID", how="inner")
        print(
            f"number of 3D cancer Patients that match with 2D:"
            f" {merged.PatientID.nunique()}"
        )
        merged = df_3d_cancer.merge(df_3d, on="StudyInstanceUID", how="inner")
        print(
            f"Sanity Check: number of 3D cancer Studies that match with 3D: {merged.StudyInstanceUID.nunique()}\n"
        )

        print(f"2d Noncancer cases count: {len(df_2d_noncancer.index)}")
        print(
            f"Unique 2D Noncancer study instances count: {df_2d_noncancer['StudyInstanceUID'].nunique()}"
        )
        print(
            f"Unique 2D Noncancer patient instances count: {df_2d_noncancer['PatientID'].nunique()}"
        )
        print(
            f"Nonunique 2D Noncancer study instances count: {len(df_2d_noncancer.index) - df_2d_noncancer['StudyInstanceUID'].nunique()}"
        )
        merged = df_2d_noncancer.merge(df_2d, on="StudyInstanceUID", how="inner")
        print(
            f"Sanity Check: number of 2D noncancer Studies that match with 2D: {merged.StudyInstanceUID.nunique()}\n"
        )

        print(f"3d Noncancer cases count: {len(df_3d_noncancer.index)}")
        print(
            f"Unique 3D Noncancer study instances count {df_3d_noncancer['StudyInstanceUID'].nunique()}"
        )
        print(
            f"Unique 3D Noncancer patient instances count {df_3d_noncancer['PatientID'].nunique()}"
        )
        print(
            f"Nonunique 3D Noncancer study instances count {len(df_3d_noncancer.index) - df_3d_noncancer['StudyInstanceUID'].nunique()}"
        )
        merged = df_3d_noncancer.merge(
            df_2d_noncancer, on="StudyInstanceUID", how="inner"
        )
        print(
            f"number of 3D noncancer studies that match with those of 2D noncancer: {merged.StudyInstanceUID.nunique()}"
        )
        merged = df_3d_noncancer.merge(df_2d_noncancer, on="PatientID", how="inner")
        print(
            f"number of 3D noncancer patients that match with those of 2D noncancer: {merged.PatientID.nunique()}\n"
        )
