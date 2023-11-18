import glob
import os
import pydicom as dicom
import pandas as pd
from multiprocessing import Pool
import time


class Data:
    files_2D = []
    files_3D = []
    folderPath = "None"
    df = pd.DataFrame()

    def __init__(self, path=None, folder=None):
        # sets the Data path variable
        self.path = path

        # sets the images acquired from which source/folder value
        self.folder = folder

        # set the folder path
        self.folderPath = str(self.path) + "/" + str(self.folder)

        self.df = pd.read_csv(
            os.path.join("/raid/data01/deephealth/labels", self.folder + "_labels.csv")
        )
        self.df = self.df.drop_duplicates()

        self.files_2D = sorted([f for f in glob.glob(str(self.folderPath) + "/*/DXm*")])
        self.files_3D = sorted([f for f in glob.glob(str(self.folderPath) + "/*/BT*")])

    def getLabel(self, studyInstanceUID, imageLaterality):
        value = self.df[(self.df["StudyInstanceUID"] == str(studyInstanceUID))]

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

    def _cal_2DLabel(self, eachFile):
        ds = dicom.filereader.dcmread(
            eachFile,
            stop_before_pixels=True,
            force=True,
            specific_tags=[
                "StudyInstanceUID",
                "ImageLaterality",
                "PatientAge",
                "Rows",
                "Columns",
                "Manufacturer",
                "Modality",
            ],
        )

        studyInstanceUID = ds.get("StudyInstanceUID")
        imageLaterality = ds.get("ImageLaterality")
        value = self.getLabel(str(studyInstanceUID), str(imageLaterality))
        df = pd.DataFrame(
            [
                {
                    "folder": str(self.folder),
                    "image": "2D",
                    "StudyInstanceUID": str(studyInstanceUID),
                    "Imagelaterality": str(imageLaterality),
                    "label": str(value),
                    "PatientAge": str(ds.get("PatientAge")),
                    "Shape": (ds.Rows, ds.Columns),
                    "Manufacturer": str(ds.get("Manufacturer")),
                    "Modality": str(ds.get("Modality")),
                }
            ]
        )

        return df

    def _cal_3DLabel(self, eachFile):
        ds = dicom.filereader.dcmread(
            eachFile,
            stop_before_pixels=True,
            force=True,
            specific_tags=[
                "StudyInstanceUID",
                "SharedFunctionalGroupsSequence",
                "PatientAge",
                "Rows",
                "Columns",
                "NumberOfFrames",
                "Manufacturer",
                "Modality",
            ],
        )

        studyInstanceUID = ds.get("StudyInstanceUID")

        # image laterality information gor 3D image
        frameLaterality = (
            ds.SharedFunctionalGroupsSequence[0].FrameAnatomySequence[0].FrameLaterality
        )

        # label
        value = self.getLabel(str(studyInstanceUID), str(frameLaterality))
        df = pd.DataFrame(
            [
                {
                    "folder": str(self.folder),
                    "image": "3D",
                    "StudyInstanceUID": str(studyInstanceUID),
                    "Imagelaterality": str(frameLaterality),
                    "label": str(value),
                    "PatientAge": str(ds.get("PatientAge")),
                    "Shape": (ds.Rows, ds.Columns, ds.NumberOfFrames),
                    "Manufacturer": str(ds.get("Manufacturer")),
                    "Modality": str(ds.get("Modality")),
                }
            ]
        )

        return df

        # multiprocessing,  here no of tasks requestede is 10

    def run(self, method, listValue):
        with Pool(10) as p:
            df = p.map(method, listValue)
            result_df = pd.concat(df)
            return result_df

    def createRecord(self, executionTime=False):
        t0 = time.time()

        labels_2D = self.run(self._cal_2DLabel, self.files_2D)
        labels_3D = self.run(self._cal_3DLabel, self.files_3D)

        result_df = pd.concat([labels_2D, labels_3D], ignore_index=True, sort=False)

        if executionTime:
            print("...took ", time.time() - t0, "seconds")

        return result_df
