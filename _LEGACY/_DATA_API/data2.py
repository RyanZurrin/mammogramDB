import matplotlib.pyplot as plt
import os
import concurrent
import pylibjpeg
import pydicom as dicom
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import requests
from enum import Enum, auto
import pprint
import multiprocessing
import pickle
from types import SimpleNamespace
import time

label_types = {
    "CANCER": "IndexCancer",
    "NOCANCER": "NonCancer",
    "PRECANCER": "PreIndexCancer",
}
Label = SimpleNamespace(**label_types)


class Data2:
    """
    Data class used to represent and explore dicom data    ...

    Attributes
    ----------
    total_2d_nocancer : int
        total number of 2D dicoms with a label of NonCancer
    total_2d_cancer : int
        total number of 2D dicoms with a label of IndexCancer
    total_2d_preindex : int
        total number of 3D dicoms with a label of PreIndexCancer
    total_3d_nocancer : int
        total number of 3D dicoms with a label of NonCancer
    total_3d_cancer : int
        total number of 3D dicoms with a label of IndexCancer
    total_3d_preindex : int
        total number of 3D dicoms with a label of PreIndexCancer
    df : dataframe
        dataframe containing:
        folder, image, StudyInstanceUID, Imagelaterality, label, PatientAge, Shape, Manufacturer, Modality
    :stats : dictionary
        prints the information contained within the Data class object
    :df_list : []->dataframes
        dataframe that holds the label information for each study in its corresponding index
        paths[0] => csv_paths[0] => df_list[0]
    :study_list : []->str
        list of all the studies in each folder held at its corresponding index paths[0] => study_list[0]
    :images : str
        sorted list of paths to all the dicom images
    """

    __files_2D = []  # all the 2D dicoms
    __files_3D = []  # all the 3D dicoms
    __files_per_study = (
        []
    )  # lists containing all the files in each study, paths[0] => __files_per_study[0]
    __files_2D_per_study = (
        []
    )  # lists containing all the 2D files in each study, paths[0] => __files_2D_per_study[0]
    __files_3D_per_study = (
        []
    )  # lists containing all the 3Dfiles in each study, paths[0] => __files_3D_per_study[0]
    __images = (
        []
    )  # all the sorted images from all the studies and this is the list that ithe Image_ID is derived from
    __master_dict_idx = (
        {}
    )  # sorted dictionary containing all the studies by index sequence
    __master_dict_uid = {}  # sorted dictionary containing all the studies bu patientUID

    stats = (
        {}
    )  # dictionary of the stats about all the data contained in the Data object instance
    df = (
        pd.DataFrame()
    )  # loaded dataframe from a pickle that contains all the label and dicom data used
    df_list = (
        []
    )  # list of dataframes that relate to each individual study, paths[0] => df_list[0]
    study_list = (
        []
    )  # study lists of all the series seperated by each study, paths[0] => study_list[0]

    def __init__(
        self,
        paths=None,
        csv_paths=None,
        pickle_path=None,
        timing=False,
        print_data=False,
    ):
        """
        Initializes the Data class and prepares the data for exploritory jupyter notebook sessions

        Parameters
        ----------
        paths : []->str, optional
            a list of the paths to the dicom study data.
            (default is dicom data in the deephealth mamography datasets)

        csv_paths : []->str, optional
            a list of the csv file paths that go with each study passed into paths so, paths[0] => csv_paths[0]
            the paths and csv_paths are set up as parallel lists and that is how the lists in this class opperate
            in general
            (default is csv files that are associated with the deephealth mamography datasets)

        pickle_path : str, optional
            a path to the pickle data that is used to load the self.df
            (default is None which sets it to a default path r'../allData' where the pickle is located for this project

        timing : float, optional
            sets the timers in the initializing methods to true, and prints out how long each one took
            (default is False)

        print_data : bool, optional
            prints out the detailed dictionary of information about all the studies and the combined data as well
            (default is True)
        """

        global t0
        if timing:
            t0 = time.time()

        # sets the Data path variable and csv paths variables to the defaults
        if paths is None:
            paths = [
                r"/raid/data01/deephealth/dh_dh2/",
                r"/raid/data01/deephealth/dh_dh0new/",
                r"/raid/data01/deephealth/dh_dcm_ast/",
            ]
        if csv_paths is None:
            csv_paths = [
                r"/raid/data01/deephealth/labels/dh_dh2_labels.csv",
                r"/raid/data01/deephealth/labels/dh_dh0new_labels.csv",
                r"/raid/data01/deephealth/labels/dh_dcm_ast_labels.csv",
            ]
        if pickle_path is None:
            pickle_path = r"../../allData"

        self.paths = paths
        self.csv_paths = csv_paths
        self.pickle_path = pickle_path

        # initialize all the data into the class
        self.__initialize(bool(timing))

        if timing:
            print(
                "{:<15s}{:<10s}{:>10f}{:^5s}".format(
                    "total _init_ ", "...took ", time.time() - t0, " seconds"
                )
            )
        if print_data:
            pprint.pprint(self.stats)

    def __initialize(self, timing=False):
        """
        Private method to call the initializing methods to load the Data class with the proper data
        """
        self.__loadStudies(timing=timing)
        self.__loadPickleData(timing=timing)
        self.__loadDfs(timing=timing)
        self.__loadImages(timing=timing)
        self.__populateDicomPaths(timing=timing)
        self.__generateFilteredDataCounts(timing=timing)
        self.__generateStats(timing=timing)

        # loads the study ID's in a sorted order for each study path passed to constructor

    def __loadStudies(self, timing=False):
        """
        Private method to load the studies into dictionaries which are then used for fast lookups
        """
        t0 = time.time()
        if self.csv_paths is None or len(self.csv_paths) == 0:
            print("no study files loaded")
            return
        # load the sorted study from list of paths
        for path in self.paths:
            study = sorted(os.listdir(path))
            self.study_list.append(study)

        studyID = 0
        for i in range(0, len(self.study_list)):
            for item in self.study_list[i]:
                self.__master_dict_idx[studyID] = {"UID": item, "path": self.paths[i]}
                self.__master_dict_uid[item] = {
                    "path": self.paths[i],
                    "Study_ID": studyID,
                }
                studyID = studyID + 1
        if timing is True:
            print(
                "{:<15s}{:<10s}{:>10f}{:^5s}".format(
                    "load studies", "...took ", time.time() - t0, " seconds"
                )
            )

    def __loadPickleData(self, timing=False):
        """
        Private method to load the pickle data into the dataframe df
        """
        t0 = time.time()
        picklefile = open(self.pickle_path, "rb")
        # unpickle the dataframe
        df_result = pickle.load(picklefile)
        # close file
        picklefile.close()
        self.df = df_result
        if timing is True:
            print(
                "{:<15s}{:<10s}{:>10f}{:^5s}".format(
                    "load pickle", "...took ", time.time() - t0, " seconds"
                )
            )

    # private method that populates the the list of dataframes with the proper csv file data
    def __loadDfs(self, timing=False):
        """
        Private method to load the csv files which contain the label data
        """
        t0 = time.time()
        if self.csv_paths == None or len(self.csv_paths) == 0:
            print("no csv files loaded")
            return
        frames = []
        # loads the csv files for each study
        for path in self.csv_paths:
            df = pd.read_csv(path)
            df.drop_duplicates(
                subset=["StudyInstanceUID", "CancerLaterality", "Label"], inplace=True
            )
            self.df_list.append(df)
            frames.append(df)
        self.df_list.append(pd.concat(frames))
        self.df_list[-1].sort_values(by=["StudyInstanceUID"], inplace=True)
        self.df_list[-1].drop_duplicates(subset="StudyInstanceUID")

        if timing is True:
            print(
                "{:<15s}{:<10s}{:>10f}{:^5s}".format(
                    "load dfs", "...took ", time.time() - t0, " seconds"
                )
            )

    def __loadImages(self, timing=False):
        """
        Private method to populate a list with all the dicom image paths from the study and then sorts the images
        """
        t0 = time.time()
        for current_path in self.paths:
            for path, currentDirectory, files in os.walk(current_path):
                for file in files:
                    if file.startswith("BT"):
                        self.__images.append(path + "/" + str(file))
                        self.__files_3D.append(path + "/" + str(file))
                    elif file.startswith("DXm"):
                        self.__images.append(path + "/" + str(file))
                        self.__files_2D.append(path + "/" + str(file))
            self.__images = sorted(self.__images)  # , key = lambda x: x[-48:])
            self.__files_3D = sorted(self.__files_3D)
            self.__files_2D = sorted(self.__files_2D)

        if timing is True:
            print(
                "{:<15s}{:<10s}{:>10f}{:^5s}".format(
                    "load images", "...took ", time.time() - t0, " seconds"
                )
            )

    # private method to calcualtes the total number of 2D and 3D dicoms in each of the studies
    def __populateDicomPaths(self, timing=False):
        """
        Private method to calculate the counts of 2D and 3D dicomes as well as populate the paths into
        """
        t0 = time.time()
        i = 0
        for study_path in self.paths:
            file2D = []
            file3D = []

            for path, currentDirectory, files in os.walk(study_path):
                for file in files:
                    if file.startswith("DXm"):
                        file2D.append(path + "/" + file)
                    if file.startswith("BT"):
                        file3D.append(path + "/" + file)
            self.__files_2D_per_study.append(file2D)
            self.__files_3D_per_study.append(file3D)
            self.__files_per_study.append(file2D + file3D)

        if timing is True:
            print(
                "{:<15s}{:<10s}{:>10f}{:^5s}".format(
                    "populate paths", "...took ", time.time() - t0, " seconds"
                )
            )

    def __generateFilteredDataCounts(self, timing=False):
        """
        Private method to generate the counts of all the possible filtered scenarios that are used in the
        next_image generator
        """
        t0 = time.time()
        filtered_volume = np.where(
            (self.df["label"] == "NonCancer") & (self.df["image"] == "2D")
        )
        self.total_2d_nocancer = len(self.df.loc[filtered_volume]) - 1

        filtered_volume = np.where(
            (self.df["label"] == "NonCancer") & (self.df["image"] == "3D")
        )
        self.total_3d_nocancer = len(self.df.loc[filtered_volume]) - 1

        filtered_volume = np.where(
            (self.df["label"] == "IndexCancer") & (self.df["image"] == "2D")
        )
        self.total_2d_cancer = len(self.df.loc[filtered_volume]) - 1

        filtered_volume = np.where(
            (self.df["label"] == "IndexCancer") & (self.df["image"] == "3D")
        )
        self.total_3d_cancer = len(self.df.loc[filtered_volume]) - 1

        filtered_volume = np.where(
            (self.df["label"] == "PreIndexCancer") & (self.df["image"] == "2D")
        )
        self.total_2d_preindex = len(self.df.loc[filtered_volume]) - 1

        filtered_volume = np.where(
            (self.df["label"] == "PreIndexCancer") & (self.df["image"] == "3D")
        )
        self.total_3d_preindex = len(self.df.loc[filtered_volume]) - 1

        filtered_volume = np.where(
            (self.df["label"] == "IndexCancer")
            & ((self.df["image"] == "3D") | (self.df["image"] == "3D"))
        )
        self.total_2D3d_nocancer = len(self.df.loc[filtered_volume]) - 1

        filtered_volume = np.where(
            (self.df["label"] == "PreIndexCancer")
            & ((self.df["image"] == "2D") | (self.df["image"] == "3D"))
        )
        self.total_2d3D_cancer = len(self.df.loc[filtered_volume]) - 1

        filtered_volume = np.where(
            (self.df["label"] == "PreIndexCancer")
            & ((self.df["image"] == "3D") | (self.df["image"] == "3D"))
        )
        self.total2d3d_preindex = len(self.df.loc[filtered_volume]) - 1

        if timing is True:
            print(
                "{:<15s}{:<10s}{:>10f}{:^5s}".format(
                    "filtered dfs", "...took ", time.time() - t0, " seconds"
                )
            )

    # private method to calcualtes the total number of 3D dicoms in each of the studies
    def __generateStats(self, timing=False):
        """
        Private method used to generate all the stats about the different studies being included in the Data instance
        """
        t0 = time.time()
        tot_cancer = 0
        tot_preindex = 0
        tot_all_dicoms = 0
        tot_all_2d = 0
        tot_all_3d = 0
        for i in range(0, len(self.paths)):
            count2d = None
            count3d = None
            folder = "folder" + str(i)
            total = len(self.__files_per_study[i])
            _2D = len(self.__files_2D_per_study[i])
            _3D = len(self.__files_3D_per_study[i])
            cancer_2d = np.where(
                (self.df["folder"].str.contains(self.paths[i]))
                & (self.df["label"] == "IndexCancer")
                & (self.df["image"] == "2D")
            )
            c2d = len(self.df.loc[cancer_2d])
            cancer_3d = np.where(
                (self.df["folder"].str.contains(self.paths[i]))
                & (self.df["label"] == "IndexCancer")
                & (self.df["image"] == "3D")
            )
            c3d = len(self.df.loc[cancer_3d])

            pre_2d = np.where(
                (self.df["folder"].str.contains(self.paths[i]))
                & (self.df["label"] == "PreIndexCancer")
                & (self.df["image"] == "2D")
            )
            p2d = len(self.df.loc[pre_2d])
            pre_3d = np.where(
                (self.df["folder"].str.contains(self.paths[i]))
                & (self.df["label"] == "PreIndexCancer")
                & (self.df["image"] == "3D")
            )
            p3d = len(self.df.loc[pre_3d])
            tot_cancer += c2d + c3d
            tot_preindex += p2d + p3d
            tot_all_dicoms += total
            tot_all_2d += _2D
            tot_all_3d += _3D
            count2d = {
                "cancer": c2d,
                "pre_index": p2d,
                "non_cancer": (_2D - (c2d - p2d)),
            }
            count3d = {
                "cancer": c3d,
                "pre_index": p3d,
                "non_cancer": (_3D - (c3d - p3d)),
            }
            case = {
                "total": total,
                "3D": _3D,
                "2D": _2D,
                "label_2D": count2d,
                "labels_3D": count3d,
            }
            self.stats[folder] = case
        self.stats["total_all_dicoms"] = tot_all_dicoms
        self.stats["total_2D_dicoms "] = tot_all_2d
        self.stats["total_3D_dicoms "] = tot_all_3d
        self.stats["total_cancer    "] = tot_cancer
        self.stats["total_preindex  "] = tot_preindex
        self.stats["total_noncancer "] = tot_all_dicoms - (tot_cancer - tot_preindex)

        if timing is True:
            print(
                "{:<15s}{:<10s}{:>10f}{:^5s}".format(
                    "gen stats", "...took ", time.time() - t0, " seconds"
                )
            )

    def get_image_id(self, imagePath, timing=False):
        """
        Gets the ImageID which can be used in the get_image method

        Parameters
        ----------
        imagePath : str
        Can pass in the full path or the file name of dicom info and will find the imageID you can use to
        locate image

        Returns : int
        -------
        The Image_ID of dicom at specified path

        """
        t0 = time.time()
        index = [idx for idx, s in enumerate(self.__images) if imagePath in s][0]
        if timing is True:
            print(
                "{:<15s}{:<10s}{:>10f}{:^5s}".format(
                    "getImage", "...took ", time.time() - t0, " seconds"
                )
            )
        return index

    # gets the file names in a series
    def get_file_names_in_series(self, StudyInstanceUID, timing=False):
        """
        Gets the dicom file names for a particular study based on the StudyInstanceUID

        Parameters
        ----------
        StudyInstanceUID : str
        Uses the StudyInstanceUID to locate the study and find all the dicom images contained in the study

        timing : bool, optional
        Sets timing flag, if true will time execution time of method, else will not (default is False)

        Returns : list
        -------
        List of all the dicom files names in a study
        """
        t0 = time.time()
        path = self.__master_dict_uid[StudyInstanceUID]["path"]
        series_path = path + "/" + str(StudyInstanceUID)
        file_paths = os.listdir(series_path)
        if timing is True:
            print(
                "{:<15s}{:<10s}{:>10f}{:^5s}".format(
                    "get_file_names_in_series", "...took ", time.time() - t0, " seconds"
                )
            )
        return file_paths

    def get_label(self, studyInstanceUID, imageLaterality, study=-1, timing=False):
        """
        Gets the labele of a particular  study based from the studyInstanceUID and the imageLaterality

        Parameters
        ----------
        param studyInstanceUID : str
        Uses the StudyInstanceUID to locate the label in the appropriate csv file

        param imageLaterality : str
        Used the ImageLaterality data to verify we are looking at the right label as well

        param study : int, optional
        Tells the method which csv_files index to look into (default is -1 which is a list of all the csv data in one df)

        param timing : bool, optional
        Sets timing flag, if true will time execution time of method, else will not (default is False)

        :Returns : str
        -------
        Will return one of the following: NonCancer, IndexCancer, PreIndexCancer, None
        """
        t0 = time.time()
        value = self.df_list[study][
            (self.df_list[study]["StudyInstanceUID"] == str(studyInstanceUID))
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
        if timing is True:
            print(
                "{:<15s}{:<10s}{:>10f}{:^5s}".format(
                    "get_label", "...took ", time.time() - t0, " seconds"
                )
            )

    def get_study(
        self, StudyInstanceUID: str = None, ID: int = None, verbose=False, timing=False
    ):
        """
        Get a study from either the Study id or our own id which is pased off of the sorted index of where
        study is saved in list

        Parameters
        ----------
        StudyInstanceUID : str, optional
        Uses the StudyInstanceUID to locate the Study information and returns data as dictionary
        (default is None, so either this or the ID needs to be set)

        ID : int, optional
        Uses the ID of a study which is based off of its index in the sorted list of all studies
        (default is None, so either this or the StudyInstanceUID needs to be set)

        verbose : bool, optional
        If this is set to True it will also include some additonal information about the study such as the
        number of 3D and 2D dicoms in study

        timing : bool, optional
        Sets timing flag, if true will time execution time of method, else will not (default is False)

        :Returns : dictionary
        -------
        a dictionary of data about the specified study such as:
        {'directory': XX, 'images': [imagefile1, imagefile2, ...]}
        (verbose==True)=>info': {'3D_count': XX, '2D_count': XX ..and others}

        """
        t0 = time.time()
        directory = ""
        image_files = []
        images = []
        sid = ""
        uid = ""
        if StudyInstanceUID is not None:
            uid = StudyInstanceUID
            sid = self.__master_dict_uid[StudyInstanceUID]["Study_ID"]
            directory = self.__master_dict_uid[StudyInstanceUID]["path"] + str(
                StudyInstanceUID
            )
            image_files = self.get_file_names_in_series(StudyInstanceUID)
        elif ID is not None:
            sid = ID
            uid = self.__master_dict_idx[ID]["UID"]
            directory = (
                self.__master_dict_idx[ID]["path"] + self.__master_dict_idx[ID]["UID"]
            )
            image_files = self.get_file_names_in_series(
                self.__master_dict_idx[ID]["UID"]
            )

        files_3D = [i for i in image_files if "BT" in i]
        files_2D = [i for i in image_files if "DXm" in i]

        if verbose:
            info = {
                "total": len(files_3D) + len(files_2D),
                "3D count": len(files_3D),
                "2D count": len(files_2D),
            }
        else:
            info = None

        study = {
            "directory": directory,
            "Study_ID": sid,
            "UID": uid,
            "images": image_files,
            "info": info,
        }
        dotDictionary = SimpleNamespace(**study)
        if timing is True:
            print(
                "{:<15s}{:<10s}{:>10f}{:^5s}".format(
                    "getStudy", "...took ", time.time() - t0, " seconds"
                )
            )
        return dotDictionary

    def get_image(
        self,
        Image_ID=None,
        Dicom_ID=None,
        path=None,
        pixels=True,
        dicom_header=False,
        timing=False,
    ):
        """
        Gets the pixels of an image as well as some additional data as specified

        Parameters
        ----------
        Image_ID : int, optional
        Uses the Image_ID to locate the Dicom information and returns the pixel data as numpy array if pixels is True
        (default is None, so either this or the Dicom_ID or path needs to be set)

        Dicom_ID : str, optional
        Uses the Dicom_ID which is the file name of the Dicom, and starts with BD or DXm, to locate the Dicom
        information and returns the pixel data as numpy array if pixels is True
        (default is None, so either this or the Dicom_ID or path needs to be set)

        path : str, optional
        Uses the path of a Dicom image to process it and return the pixel data as numpy array if pixels is True
        (default is None, so either this or the Image_ID or Image_ID needs to be set)

        pixels : bool, optional
        If True will return the pixel array of image, else will return None

        dicom_header : bool, optional
        If True will return the dicom header as a dictionary, else will return None

        timing : bool, optional
        Sets timing flag, if true will time execution time of method, else will not (default is False)

        :Returns : NumPy Array, dictionary
        -------
        [np.array just the pixels or None if pixels=False, info]
        info is dictionary: {'label': LABELS.Cancer, 'filepath':... 'shape'...}
        """
        t0 = time.time()
        if Image_ID is not None:
            img_path = self.__images[Image_ID]
            ds = dicom.dcmread(img_path, force=True)
        elif path is not None:
            ds = dicom.dcmread(path, force=True)
            img_path = path
        elif Dicom_ID is not None:
            img_path = "".join([s for s in self.__images if Dicom_ID in s])
            ds = dicom.dcmread(img_path, force=True)
        else:
            "Error: No image path or location ID was specified"
            return
        studyInstanceUID = ds.get("StudyInstanceUID")

        img = None
        metadata = None
        if "BT" in img_path:
            imageLaterality = (
                ds.SharedFunctionalGroupsSequence[0]
                .FrameAnatomySequence[0]
                .FrameLaterality
            )
            imageShape = (int(ds.NumberOfFrames), int(ds.Rows), int(ds.Columns))
        else:
            imageLaterality = ds.get("ImageLaterality")
            imageShape = (int(ds.Rows), int(ds.Columns))

        label = self.get_label(studyInstanceUID, imageLaterality)
        if pixels is True:
            img = ds.pixel_array

        if not dicom_header:
            ds = None

        dictionary = {
            "label": label,
            "filePath": img_path,
            "shape": imageShape,
            "metadata": ds,
        }

        if timing is True:
            print(
                "{:<15s}{:<10s}{:>10f}{:^5s}".format(
                    "get_images", "...took ", time.time() - t0, " seconds"
                )
            )
        return img, dictionary

    def next_image(self, _2D=False, _3D=False, label=None, timing=False):
        """
        Generator to filter and return iterativly all the filtered images

        Parameters
        ----------
        _2D : int, optional
        Filter flag used to add 2D images to returned set of Dicoms
        (default is None, so either this or the _3D or both needs to be set)

         _3D : str, optional
        Filter flag used to add 3D images to returned set of Dicoms
        (default is None, so either this or the _2D or both needs to be set)

        label : str, optional
        Uses the path of a Dicom image to process it and return the pixel data as numpy array if pixels is True
        (default is None whcih is all images, additional options are IndexCancer, NonCancer, PreIndexCancer)

        timing : bool, optional
        Sets timing flag, if true will time execution time of method, else will not (default is False)

        :Returns : NumPy Array, dictionary
        -------
        [np.array just the pixels, info]
        info is dictionary: {'label': LABELS.Cancer, 'filepath':... 'shape'...}
        """
        t0 = time.time()
        imageLaterality = None
        imageShape = None
        filtered_volume = pd.DataFrame()
        if _2D and _3D:
            if label is None:
                filtered_volume = self.df
            else:
                filtered_volume = np.where((self.df["label"] == label))
        elif _2D:
            if label is None:
                filtered_volume = np.where((self.df["image"] == "2D"))
            else:
                filtered_volume = np.where(
                    (self.df["label"] == label) & (self.df["image"] == "2D")
                )
        elif _3D:
            if label is None:
                filtered_volume = np.where((self.df["image"] == "3D"))
            else:
                filtered_volume = np.where(
                    (self.df["label"] == label) & (self.df["image"] == "3D")
                )
        else:
            print("error: incorrect flags")
        new_df = self.df.loc[filtered_volume]
        print("filtered data size: ", len(new_df))

        for i in range(len(new_df)):
            try:
                path = new_df["folder"].iloc[i]
                ds = dicom.filereader.dcmread(
                    path,
                    force=True,
                    specific_tags=[
                        "SamplesPerPixel",
                        "PhotometricInterpretation",
                        "PlanarConfiguration",
                        "Rows",
                        "Columns",
                        "PixelAspectRatio",
                        "BitsAllocated",
                        "BitsStored",
                        "HighBit",
                        "PixelRepresentation",
                        "SmallestImagePixelValue",
                        "LargestImagePixelValue",
                        "PixelPaddingRangeLimit",
                        "RedPaletteColorLookupTableDescriptor",
                        "GreenPaletteColorLookupTableDescriptor",
                        "BluePaletteColorLookupTableDescriptor",
                        "RedPaletteColorLookupTableData",
                        "GreenPaletteColorLookupTableData",
                        "BluePaletteColorLookupTableData",
                        "ICCProfile",
                        "ColorSpace",
                        "PixelDataProviderURL",
                        "ExtendedOffsetTable",
                        "NumberOfFrames",
                        "ExtendedOffsetTableLengths",
                        "PixelData",
                    ],
                )
                img = ds.pixel_array
                i += 1
                dictionary = {
                    "filePath": path,
                    "label": new_df["label"].iloc[i],
                    "laterality": imageLaterality,
                    "shape": ds.pixel_array.shape,
                }
                yield img, dictionary
            except StopIteration:
                break

        if timing is True:
            print(
                "{:<15s}{:<10s}{:>10f}{:^5s}".format(
                    "next_image", "...took ", time.time() - t0, " seconds"
                )
            )

    # views the dicom image, will work with both 2d and 3d dicoms with no need to specifiy which one is
    # being loaded. For 3D dicoms it is looking at the center slice
