import matplotlib.pyplot as plt
import glob
import os
import pylibjpeg
import pydicom as dicom
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import datetime
import requests
import multiprocessing
import pickle
import itertools as it
from types import SimpleNamespace
from collections import Counter

import time


# Data class
# constructor: no argument will read within the current paths directory
# constructor: path as argument to acquire data from a specified location
class Data:
    """ """

    df_list = []  # list of label csv files for studies
    # df2 = [] # used to store the duplicates incase you need to determine what ones they are
    study_list = []  # sorted list of series folders for each study
    images = []
    master_dict_idx = (
        {}
    )  # sorted master list containing all the studies by index sequence
    master_dict_uid = {}  # sorted master list containing all the studies bu patientUID

    total_series_per_study_list = []  # list of total number of series in each study
    total_dicoms_per_study_list = []  # list of total dicoms in each study
    total_2D_dicoms_per_study_list = []  # list of total 2D dicoms in each study
    total_3D_dicoms_per_study_list = []  # list of total 3D dicoms in each study
    stats = {}

    cancer_2D_list = []  # list of only the 2D cancer dicom paths
    cancer_3D_list = []  # list of only the 3D cancer dicom paths
    noncancer_2D_list = []  # list of only the 2D noncancer paths
    noncancer_3D_list = []  # list of only the 3D noncancer paths
    preindex_2D_list = []
    preindex_3D_list = []
    none_2D_list = []
    none_3D_list = []

    # creates the sorted list of series as well as counts the 2D and 3D dicoms -> O(n^2 * log n)
    def __init__(
        self,
        paths=[
            r"/raid/data01/deephealth/dh_dcm_ast/",
            r"/raid/data01/deephealth/dh_dh0new/",
            r"/raid/data01/deephealth/dh_dh2/",
        ],
        csv_paths=[
            r"/raid/data01/deephealth/labels/dh_dcm_ast_labels.csv",
            r"/raid/data01/deephealth/labels/dh_dh0new_labels.csv",
            r"/raid/data01/deephealth/labels/dh_dh2_labels.csv",
        ],
        timing=False,
        print_data=True,
    ):
        # sets the Data path variable and csv paths variables
        self.paths = paths
        self.csv_paths = csv_paths
        if timing:
            t0 = time.time()

        # sets the total series variable so when the getSeriesCount() mehtod is called it does
        # not have to recompute

        self.__loadDfs(timing=True)
        self.__loadStudies(timing=True)
        self.__calculateSeriesCounts(timing=True)
        self.__calculateDicoms(timing=True)
        self.__calculateStats(timing=True)
        self.__loadImages(timing=True)
        #         self.files_2D_study0 = sorted([f for f in glob.glob(str(paths[0])+ '/*/DXm*')])
        #         self.files_3D_study0 = sorted([f for f in glob.glob(str(paths[0])+ '/*/BT*')])
        #         self.files_2D_study1 = sorted([f for f in glob.glob(str(paths[1])+ '/*/DXm*')])
        #         self.files_3D_study1 = sorted([f for f in glob.glob(str(paths[1])+ '/*/BT*')])
        #         self.files_2D_study2 = sorted([f for f in glob.glob(str(paths[1])+ '/*/DXm*')])
        #         self.files_3D_study2 = sorted([f for f in glob.glob(str(paths[1])+ '/*/BT*')])
        #         #self.__populateImages()
        #         self.__populate2D()
        #         self.__populate3D()
        if timing:
            print("...took ", time.time() - t0, "seconds")

    # returns the series folder at a specified index which will always be the same. Uses timsort
    # algorithm which has been used since python 2.3 and is the defacto standard for python across
    # different OS systems as well. It is a stable sort and is reliable to maintain our data as we need.
    # @returns
    #     def getUID(self, series_number, timing = False):
    #         '''
    #         '''
    #         t0 = time.time() # for time start of method
    #         if series_number < 0 or series_number > (len(self.__sorted_files) - 1):
    #             return "out of bounds"
    #         series_folder = self.__sorted_files[series_number]
    #         if timing is True: # if timing is True then print the time the method took to run
    #             print("...took ", time.time()-t0, "seconds")
    #         return series_folder

    # private method that populates the the list of dataframes with the proper csv file data
    def __loadDfs(self, timing=False):
        """ """
        t0 = time.time()
        if self.csv_paths == None or len(self.csv_paths) == 0:
            print("no csv files loaded")
            return
        frames = []
        # loads the csv files for each study
        for path in self.csv_paths:
            df = pd.read_csv(path)
            # df_2 = df[df[['StudyInstanceUID', 'CancerLaterality','Label']].duplicated()]
            # df_2.sort_values(by = ['StudyInstanceUID'], inplace=True)
            # self.df2.append(df_2)
            df.drop_duplicates(
                subset=["StudyInstanceUID", "CancerLaterality", "Label"], inplace=True
            )
            self.df_list.append(df)
            frames.append(df)
        self.df_list.append(pd.concat(frames))
        self.df_list[-1].sort_values(by=["StudyInstanceUID"], inplace=True)
        self.df_list[-1].drop_duplicates(subset="StudyInstanceUID")
        if timing is True:
            print("...took ", time.time() - t0, "seconds")

    # loads the study ID's in a sorted order for each study path passed to constructor
    def __loadStudies(self, timing=False):
        """ """
        t0 = time.time()
        if self.csv_paths == None or len(self.csv_paths) == 0:
            print("no study files loaded")
            return
        # load the sorted study from list of paths
        for path in self.paths:
            study = sorted(os.listdir(path))
            self.study_list.append(study)

        # self.master = sorted(it.chain(*self.study_list))
        studyID = 0
        for i in range(0, len(self.study_list)):
            for item in self.study_list[i]:
                self.master_dict_idx[studyID] = {"UID": item, "path": self.paths[i]}
                self.master_dict_uid[item] = {
                    "path": self.paths[i],
                    "Study_ID": studyID,
                }
                studyID = studyID + 1
        if timing is True:
            print("...took ", time.time() - t0, "seconds")

    # private method to calcualtes the total number of series in a study
    def __calculateSeriesCounts(self, timing=False):
        """ """
        t0 = time.time()
        for study in self.study_list:
            self.total_series_per_study_list.append(len(study))
        if timing is True:
            print("...took ", time.time() - t0, "seconds")

    # private method to calcualtes the total number of 2D dicoms in each of the studies
    def __calculateDicoms(self, timing=False):
        """ """
        t0 = time.time()
        for study_path in self.paths:
            images2D = 0
            images3D = 0
            for path, currentDirectory, files in os.walk(study_path):
                for file in files:
                    if file.startswith("DXm"):
                        images2D = images2D + 1
                    if file.startswith("BT"):
                        images3D = images3D + 1
            self.total_2D_dicoms_per_study_list.append(images2D)
            self.total_3D_dicoms_per_study_list.append(images3D)
            self.total_dicoms_per_study_list.append(images2D + images3D)
        if timing is True:
            print("...took ", time.time() - t0, "seconds")

    # private method to calcualtes the total number of 3D dicoms in each of the studies
    def __calculateStats(self, timing=False):
        """ """
        t0 = time.time()
        for i in range(0, len(self.paths)):
            folder = "folder" + str(i)
            total = self.total_dicoms_per_study_list[i]
            _2D = self.total_2D_dicoms_per_study_list[i]
            _3D = self.total_3D_dicoms_per_study_list[i]
            case = {"total": total, "3D": _3D, "2D": _2D}
            self.stats[folder] = case
        self.stats["total_dicoms   "] = sum(self.total_dicoms_per_study_list)
        self.stats["total_2D_dicoms"] = sum(self.total_2D_dicoms_per_study_list)
        self.stats["total_3D_dicoms"] = sum(self.total_3D_dicoms_per_study_list)

        if timing is True:
            print("...took ", time.time() - t0, "seconds")

    #     # populates a list with all the dicom image paths from the study and then sorts the images
    def __loadImages(self, timing=False):
        """ """
        t0 = time.time()
        for current_path in self.paths:
            for path, currentDirectory, files in os.walk(current_path):
                for file in files:
                    if file.startswith("BT") or file.startswith("DXm"):
                        self.images.append(path + "/" + str(file))
            self.images = sorted(self.images)  # , key = lambda x: x[-48:])
        if timing is True:
            print("...took ", time.time() - t0, "seconds")

    def getImageID(self, imagePath, timing=False):
        t0 = time.time()
        index = [idx for idx, s in enumerate(self.images) if imagePath in s][0]
        if timing is True:
            print("...took ", time.time() - t0, "seconds")
        return index

    # gets the file names in a series
    def getFileNamesInSeries(self, StudyInstanceUID, timing=False):
        """ """
        t0 = time.time()
        path = self.master_dict_uid[StudyInstanceUID]["path"]
        series_path = path + "/" + str(StudyInstanceUID)
        file_paths = os.listdir(series_path)
        if timing is True:
            print("...took ", time.time() - t0, "seconds")
        return file_paths

    # tests that the studyInstanceUID and imageLaterality are correct and returns the labeled data from the df
    # when in an interactive notebook you can pass in the class instance df dataframe. -> (data.df)
    def getLabel(self, studyInstanceUID, imageLaterality, df, timing=False):
        """ """
        t0 = time.time()
        value = df.loc[
            (
                (df["StudyInstanceUID"] == str(studyInstanceUID))
                & (df["CancerLaterality"] == str(imageLaterality))
            ),
            "Label",
        ].tolist()

        if len(value) == 0:
            label = "None"
        elif len(value) == 1:
            label = str(value[-1])
        else:
            print("error")
        if timing is True:
            print("...took ", time.time() - t0, "seconds")
        return label

    # GRAB STUDY
    # same method to call either with study id or own id
    # d.get_study(StudyInstanceUID, verbose=False, timing=False)
    # d.get_study(OUR_OWN_ID, verbose=False, timing=False)
    # returns some meta info dict
    # {
    #  'directory': XX,
    #  'images': [imagefile1, imagefile2],
    #  'info': {'3D_count': XX, '2D_count': XX ..and others} # if verbose=True
    # }
    def get_study(self, StudyInstanceUID=None, ID=None, verbose=False, timing=False):
        """ """
        t0 = time.time()
        directory = ""
        image_files = []
        images = []
        sid = ""
        uid = ""
        if StudyInstanceUID != None:
            uid = StudyInstanceUID
            sid = self.master_dict_uid[StudyInstanceUID]["Study_ID"]
            directory = self.master_dict_uid[StudyInstanceUID]["path"] + str(
                StudyInstanceUID
            )
            image_files = self.getFileNamesInSeries(StudyInstanceUID)
        elif ID != None:
            sid = ID
            uid = self.master_dict_idx[ID]["UID"]
            directory = (
                self.master_dict_idx[ID]["path"] + self.master_dict_idx[ID]["UID"]
            )
            image_files = self.getFileNamesInSeries(self.master_dict_idx[ID]["UID"])

        files_3D = [i for i in image_files if "BT" in i]
        files_2D = [i for i in image_files if "DXm" in i]

        # for image in image_files:
        #    images.append(self.getImageID(image))

        if verbose == True:
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
        ns = SimpleNamespace(**study)
        if timing is True:
            print("...took ", time.time() - t0, "seconds")
        return ns

    # GRAB PIXELS
    ## directory path
    ## list of images
    ##
    # d.get_image(Image_ID, pixels=True) # again both our id and the dicom id and also the image filepath would be nice
    # returns [np.array just the pixels or None if pixels=False, info]
    # info is dictionary like
    # {'label': LABELS.Cancer, 'filepath':... 'shape'...}
    def get_image(
        self,
        Image_ID=None,
        Dicom_ID=None,
        path=None,
        pixels=True,
        dicom_header=False,
        timing=False,
    ):
        """ """
        t0 = time.time()
        if Image_ID != None:
            img_path = self.images[Image_ID]
            ds = dicom.dcmread(img_path)
        elif path != None:
            ds = dicom.dcmread(path)
            img_path = path
        elif Dicom_ID != None:
            img_path = "".join([s for s in self.images if Dicom_ID in s])
            ds = dicom.dcmread(img_path)
        else:
            "Error: No image path or location ID was specified"
            return

        studyInstanceUID = ds.get("StudyInstanceUID")
        imageLaterality = ds.get("ImageLaterality")
        label = self.getLabel(studyInstanceUID, imageLaterality, self.df_list[-1])
        img = None
        metadata = None
        imageShape = ds.pixel_array.shape
        if pixels is True:
            img = np.array(ds.pixel_array)

        if dicom_header is True:
            metadata = ds

        dict = {
            "label": label,
            "file Path": img_path,
            "shape": imageShape,
            "metadata": metadata,
        }

        if timing is True:
            print("...took ", time.time() - t0, "seconds")

        return img, dict

    #
    # generator that returns iteratively all images
    #
    # d.next_image(2D=True,3D=True,Label=Labels.NOCANCER) # with filtering flags
    # yields [np.array just the pixels, info] # same as d.get_image

    #
    # So I want to run
    #
    # for i in range(d.total_2d_nocancer):
    #     d.next_image(2D=True, 3D=False, Label=Labels.NOCANCER)
    # to get all 2D images without cancer


#     def next_image(self, _2D = True, _3D = True, Label=None, timing = False):
#         '''
#         '''
#         t0 = time.time()
#         print("not implemented")
#         if timing is True:
#             print("...took ", time.time()-t0, "seconds")


#     # populates the list of 2D non-cancer and cancer paths to all the dicom images
#     def __populate2D(self, timing = False):
#         t0 = time.time()
#         img = self.__total_3D_dicoms # 3D dicoms are first in the images list so we can start at this index looking for 2D only
#         countLabel  = []
#         for dc in self.images[img:]:
#             ds = dicom.dcmread(dc)
#             studyInstanceUID = ds.get("StudyInstanceUID")
#             imageLaterality = ds.get("ImageLaterality")
#             label = self.getLabel(studyInstanceUID, imageLaterality, self.df)
#             countLabel.append(label)

#             // find count for each unique element in countLabel
# #             if "IndexCancer" not in label:
# #                 self.total_2D_noncancer.append(dc)
# #             else:
# #                 self.total_2D_cancer.append(dc)
#             if "NonCancer" in label:
#                 self.total_2D_noncancer.append(dc)
#             elif:
#                 self.total_2D_noncancer.append(dc)
#             else:
#                 self.total_2D_cancer.append(dc)

#             img = img + 1
#         if timing is True:
#             print("...took ", time.time()-t0, "seconds")

# populates the lists of 3D non-cancer and cancer paths to all the dicom images
#     def __populate3D(self, timing = False):
#         t0 = time.time()
#         img = 0
#         end = self.__total_3D_dicoms
#         for dc in self.images[:end]:
#             ds = dicom.dcmread(dc)
#             studyInstanceUID = ds.get("StudyInstanceUID")
#             imageLaterality = ds.get("ImageLaterality")
#             label = self.getLabel(studyInstanceUID, imageLaterality, self.df)
#             if "IndexCancer" not in label:
#                 self.total_3D_noncancer.append(dc)
#             else:
#                 self.total_3D_cancer.append(dc)
#             img = img + 1
#         if timing is True:
#             print("...took ", time.time()-t0, "seconds")


#     # neha's code
#     # start
#     path = '/raid/data01/deephealth/'

#     folder = 'dh_dcm_ast'
#     labelPath = os.path.join('/raid/data01/deephealth/labels', folder+'_labels.csv')
#     df = pd.read_csv(labelPath)

#     folderPath = str(path) + '/' + str(folder)


#     def getTag(ds, tagValue):
#     tag = ds.get(str(tagValue))
#     return tag

#     def __populate2D(self, df, executionTime=False):
#         t0 = time.time()

#         labels = []
#         for eachFile in files_2D:
#             ds = dicom.dcmread(eachFile)
#             value  = getLabel(getTag( ds, "StudyInstanceUID"), getTag(ds, "ImageLaterality"), self.df)
#             labels.append(value)

#         if executionTime:
#             print("...took ", time.time()-t0, "seconds")
#         print("Total 2D dicoms", len(files_2D))
#         print("Total labels", len(labels))
#         print(Counter(labels))

#         return labels

#     #end code


#     # views a 2D image within a particular series
#     def viewSeries2DImages(self, seriesID, imgNum,timing = False):
#         '''
#         '''
#         t0 = time.time()
#         series_path = self.path + '/' + str(self.getUID(seriesID))
#         onlyfiles = [f for f in listdir(series_path) if isfile(join(series_path, f))]
#         if imgNum < 0 or imgNum > len(onlyfiles):
#             print("only ", len(onlyfiles), " images in series")
#             return None

#         if onlyfiles[imgNum].startswith("B"):
#             print("not a 2D image")

#         ds = dicom.dcmread(str(series_path + '/' + onlyfiles[imgNum]))
#         img = ds.pixel_array

#         plt.imshow(img)
#         if timing is True:
#             print("...took ", time.time()-t0, "seconds")

#     # returns the pixal array of a dicom image
#     def getPixels(self, seriesID, imgNum, timing=False):
#         '''
#         '''
#         t0 = time.time()
#         series_path = self.path + '/' + str(self.getUID(seriesID))
#         onlyfiles = [f for f in listdir(series_path) if isfile(join(series_path, f))]
#         if imgNum < 0 or imgNum > len(onlyfiles):
#             print("only", len(onlyfiles), " images in series")
#             return None
#         ds = dicom.dcmread(str(series_path + '/' + onlyfiles[imgNum]))
#         img = ds.pixel_array
#         if timing:
#             print("...took", time.time()-t0, "seconds")
#         return img

#     # reverse search from the StudyInstanceUID to the index where it is located in the sorted array
#     def findSeriesID(self, StudyInstanceUID):
#         '''
#         '''
#         return self.__sorted_files.index(StudyInstanceUID, 0, self.__total_series)

#     # find the Image_ID in the images folder of a seriesID and imgNum
#     def findImageID(self, seriesID, imgNum):
#         '''
#         '''
#         imgName = self.getFileNamesInSeries(seriesID)[imgNum]
#         lookup = self.getUID(seriesID)+'/'+imgName
#         index = 0
#         for img in self.images:
#             if lookup in img:
#                 break
#             index = index + 1
#         return index


#     # gets the study instance ID, image laterality and shape of the image
#     def getDicomInfo(self, seriesID, imgNum, timing=False):
#         '''
#         '''
#         t0 = time.time()
#         ds = dicom.dcmread(self.path+'/'+self.getUID(seriesID)+'/'+self.getFileNamesInSeries(seriesID)[imgNum])
#         studyInstanceUID = ds.get("StudyInstanceUID")
#         imageLaterality = ds.get("ImageLaterality")
#         shape = ds.pixel_array.shape
#         if timing is True:
#             print("...took ", time.time()-t0, "seconds")
#         return str(studyInstanceUID), str(imageLaterality), shape

#     # gets the image laterality of an image
#     def getImageLaterality(self, series, imgNum, timing=False):
#         '''
#         '''
#         t0 = time.time()
#         ds = dicom.dcmread(self.path+'/'+self.getUID(series)+'/'+self.getFileNamesInSeries(series)[imgNum])
#         imageLaterality = ds.get("ImageLaterality")
#         if timing is True:
#             print("...took ", time.time()-t0, "seconds")
#         return imageLaterality

#     # using the seriesID and imgNum returns true if it is a 2D dicom else returns false
#     def is2D(self, seriesID, imgNum, timing = False):
#         '''
#         '''
#         t0 = time.time()
#         str = self.getFileNamesInSeries(seriesID)[imgNum]
#         if timing is True:
#             print("...took ", time.time()-t0, "seconds")
#         if str.startswith("DXm") is True:
#             return True
#         return False

#     # using the seriesID and imgNum returns true if it is a 3D dicom else returns false
#     def is3D(self, seriesID, imgNum, timing = False):
#         '''
#         '''
#         t0 = time.time()
#         str = self.getFileNamesInSeries(seriesID)[imgNum]
#         if timing is True:
#             print("...took ", time.time()-t0, "seconds")
#         if str.startswith("BT") is True:
#             return True
#         return False

#     # using the Image_ID returns true if it is a 2D dicom else returns false
#     def _2D(self, Image_ID, timing = False):
#         '''
#         '''
#         t0 = time.time()
#         str = self.images[Image_ID]
#         if timing is True:
#             print("...took ", time.time()-t0, "seconds")
#         if "DXm" in str:
#             return True
#         return False

#     # using the Image_ID returns true if it is a 3D dicom else returns false
#     def _3D(self, Image_ID, timing = False):
#         '''
#         '''
#         t0 = time.time()
#         str = self.images[Image_ID]
#         if timing is True:
#             print("...took ", time.time()-t0, "seconds")
#         if "BT" in str:
#             return True
#         return False

#     # views the dicom image, will work with both 2d and 3d dicoms with no need to specifiy which one is
#     # being loaded. For 3D dicoms it is looking at the center slice
#     def view(self, seriesID, imgNum, timing = False):
#         '''
#         '''
#         t0 = time.time()
#         ds = dicom.dcmread(self.path+'/'+self.getUID(seriesID)+'/'+self.getFileNamesInSeries(seriesID)[imgNum])
#         imageShape = ds.pixel_array.shape
#         if self.is3D(seriesID, imgNum) is True:
#             img = ds.pixel_array
#             center_slice = img[img.shape[0] // 2]
#             plt.imshow( center_slice )
#         else:
#             self.viewSeries2DImages(seriesID, imgNum)
#         if timing is True:
#             print("...took ", time.time()-t0, "seconds")
#         return self.getFileNamesInSeries(seriesID)[imgNum]

#     # gets the data related to a dicom image, will work with both 2d and 3d dicoms with no need to specifiy which one is
#     # being loaded
#     def getData(self, series, imgNum, timing = False):
#         '''
#         '''
#         t0 = time.time()
#         ds = dicom.dcmread(self.path+'/'+self.getUID(series)+'/'+self.getFileNamesInSeries(series)[imgNum])
#         studyInstanceUID = ds.get("StudyInstanceUID")
#         imageLaterality = ds.get("ImageLaterality")
#         label = self.getLabel(studyInstanceUID, imageLaterality, self.df)
#         imageShape = ds.pixel_array.shape
#         img = ds.pixel_array
#         if timing is True:
#             print("...took ", time.time()-t0, "seconds")
#         return str(studyInstanceUID), str(imageLaterality), imageShape, label, np.array(img)


#     # returns the informaion and pixal array on a specified image, can specify if you want 2D or 3D
#     def load(self, seriesID, imgNum, df ,dicom_2D = False, dicom_3D = False, timing = False):
#         '''
#         '''
#         t0 = time.time()
#         series_path = self.path + '/' + str(self.getUID(seriesID))
#         if dicom_2D == True and dicom_3D ==False:
#             onlyfiles = [f for f in glob.glob(str(series_path)+ '/DXm*')]

#         elif dicom_3D == True and dicom_2D ==False:
#             onlyfiles = [f for f in glob.glob(str(series_path)+ '/BT*')]
#         else:
#             onlyfiles = [f for f in glob.glob(str(series_path)+ '/*')]

#         onlyfiles.sort()

#         if imgNum < 0 or imgNum > len(onlyfiles):
#             print("only ", len(onlyfiles), " images in series")
#             return None

#         ds = dicom.dcmread(onlyfiles[imgNum])

#         studyInstanceUID = ds.get("StudyInstanceUID")
#         imageLaterality = ds.get("ImageLaterality")

#         patientID = ds.get("PatientID")
#         value = self.getLabel(studyInstanceUID, imageLaterality, df)

#         if len(value) == 0:
#             label = 'None'
#         else:
#             label = str(value[-1])
#         img = ds.pixel_array

#         if timing is True:
#             print("...took ", time.time()-t0, "seconds")
#         print("dimension:",img.ndim)
#         print("shape:",img.shape)
#         if self.getUID(seriesID) == studyInstanceUID:
#             print("StudyInstanceUID", studyInstanceUID)
#         else:
#             print("not matched")
#         print("PatientID:",patientID)
#         print("ImageLaterality:",imageLaterality)
#         print("label:",label)
#         if timing is True:
#             print("...took ", time.time()-t0, "seconds")
#         return img
