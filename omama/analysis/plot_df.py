import glob
import os

# reading in dicom files
import pydicom as dicom
from os import listdir
from os.path import isfile, join
import pandas as pd
from collections import Counter
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

import time


class Plot:
    df = pd.DataFrame()

    def __init__(self, path=None, folder=None):
        # sets the Data path variable
        self.path = str(path)

        # sets the images acquired from which source/folder value
        self.folder = str(folder)
        t0 = time.time()
        # read the pickle file
        picklefile = open(self.path + self.folder + "Label", "rb")
        # unpickle the dataframe
        self.df = pickle.load(picklefile)
        # close file
        picklefile.close()

        self.df = self.df.rename(columns={"2D/3D": "image"})
        print("For dicoms in folder " + self.folder)
        print(Counter(self.df["label"]))
        print(Counter(self.df["image laterality"]))
        print(Counter(self.df["image"]))

    def index_pre(self):
        df1 = self.df.copy()
        df1["label"] = df1["label"].replace(["IndexCancer", "PreIndexCancer"], "Cancer")
        df1 = df1[df1["label"] != "No Label information"]

        g = sns.FacetGrid(df1, col="image", height=4, aspect=1)
        f = g.map(sns.histplot, "label")
        f.fig.subplots_adjust(top=0.8)
        f.fig.suptitle("Cancer(Index+PreIndex) vs Non Cancer dicoms for " + self.folder)
        return df1

    def nonCancer_pre(self):
        df1 = self.df.copy()

        df1["label"] = df1["label"].replace(
            ["IndexCancer", "PreIndexCancer"], ["Cancer", "NonCancer"]
        )
        df1 = df1[df1["label"] != "No Label information"]

        g = sns.FacetGrid(df1, col="image", height=4, aspect=1)
        f = g.map(sns.histplot, "label")
        f.fig.subplots_adjust(top=0.8)
        f.fig.suptitle(
            "Cancer(Index) vs Non Cancer(+PreIndex) dicoms for " + self.folder
        )

        return df1

    def pltDF(self, executionTime=False):
        t0 = time.time()
        #         #read the pickle file
        #         picklefile = open(self.path + self.folder + 'Label', 'rb')
        #         #unpickle the dataframe
        #         df = pickle.load(picklefile)
        #         #close file
        #         picklefile.close()

        can_plus_pre = self.index_pre()
        notCan_plus_pre = self.nonCancer_pre()
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 2, 1)
        sns.countplot(data=can_plus_pre, x="label", hue="image")
        plt.title("Cancer(Index+Pre) vs Non Cancer dicoms for " + self.folder)
        plt.ylabel("No. of labels")
        plt.subplot(2, 2, 2)
        sns.countplot(data=notCan_plus_pre, x="label", hue="image")
        plt.title("Cancer(Index) vs Non Cancer(+PreIndex) dicoms for " + self.folder)
        plt.ylabel("No. of labels")
        plt.subplot(2, 2, 3)
        sns.countplot(
            data=self.df[self.df["image laterality"] != "None"],
            x="image laterality",
            hue="image",
        )
        plt.title("Left vs Right Breast dicom images for " + self.folder)
        plt.ylabel("No. of dicoms")
        plt.subplot(2, 2, 4)
        sns.countplot(data=self.df, x="image")
        plt.title("2D vs 3D dicom images for " + self.folder)
        plt.ylabel("No. of dicoms")

        plt.subplots_adjust(
            left=0.12, bottom=0.13, right=0.9, top=0.9, wspace=0.25, hspace=0.45
        )
        plt.show()

        if executionTime:
            print("...took ", time.time() - t0, "seconds")
            image2D = self.df["image"] != "3D"
            print("For 2D dicoms in folder " + self.folder)
            print(Counter(self.df[image2D]["label"]))
            print(Counter(self.df[image2D]["image laterality"]))
            print(Counter(self.df[image2D]["image"]))
            print("\n")

            image3D = self.df["image"] != "2D"
            print("For 3D dicoms in folder " + self.folder)
            print(Counter(self.df[image3D]["label"]))
            print(Counter(self.df[image3D]["image laterality"]))
            print(Counter(self.df[image3D]["image"]))

        return can_plus_pre, notCan_plus_pre, self.df
