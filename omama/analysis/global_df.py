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
import hvplot.pandas
import time


class Global_Plot:
    #     df = pd.DataFrame()

    def __init__(self, df, path=None):
        # sets the Data path variable
        self.path = str(path)

        t0 = time.time()

        #         pickles = ['dh_dh2', 'dh_dh0new', 'dh_dcm_ast']

        #         for i in range(len(pickles)):
        #             #read the pickle file
        #             picklefile = open(self.path + pickles[i] + 'Label', 'rb')
        #             #unpickle the dataframe
        #             df_new = pickle.load(picklefile)
        #             self.df = pd.concat([self.df, df_new], ignore_index=True, sort=False)
        #             #close file
        #             picklefile.close()

        #         self.df = df
        print("Total dicoms available")
        print(Counter(df["image"]))
        print(Counter(df["label"]))
        print(Counter(df["Imagelaterality"]))

    def index_pre(self, df):
        df1 = df.copy()
        df1["label"] = df1["label"].replace(["IndexCancer", "PreIndexCancer"], "Cancer")
        df1 = df1[df1["label"] != "No Label information"]
        df1 = df1[df1["Imagelaterality"] != "B"]

        #         df_plot = df1.groupby(['label', 'image', 'Imagelaterality']).size().reset_index().pivot(columns=['label'], index=['Imagelaterality', 'image'], values=0)

        #         df_plot.plot(kind='bar', stacked=True, cmap = 'Paired')
        #         plt.title("Cancer(Index+PreIndex) vs Non Cancer dicoms for global")
        #         plt.ylabel('No. of labels')
        #         plt.show()

        return df1

    def nonCancer_pre(self, df):
        df1 = df.copy()

        df1["label"] = df1["label"].replace(
            ["IndexCancer", "PreIndexCancer"], ["Cancer", "NonCancer"]
        )
        df1 = df1[df1["label"] != "No Label information"]
        df1 = df1[df1["Imagelaterality"] != "B"]

        #         df_plot = df1.groupby(['label', 'image', 'Imagelaterality']).size().reset_index().pivot(columns=['label'], index=['Imagelaterality', 'image'], values=0)

        #         df_plot.plot(kind='bar', stacked=True, cmap = 'Paired')
        #         plt.title("Cancer(Index) vs Non Cancer(+PreIndex) dicoms for global")
        #         plt.ylabel('No. of labels')
        #         plt.show()

        return df1

    def create(self, df, xLabel, category, plotLabel, value, plot):
        if plot == "xaxis":
            plot = df[df[str(plotLabel)] == value].hvplot.bar(
                x=str(xLabel),
                y="count",
                by=str(category),
                stacked=True,
                label=str(plotLabel) + ": " + value,
                legend="top_right",
                yformatter="%.0f",
                ylim=(0, 900000),
            )
        else:
            plot = df[df[str(plotLabel)] == value].hvplot.barh(
                x=str(xLabel),
                y="count",
                by=str(category),
                stacked=True,
                label=str(plotLabel) + ": " + value,
                legend="top_right",
                xformatter="%.0f",
            )

        return plot

    def labelPlot(self, df, PreIndex="None"):
        df1 = df.copy()
        plotTitle = "None"
        if PreIndex == "NonCancer":
            df1["label"] = df1["label"].replace(
                ["IndexCancer", "PreIndexCancer"], ["Cancer", "NonCancer"]
            )
            plotTitle = "Cancer(Index) vs Non Cancer(+PreIndex) dicoms for global"
        elif PreIndex == "Cancer":
            df1["label"] = df1["label"].replace(
                ["IndexCancer", "PreIndexCancer"], "Cancer"
            )
            plotTitle = "Cancer(Index+PreIndex) vs Non Cancer dicoms for global"
        else:
            print("Error, please provide what you wanna see PreIndex as....")

        df1 = df1[df1["label"] != "No Label information"]
        df1 = df1[df1["Imagelaterality"] != "B"]

        #         df1 = df1.groupby(['label', 'image','Imagelaterality']).size().reset_index(name = 'count')

        #         plotA = self.create(df1, '2D')
        #         plotB = self.create(df1, '3D').opts(show_legend=False, yaxis=None)

        #         # add separate plots together again
        #         return (plotA + plotB).opts(title = plotTitle)
        return df1, plotTitle

    def manufacturer(self, df, PreIndex="None"):
        df1 = df.copy()

        if PreIndex == "Cancer":
            df1 = self.index_pre(df1)

        elif PreIndex == "NonCancer":
            df1 = self.nonCancer_pre(df1)

        else:
            print("Error, please provide what you wanna see PreIndex as....")

        # replacing case sensitive and same manufarcturer name values with one value
        df1["Manufacturer"] = df1["Manufacturer"].replace(
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
        df1["Manufacturer"] = df1["Manufacturer"].replace(
            ["FUJI PHOTO FILM Co., ltd.", "FUJIFILM Corporation"],
            "FUJIFILM Corporation",
        )

        #         #multi indexing
        #         df1 = df1.groupby(['Manufacturer', 'image']).size().reset_index(name = 'count')
        #         plot = df1.hvplot.barh( x='Manufacturer', y='count', by='image', cmap ='Paired', stacked=True, legend='top_right', height = 500, width = 800,)
        # #         df1.plot(kind='barh', stacked=True, cmap = 'Paired_r')
        # #         plt.title("Manufaractures and modality informations for 2d and 3d dicoms")
        # #         plt.xlabel('Count')
        #         return plot.opts(title = "Manufaractures informations for 2d and 3d dicoms")
        plotTitle = "Manufacturer distribution"
        return df1, plotTitle

    def imgShape(self, df, PreIndex="None"):
        # since no of frames can be different so replacing all the similaer pattern 3D shape whith 'x'
        df1 = df.copy()

        if PreIndex == "Cancer":
            df1 = self.index_pre(df1)

        elif PreIndex == "NonCancer":
            df1 = self.nonCancer_pre(df1)

        else:
            print("Error, please provide what you wanna see PreIndex as....")

        df1["Shape"] = df1["Shape"].astype(str)
        df1["Shape"] = df1["Shape"].str.replace("(2457, 1890,.*)", "2457, 1890, x)")
        df1["Shape"] = df1["Shape"].str.replace("(2457, 1996,.*)", "2457, 1996, x)")

        # remove shape less than (1024, 1024)
        pattern = "(2059, 652,.*)|(2023, 918,.*)|(512, 512)|(707, 989)|(794, 990)"
        df1 = df1[~df1["Shape"].str.contains(pattern)]
        # with 2.2% images filtered out

        #         df1 = df1.groupby(['Manufacturer', 'Shape']).size().reset_index(name = 'count')
        # not select if count < 50
        counts = df1["Shape"].value_counts()
        df1 = df1[~df1["Shape"].isin(counts[counts < 50].index)]

        df1 = df1.sort_values(by="Shape")
        #         # multi indexing
        #         df1 = df1.pivot(columns=['image'], index=['Shape'], values='count')

        #         df1.plot(kind='barh', stacked=True, cmap = 'Paired_r')
        #         plt.title("img_shape for 2d and 3d dicoms")
        #         plt.xlabel('Count')
        #         plt.show()

        #         plot = df1.hvplot.barh( x='Shape', y='count', by='Manufacturer', stacked=True, legend='top_right', height = 500, width = 800,)
        #         return plot.opts(title = "img_shape for 2d and 3d dicoms")
        plotTitle = "Image Shape distribution"
        return df1, plotTitle

    def age(self, df, PreIndex="None"):
        df1 = df.copy()

        if PreIndex == "Cancer":
            df1 = self.index_pre(df1)

        elif PreIndex == "NonCancer":
            df1 = self.nonCancer_pre(df1)

        else:
            print("Error, please provide what you wanna see PreIndex as....")

        # extracting numerical value from the string
        df1["PatientAge"] = df1["PatientAge"].str.extract("(\d+)", expand=False)

        # converting month into year
        df1["PatientAge"] = df1["PatientAge"].apply(lambda x: float(x) // 12)
        df1 = df1[~df1.PatientAge.isna()]

        # dividing pateint age into groups
        df1["PatientAgeGroup"] = pd.cut(
            x=df1["PatientAge"], bins=list(i for i in range(0, 100, 10)), right=False
        )

        df1 = df1.sort_values(by="PatientAgeGroup")
        df1["PatientAgeGroup"] = df1["PatientAgeGroup"].astype(str)
        #         df1 = df1.groupby(['label', 'PatientAgeGroup']).size().reset_index(name = 'count')

        #         plot = df1.hvplot.bar( x='PatientAgeGroup', y='count', by='label', stacked=True, legend='top_right', height = 500, width = 800,)
        plotTitle = "Patient Age distribution"
        #         return plot.opts(title = "Patient Age distribution over Cancer and NonCancer labels")
        return df1, plotTitle

    def descriptivePlot(
        self,
        df,
        PreIndex="None",
        xLabel="Imagelaterality",
        category="label",
        plotLabel="image",
        plot="xaxis",
    ):
        df1 = df.copy()
        plotTitle = "None"

        if xLabel == "Imagelaterality" and category == "label":
            df1, plotTitle = self.labelPlot(df1, PreIndex)
        if xLabel == "PatientAgeGroup":
            df1, plotTitle = self.age(df1, PreIndex)
        if xLabel == "Shape":
            df1, plotTitle = self.imgShape(df1, PreIndex)
        if (
            xLabel == "Manufacturer"
            or category == "Manufacturer"
            or plotLabel == "Manufacturer"
        ):
            df1, plotTitle = self.manufacturer(df1, PreIndex)

        df1 = (
            df1.groupby([str(xLabel), str(category), str(plotLabel)])
            .size()
            .reset_index(name="count")
        )
        df1 = df1.sort_values(by=category)

        if plotLabel == "image":
            plotA = self.create(df1, xLabel, category, "image", "2D", plot).opts(
                cmap="Set2"
            )
            plotB = self.create(df1, xLabel, category, "image", "3D", plot).opts(
                show_legend=False, cmap="Set2", yaxis=None
            )

            # add separate plots together again
            return df1, (plotA + plotB).opts(title=plotTitle)
        else:
            plotA = self.create(
                df1, xLabel, category, "Manufacturer", "HOLOGIC, INC.", "yaxis"
            ).opts(cmap="Set2")
            plotB = self.create(
                df1, xLabel, category, "Manufacturer", "GE MEDICAL SYSTEMS", "yaxis"
            ).opts(show_legend=False, cmap="Set2", yaxis=None)

            # add separate plots together again
            return df1, (plotA + plotB).opts(title=plotTitle)

    def testDataset(self, df):
        # pablo style start here
        #         df = df[df['label'] != 'No Label information']
        #         df = df[df['Imagelaterality'] != 'B']
        #         print(Counter(df['Manufacturer']))

        #         df,_ = self.manufacturer(df, PreIndex='Cancer')

        #         df = df[df['Manufacturer'].isin(['HOLOGIC, INC.', 'GE MEDICAL SYSTEMS'])]
        #         print(Counter(df['Shape']))

        #         df,_ = self.imgShape(df, PreIndex='Cancer')
        #         counts = df['Shape'].value_counts()
        #         df = df[~df['Shape'].isin(counts[counts < 500].index)]

        #         dfCancer = dfnonCancer = df
        # pablo style ends here

        df = df[df["label"] != "No Label information"]
        df = df[df["Imagelaterality"] != "B"]
        dfCancer = dfnonCancer = df
        dfCancer = dfCancer[dfCancer["label"] != "NonCancer"]

        value1 = dfCancer[dfCancer["image"] == "3D"].drop_duplicates(
            subset=["StudyInstanceUID"]
        )
        value2 = dfCancer[dfCancer["image"] == "2D"].drop_duplicates(
            subset=["StudyInstanceUID"]
        )

        dfCancer = (
            pd.concat([value1, value2])
            .sort_values("image")
            .drop_duplicates(
                subset=["StudyInstanceUID", "StudyInstanceUID"], keep="last"
            )
        )

        dfnonCancer = dfnonCancer[dfnonCancer["label"] == "NonCancer"]

        value1 = dfnonCancer[dfnonCancer["image"] == "3D"].drop_duplicates(
            subset=["StudyInstanceUID"]
        )
        value2 = dfnonCancer[dfnonCancer["image"] == "2D"].drop_duplicates(
            subset=["StudyInstanceUID"]
        )

        dfnonCancer = (
            pd.concat([value1, value2])
            .sort_values("image")
            .drop_duplicates(
                subset=["StudyInstanceUID", "StudyInstanceUID"], keep="last"
            )
        )

        test_df = (
            pd.concat([dfnonCancer, dfCancer])
            .sort_values("image")
            .drop_duplicates(
                subset=["StudyInstanceUID", "StudyInstanceUID"], keep="last"
            )
        )

        test_df, _ = self.manufacturer(test_df, PreIndex="Cancer")

        test_df = test_df[
            test_df["Manufacturer"].isin(["HOLOGIC, INC.", "GE MEDICAL SYSTEMS"])
        ]

        test_df, _ = self.imgShape(test_df, PreIndex="Cancer")
        counts = test_df["Shape"].value_counts()
        test_df = test_df[~test_df["Shape"].isin(counts[counts < 500].index)]

        return test_df


#     def pltDF(self, executionTime=False):

#         t0 = time.time()

#         can_plus_pre = self.index_pre()
#         notCan_plus_pre = self.nonCancer_pre()


#         plt.figure(figsize=(12, 6))
#         plt.subplot(2,2, 1)
#         sns.countplot(data=can_plus_pre, x='label', hue='image')
#         plt.title("Cancer(Index+Pre) vs Non Cancer dicoms for global")
#         plt.ylabel('No. of labels')
#         plt.subplot(2, 2, 2)
#         sns.countplot(data=notCan_plus_pre, x='label', hue='image')
#         plt.title("Cancer(Index) vs Non Cancer(+PreIndex) dicoms for global")
#         plt.ylabel('No. of labels')
#         plt.subplot(2, 2, 3)
#         sns.countplot(data=self.df[self.df['Imagelaterality'] != 'None'], x='Imagelaterality', hue='image')
#         plt.title("Left vs Right Breast dicom images for global")
#         plt.ylabel('No. of dicoms')
#         plt.subplot(2, 2, 4)
#         sns.countplot(data=self.df, x='image')
#         plt.title("2D vs 3D dicom images for global")
#         plt.ylabel('No. of dicoms')

#         plt.subplots_adjust(left=0.12,
#                     bottom=0.13,
#                     right=0.9,
#                     top=0.9,
#                     wspace=0.25,
#                     hspace=0.45)
#         plt.show()

#         if executionTime:
#             print("...took ", time.time()-t0, "seconds")
#             image2D= self.df['image'] != '3D'
#             print("For 2D dicoms in global")
#             print(Counter(self.df[image2D]['label']))
#             print(Counter(self.df[image2D]['Imagelaterality']))
#             print(Counter(self.df[image2D]['image']))
#             print("\n")

#             image3D= self.df['image'] != '2D'
#             print("For 3D dicoms in global")
#             print(Counter(self.df[image3D]['label']))
#             print(Counter(self.df[image3D]['Imagelaterality']))
#             print(Counter(self.df[image3D]['image']))

#         return self.df
