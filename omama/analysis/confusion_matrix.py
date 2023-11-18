# class to build confusion matrix from deep_sight results
import json
import time
from pathlib import Path

import omama as O
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

__author__ = "Ryan Zurrin"

CACHE_PATH = r"/raid/mpsych/OMAMA/DATA/cache_files/predictions_cache.json"


class ConfusionMatrix:
    """Class to build confusion matrix from deep_sight results.
    def build(self):
        builds the confusion matrix from the data that was generated
    """

    def __init__(
        self,
        config_num=2,
        caselist_path=None,
        threshold=0.5,
        sample_size=None,
        timing=False,
    ):
        t0 = time.time()
        print("Initializing ConfusionMatrix")
        omama_loader = O.OmamaLoader(config_num=config_num)
        self._data = O.DataHelper.check_data_instance(config_num=config_num)
        if sample_size is None:
            self._sample_size = len(self._data)
        else:
            self._sample_size = sample_size

        if caselist_path is None:
            self._caselist_path = omama_loader.caselist_path
        else:
            self._caselist_path = caselist_path

        self._pred_dict = json.loads(open(CACHE_PATH, "r").read())
        self._threshold = threshold
        self._actual = np.empty(self._sample_size)
        self._predicted = np.empty(self._sample_size)

        self._generate_data(self._threshold, self._sample_size, timing=timing)

        if timing:
            print(f"Time to initialize ConfusionMatrix: {time.time() - t0}")

    def _get_prediction_from_cache(self, sop_uid):
        """Gets the prediction from the cache file

        Parameters
        ----------
        sop_uid : str
            The SOPInstanceUID of the image that the prediction is for.

        Returns
        -------
        float
            The prediction for the image.
        """
        # check if the sop_uid is in the cache and if it is return the score
        if sop_uid in self._pred_dict:
            return self._pred_dict[sop_uid]["score"]
        else:
            return None

    def _generate_data(self, threshold, sample_size, timing=False):
        """Generates the data that will be used in the confusion matrix

        Parameters
        ----------
        threshold : float
            The threshold that will be used to determine if a sample is
            considered correct or incorrect.
        sample_size : int
            The number of samples that will be used to build the confusion
            matrix.
        timing : bool
            If True, the time to generate the data will be printed.
        """
        t0 = time.time()
        path = Path(self._caselist_path)
        whitelist = []
        SOPInstanceUIDs = []
        with open(path, "r") as f:
            for line in f:
                whitelist.append(line.strip())
                sopuid = line.strip().split("/")[-1]
                # remove the prefix that starts with BT. or DXm.
                if sopuid.startswith("BT."):
                    sopuid = sopuid[3:]
                elif sopuid.startswith("DXm."):
                    sopuid = sopuid[4:]
                SOPInstanceUIDs.append(sopuid)

        # iterate over the whitelist and get the actual and predicted values
        # and test if they are correct or incorrect
        for i, case in enumerate(whitelist):
            if i >= sample_size:
                break
            # get the actual value from the SOPInstanceUID
            label = self._data.get_label(SOPInstanceUIDs[i])
            # print(f'Actual: {label}')
            if label == "IndexCancer" or label == "PreIndexCancer":
                self._actual[i] = 1
            else:
                self._actual[i] = 0
            # get the predicted value from deep_sight
            pred = self._get_prediction_from_cache(SOPInstanceUIDs[i])
            if pred is None:
                print(f"No prediction found for {SOPInstanceUIDs[i]}")
                print("running deep_sight on image now please wait")
                pred = O.DeepSight.run(case)
                pred = pred[SOPInstanceUIDs[i]]["score"]
            if pred is None:
                print(f"No prediction found for {SOPInstanceUIDs[i]}")
                self._predicted[i] = 0
            else:
                if pred >= threshold and self._actual[i] == 1:
                    self._predicted[i] = 1
                elif pred < threshold and self._actual[i] == 0:
                    self._predicted[i] = 0
                elif pred >= threshold and self._actual[i] == 0:
                    self._predicted[i] = 1
                elif pred < threshold and self._actual[i] == 1:
                    self._predicted[i] = 0

        if timing:
            print(f"Time to generate data: {time.time() - t0}")

    def build(self):
        confusion_matrix = metrics.confusion_matrix(self._actual, self._predicted)
        cm_display = metrics.ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix, display_labels=["NonCancer", "Cancer"]
        )
        cm_display.plot()
        plt.show()

        # Calculate metrics
        accuracy = accuracy_score(self._actual, self._predicted)
        precision = precision_score(self._actual, self._predicted)
        recall = recall_score(self._actual, self._predicted)
        f1 = f1_score(self._actual, self._predicted)
        tn, fp, fn, tp = confusion_matrix.ravel()
        specificity = tn / (tn + fp)

        # Print metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Specificity: {specificity:.4f}")
