import traceback
from datetime import datetime
import hashlib
import json
import logging
import os
import statistics
import time
import contextlib
from types import SimpleNamespace

import numpy as np

import omama as O
from omama.algorithms import Algorithms

DEBUG = True

ALGORITHMS = [
    "AE",
    "AvgKNN",
    "VAE",
    "SOGAAL",
    "DeepSVDD",
    "AnoGAN",
    "HBOS",
    "LOF",
    "OCSVM",
    "IForest",
    "CBLOF",
    "COPOD",
    "SOS",
    "KDE",
    "Sampling",
    "PCA",
    "LMDD",
    "COF",
    "ECOD",
    "KNN",
    "MedKNN",
    "SOD",
    "INNE",
    "FB",
    "LODA",
    "SUOD",
]

DATA_SETS = ["8%", "13%", "24%"]
# DATA_SETS = ['24%', '24%', '24%']

BAD_IMAGE_ID_PATH = r"/raid/mpsych/cache_files/bad_image_sopuids.txt"
# BAD_IMAGE_ID_PATH = r'/raid/mpsych/cache_files/good_image_sopuids.txt'
# BAD_IMAGE_ID_PATH = r'/raid/mpsych/cache_files/bad_image_sopuids_AB_star.txt'

CACHE_PATH = r"/raid/mpsych/cache_files/"

# CACHE_FILE = r'outlier_detector_cache.json'
# STATS_CACHE = r'outlier_stats_cache_11_19_22.json'

# CACHE_FILE = r'MIDL_OD_CACHE_NEW.json'
# STATS_CACHE = r'MIDL_OD_STATS_CACHE_NEW.json'

# CACHE_FILE = r'MIDL_OD_CACHE_D.json'
# STATS_CACHE = r'MIDL_OD_STATS_CACHE_D.json'

CACHE_FILE = r"ODL_CACHE_RERUN_MEGARUN.json"
STATS_CACHE = r"ODL_OD_STATS_CACHE_RERUN_MEGARUN.json"

LOG_DIR = r"/raid/mpsych/cache_files/RERUN_MEGARUN/"


class OutlierDetector(Algorithms):
    def __init__(
        self,
        run_id,
        algorithms=None,
        imgs=None,
        features=None,
        bad_ids=None,
        number_bad=None,
        exclude=None,
        timing=False,
        **kwargs,
    ):
        """Initializes the OutlierDetector class"""
        t0 = time.time()
        self.__run_id = run_id
        if algorithms is None:
            self.__algorithms = ALGORITHMS
        else:
            self.__algorithms = algorithms
        if bad_ids is not None:
            self.__bad_ids = bad_ids
        else:
            self.__bad_ids = BAD_IMAGE_ID_PATH
        if number_bad is not None:
            self.__number_bad = number_bad
        else:
            self.__number_bad = None
        if exclude is not None:
            self.__exclude = exclude
        else:
            self.__exclude = []
        if features is not None:
            self.__features = features
        else:
            self.__features = None
        verbose = kwargs.get("verbose", False)
        if verbose is False:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            logging.getLogger("tensorflow").disabled = True
            logging.getLogger("pyod").disabled = True

        if imgs is not None:
            self.__accuracy_scores, self.__errors = self.run_all_algorithms(
                imgs,
                self.__features,
                self.__bad_ids,
                self.__number_bad,
                timing,
                **kwargs,
            )

        else:
            self.__accuracy_scores = None
            self.__errors = None

        if timing:
            timing(t0, "OD __init__")

    def run_all_algorithms(
        self, imgs, features, bad_ids, number_bad, timing=False, **kwargs
    ):
        """Automates the running of all the pyod algorithms and builds a
        dictionary of each algorithm to its accuracy
        Parameters
        ----------
        imgs : list
            The images to be analyzed
        features : list
            The features to be used in the analysis
        bad_ids : str, list
            The path to a text file with a list of bad ids or a list of bad ids
        number_bad : int
            The number of bad images
        verbose : bool, optional (default=False)
            Whether to print the results of each algorithm
        timing : bool, optional (default=False)
            Whether to time the results
        **kwargs : dict
            Keyword arguments to pass to the algorithms
        Returns
        -------
        accuracy_scores, errors : dict
            A dictionary of each algorithm to its accuracy, and a dictionary of
            each algorithm to its error
        """
        # create a 8bit hash from the kwargs to use as part of the cache key
        kwargs_hash = dict_to_hash(kwargs)

        # put the feature type name with the number bad to use in the cache file
        cache_key = f"{self.__run_id}_{number_bad}%_{kwargs_hash}"

        accuracy_scores = {}
        errors = {}
        t0 = time.time()
        verbose = kwargs.get("verbose", False)
        for alg in self.__algorithms:
            accuracy = None
            if alg not in self.__exclude:
                try:
                    if verbose is False:
                        print(f"Running in if, {alg}...", end=" ")
                        out_file = "outlier_output.txt"
                        with open(out_file, "w") as f:
                            with contextlib.redirect_stdout(f):
                                t_scores, t_labels = self._detect_outliers(
                                    features,
                                    pyod_algorithm=alg,
                                    verbose=verbose,
                                    **kwargs,
                                )
                    else:
                        print(f"Running in else, {alg}...")
                        t_scores, t_labels = self._detect_outliers(
                            features, pyod_algorithm=alg, verbose=verbose, **kwargs
                        )

                # if the algorithm causes an error, skip it and move on
                except Exception as e:
                    print(f"Error: {e}")
                    if verbose:
                        print(f"Error with {alg}: {e}")
                    accuracy_scores[alg] = -1
                    errors[alg] = e
                    continue
                if number_bad is not None:
                    accuracy = self.accuracy(imgs, [t_scores], bad_ids, number_bad)
                    accuracy_scores[alg] = accuracy
                # save the accuracy scores to the cache file
                filename = f"{cache_key}_{alg}"

                OutlierDetector._save_outlier_path_log(filename, imgs, t_scores)
            if verbose:
                print(f"{alg} accuracy: {accuracy}")

        accuracy_scores = {
            k: v
            for k, v in sorted(
                accuracy_scores.items(), key=lambda item: item[1], reverse=True
            )
        }

        OutlierDetector.save_to_cache(cache_key, accuracy_scores, kwargs, errors)

        if timing:
            display_timing(t0, "run_all_algorithms")

        return accuracy_scores, errors

    @property
    def results(self):
        return self.__accuracy_scores, self.__errors

    @property
    def algorithms(self):
        return self.__algorithms

    @staticmethod
    def get_from_cache(cache_key):
        """loads the cache file as a json file, if the file does not exist
        it will return None"""
        # if the cache file exists, load it
        if os.path.exists(CACHE_PATH + CACHE_FILE):
            with open(CACHE_PATH + CACHE_FILE, "r") as f:
                cache = json.loads(f.read())
            # if the cache key exists, return the results
            if cache_key in cache:
                return cache[cache_key]
            # if the cache key does not exist, return None
            else:
                return None
        # if the cache file does not exist, return None
        else:
            return None

    @staticmethod
    def save_to_cache(cache_key, results, configurations=None, errors=None):
        """saves the results to the cache file"""
        # if the cache file exists, load it
        if os.path.exists(CACHE_PATH + CACHE_FILE):
            with open(CACHE_PATH + CACHE_FILE, "r") as f:
                cache = json.loads(f.read())
        # if the cache file does not exist, create it
        else:
            cache = {}
        # add the results to the cache
        cache[cache_key] = results
        cache[cache_key + "_config"] = configurations
        # cache[cache_key + "_errors"] = errors
        # sort the cache alphabetically by key
        cache = {k: v for k, v in sorted(cache.items(), key=lambda item: item[0])}
        # save the cache
        with open(CACHE_PATH + CACHE_FILE, "w") as f:
            json.dump(cache, f)

    @staticmethod
    def extract_feature_name(cache_key):
        """extracts the feature from the name of the cache key
        if 'hist' found within key feature = "Histogram"
        if 'downsample' found within key feature = "Downsample"
        if 'sift' found within key feature = "SIFT"
        if 'orb' found within key feature = "ORB"
        """
        if "hist" in cache_key:
            return "Histogram"
        if "downsample" in cache_key:
            return "Downsample"
        if "sift" in cache_key:
            return "SIFT"
        if "orb" in cache_key:
            return "ORB"

    @staticmethod
    def extract_dataset(cache_key):
        """extracts the dataset from the name of the cache key
        key looks like: 'hist_24%_061ffe01df04d1c7d1dfe4f764e24017'
        the dataset is the two number before the % sign.
        """
        return cache_key.split("_")[1]

    @staticmethod
    def extract_dataset_name(cache_key):
        """extracts the dataset name from the name of the cache key
        key looks like: MedKNN_gaussian_hist_24%-errors_31aeb7
        the dataset is the two number before the % sign.
        """
        return cache_key.split("_")[3]

    @staticmethod
    def extract_norm_type(cache_key):
        """extracts the normalization from the name of the cache key
        key looks like: MedKNN_gaussian_hist_24%-errors_31aeb7
        the norm type is the after the second _ sign.
        """
        norm = cache_key.split("_")[1]
        if norm == "minmax":
            return "Min-Max"
        if norm == "gaussian":
            return "Gaussian"
        if norm == "standard":
            return "Standard"
        if norm == "max":
            return "Max"

    @staticmethod
    def show_records(hash=None):
        """prints the records in the cache file"""
        # if the cache file exists, load it
        if hash is None:
            if os.path.exists(CACHE_PATH + CACHE_FILE):
                with open(CACHE_PATH + CACHE_FILE, "r") as f:
                    cache = json.load(f)
                # print the cache in a pretty format
                print(json.dumps(cache, indent=4))
                # get the length of records and print it
                print(f"Number of records: {len(cache)}")
            # if the cache file does not exist, return None
            else:
                print("No cache file exists")
        else:
            # find all the records that contain the hash string within the key
            records = []
            if os.path.exists(CACHE_PATH + CACHE_FILE):
                with open(CACHE_PATH + CACHE_FILE, "r") as f:
                    cache = json.loads(f.read())
                for key in cache:
                    if hash in key:
                        records.append(key)
                # print the records in a pretty format
                print(json.dumps(records, indent=4))
                # get the length of records and print it
                print(f"Number of records: {len(records)}")
            # if the cache file does not exist, return None
            else:
                print("No cache file exists")

    @staticmethod
    def show_stats_records(hash=None):
        """prints the records in the cache file"""
        # if the cache file exists, load it
        if hash is None:
            if os.path.exists(CACHE_PATH + STATS_CACHE):
                with open(CACHE_PATH + STATS_CACHE, "r") as f:
                    cache = json.loads(f.read())
                # print the cache in a pretty format using json dumps
                print(json.dumps(cache, indent=4))
        else:
            # find all the records that contain the hash string within the key
            records = []
            if os.path.exists(CACHE_PATH + STATS_CACHE):
                with open(CACHE_PATH + STATS_CACHE, "r") as f:
                    cache = json.loads(f.read())
                for key in cache:
                    if hash in key:
                        records.append(key)
                # print the records in a pretty format using json dumps
                print(json.dumps(records, indent=4))
            # if the cache file does not exist, return None
            else:
                print("No cache file exists")

    @staticmethod
    def show_best(N=10):
        """prints the best results in the cache file"""
        # if the cache file exists, load it
        best_list = []
        if os.path.exists(CACHE_PATH + CACHE_FILE):
            with open(CACHE_PATH + CACHE_FILE, "r") as f:
                cache = json.loads(f.read())
            # sort the cache by the accuracy score
            for k, v in cache.items():
                # bring the key and the value of the first index of the
                # nexted value only if the key does not end with _config

                if not k.endswith("_config"):
                    # print(f'key:{k}: alg:{list(v.items())[0][0]} acc
                    # :{list(v.items())[0][1]}')
                    best_list.append((k, list(v.items())[0][0], list(v.items())[0][1]))
                # sort the list by the accuracy score

            best_list = sorted(best_list, key=lambda x: x[2], reverse=True)

            # print the top number of results using the best list but print
            # it in a pretty format even thought it is a list
            print(json.dumps(best_list[:N], indent=4))
        else:
            print("No cache file exists")

    @staticmethod
    def show_best_stats(N=10):
        """prints the best results in the cache file"""
        # if the cache file exists, load it
        best_list = []
        if os.path.exists(CACHE_PATH + STATS_CACHE):
            with open(CACHE_PATH + STATS_CACHE, "r") as f:
                cache = json.loads(f.read())
            # stats cache is already sorted by accuracy score so just put the N best
            # results in a list and print it
            for k, v in cache.items():
                best_list.append((k, v))
            print(json.dumps(best_list[:N], indent=4))
        else:
            print("No cache file exists")

    @staticmethod
    def get_best_algorithm_runs(
        algorithm, feature=None, dataset=None, N=10, display=True, include_hash=False
    ):
        """returns the best run for the given algorithm
        Parameters
        ----------
        algorithm : str
            the name of the algorithm to get the best run for
        feature : str
            the name of the feature to get the best run for
        dataset : str
            the name of the dataset to get the best run for
        N : int
            the number of best runs to return
        display : bool
            if True, the results will be printed
        include_hash : bool
            if True, the hash of the run will be included in the results
        Returns
        -------
        str
            list of the best runs for the given algorithm, feature, and dataset
            in a format that can easily be used to make latex tables
        """
        best_list = []
        if os.path.exists(CACHE_PATH + CACHE_FILE):
            with open(CACHE_PATH + CACHE_FILE, "r") as f:
                cache = json.loads(f.read())

            for k, v in cache.items():
                if not k.endswith("_config"):
                    # if feature is not None and dataset is not None:
                    if feature is not None and dataset is not None:
                        if feature in k and str(dataset) in k:
                            for i in v:
                                if i == algorithm:
                                    print(f"{k}: {v[i]}")
                                    best_list.append((k, i, v[i]))
                    elif feature is not None:
                        if feature in k:
                            for i in v:
                                if i == algorithm:
                                    best_list.append((k, i, v[i]))
                    elif dataset is not None:
                        if str(dataset) in k:
                            for i in v:
                                if i == algorithm:
                                    best_list.append((k, i, v[i]))
                    else:
                        for i in v:
                            if i == algorithm:
                                best_list.append((k, i, v[i]))

        else:
            print("No cache file exists")
            # sort the list by the accuracy score

        best_list = sorted(best_list, key=lambda x: x[2], reverse=True)
        # parse the best results in order to use in latex table in the
        # format: Algorithm & Feature & Accuracy & Dataset & cachekey \\
        # use best list to get the results
        latex_string = ""
        for i in best_list[:N]:
            feature = OutlierDetector.extract_feature_name(i[0])
            # if the feature is not None, then print the feature
            if feature is not None:
                if display:
                    if include_hash:
                        print(f"{i[1]} & {feature} & {i[2]} & {hash} \\\\")
                    else:
                        print(f"{i[1]} & {feature} & {i[2]:.4f} \\\\")

                if include_hash:
                    latex_string += (
                        f"{algorithm} & "
                        f"{feature} & "
                        # print the accuracy out to 4 decimal places
                        f"{i[2]:.4f} & "
                        f"{i[0]} "
                        f"\\\\" + "\n"
                    )
                else:
                    latex_string += (
                        f"{algorithm} & "
                        f"{feature} & "
                        # print the accuracy out to 4 decimal places
                        f"{i[2]:.4f} "
                        f"\\\\" + "\n"
                    )
        return latex_string

    @staticmethod
    def get_best_algorithm_runs_stats(
        algorithm,
        feature=None,
        dataset=None,
        N=10,
        display=False,
        include_dataset=False,
        include_hash=False,
    ):
        """returns the best run for the given algorithm
        Parameters
        ----------
        algorithm : str
            the name of the algorithm to get the best run for
        feature : str
            the name of the feature to get the best run for
        dataset : str
            the name of the dataset to get the best run for
        N : int
            the number of best runs to return
        display : bool
            if True, the results will be printed
        include_hash : bool
            if True, the hash of the run will be included in the results
        Returns
        -------
        str
            list of the best runs for the given algorithm, feature, and dataset
            in a format that can easily be used to make latex tables
        """
        best_list = []
        if os.path.exists(CACHE_PATH + STATS_CACHE):
            with open(CACHE_PATH + STATS_CACHE, "r") as f:
                cache = json.loads(f.read())

            for k, v in cache.items():
                dataset_name = OutlierDetector.extract_dataset_name(k)
                # v['pyod_algorithm'] is the algorithm name to search for
                # if feature is not None and dataset is not None:
                if feature is not None and dataset is not None:
                    if feature in k and str(dataset) in k:
                        if v["pyod_algorithm"] == algorithm:
                            if include_dataset:
                                best_list.append(
                                    (
                                        k,
                                        v["pyod_algorithm"],
                                        v["avg_accuracy"],
                                        v["standard_deviation"],
                                        dataset_name,
                                    )
                                )
                            else:
                                best_list.append(
                                    (
                                        k,
                                        v["pyod_algorithm"],
                                        v["avg_accuracy"],
                                        v["standard_deviation"],
                                    )
                                )
                elif feature is not None:
                    if feature in k:
                        if v["pyod_algorithm"] == algorithm:
                            if include_dataset:
                                best_list.append(
                                    (
                                        k,
                                        v["pyod_algorithm"],
                                        v["avg_accuracy"],
                                        v["standard_deviation"],
                                        dataset_name,
                                    )
                                )
                            else:
                                best_list.append(
                                    (
                                        k,
                                        v["pyod_algorithm"],
                                        v["avg_accuracy"],
                                        v["standard_deviation"],
                                    )
                                )
                elif dataset is not None:
                    if str(dataset) in k:
                        if v["pyod_algorithm"] == algorithm:
                            if include_dataset:
                                best_list.append(
                                    (
                                        k,
                                        v["pyod_algorithm"],
                                        v["avg_accuracy"],
                                        v["standard_deviation"],
                                        dataset_name,
                                    )
                                )
                            else:
                                best_list.append(
                                    (
                                        k,
                                        v["pyod_algorithm"],
                                        v["avg_accuracy"],
                                        v["standard_deviation"],
                                    )
                                )
                else:
                    if v["pyod_algorithm"] == algorithm:
                        if include_dataset:
                            best_list.append(
                                (
                                    k,
                                    v["pyod_algorithm"],
                                    v["avg_accuracy"],
                                    v["standard_deviation"],
                                    dataset_name,
                                )
                            )
                        else:
                            best_list.append(
                                (
                                    k,
                                    v["pyod_algorithm"],
                                    v["avg_accuracy"],
                                    v["standard_deviation"],
                                )
                            )

        else:
            print("No cache file exists")
            # sort the list by the accuracy score
        best_list = sorted(best_list, key=lambda x: x[2], reverse=True)
        # parse the best results in order to use in latex table in the
        # format: Algorithm & Feature & Average Accuracy & SD & Dataset & cachekey \\
        # use best list to get the results
        latex_string = ""
        for i in best_list[:N]:
            feature = OutlierDetector.extract_feature_name(i[0])
            norm_type = OutlierDetector.extract_norm_type(i[0])
            # if the feature is not None, then print the feature
            if feature is not None:
                if display:
                    if include_hash:
                        print(
                            f"{i[1]} & {norm_type} + {feature} & $ {i[2]:.4f} \\pm{i[3]:.4f}$ & {i[4]} & {i[0]} \\\\"
                        )
                    else:
                        print(
                            f"{i[1]} & {norm_type} + {feature} & $ {i[2]:.4f} \\pm{i[3]:.4f}$ & {i[4]} \\\\"
                        )

                if include_hash:
                    if include_dataset:
                        latex_string += (
                            f"{algorithm} & "
                            f"{norm_type} + {feature} & "
                            # print the accuracy out to 4 decimal places
                            f"$ {i[2]:.4f} \\pm{i[3]:.4f}$ & "
                            f"{i[4]} & "
                            f"{i[0]} "
                            f"\\\\" + "\n"
                        )
                    else:
                        latex_string += (
                            f"{algorithm} & "
                            f"{norm_type} + {feature} & "
                            # print the accuracy out to 4 decimal places
                            f"$ {i[2]:.4f} \\pm{i[3]:.4f}$ & "
                            f"{i[0]} "
                            f"\\\\" + "\n"
                        )
                else:
                    if include_dataset:
                        latex_string += (
                            f"{algorithm} & "
                            f"{norm_type} + {feature} & "
                            # print the accuracy out to 4 decimal places
                            f"$ {i[2]:.4f} \\pm{i[3]:.4f}$ & "
                            f"{i[4]} "
                            f"\\\\" + "\n"
                        )
                    else:
                        latex_string += (
                            f"{algorithm} & "
                            f"{norm_type} + {feature} & "
                            # print the accuracy out to 4 decimal places
                            f"$ {i[2]:.4f} \\pm{i[3]:.4f}$ "
                            f"\\\\" + "\n"
                        )
            else:
                if display:
                    if include_hash:
                        print(
                            f"{i[1]} & $ {i[2]:.4f} \\pm{i[3]:.4f}$ & {i[4]} & {i[0]} \\\\"
                        )
                    else:
                        print(f"{i[1]} & $ {i[2]:.4f} \\pm{i[3]:.4f}$ & {i[4]} \\\\")

                if include_hash:
                    if include_dataset:
                        latex_string += (
                            f"{algorithm} & "
                            f"$ {i[2]:.4f} \\pm{i[3]:.4f}$ & "
                            f"{i[4]} & "
                            f"{i[0]} "
                            f"\\\\" + "\n"
                        )
                    else:
                        latex_string += (
                            f"{algorithm} & "
                            f"$ {i[2]:.4f} \\pm{i[3]:.4f}$ & "
                            f"{i[0]} "
                            f"\\\\" + "\n"
                        )
                else:
                    if include_dataset:
                        latex_string += (
                            f"{algorithm} & "
                            f"$ {i[2]:.4f} \\pm{i[3]:.4f}$ & "
                            f"{i[4]} "
                            f"\\\\" + "\n"
                        )
                    else:
                        latex_string += (
                            f"{algorithm} & "
                            f"$ {i[2]:.4f} \\pm{i[3]:.4f}$ "
                            f"\\\\" + "\n"
                        )

        return latex_string

    @staticmethod
    def generate_analysis_report(
        inLaTeX=True,
        display=True,
        output_file=None,
        include_hash=False,
        num_best=1,
        timing=False,
    ):
        """generates a report of the best results for each algorithm of the
        algorithms within each of the datasets. this method returns 3
        different reports, one for each of the datasets. Each report is in
        latex format and can easily be used to make a table in latex.
        Parameters
        ----------
        inLaTeX : bool
            if True, the results will be printed in latex format
        display : bool
            if True, the results will be printed
        output_file : str
            the name of the file to write the results to
        include_hash : bool
            if True, the hash of the run will be included in the results
        num_best : int
            the number of best runs per each algorithm to include in the report
        timing : bool
            if True, the timing results will be included in the report
        Returns
        -------
        str
            the report of the best results for each algorithm printed out to
            a file or to the screen or both
        """
        t0 = time.time()
        # get the best results for each algorithm
        best_results = {}
        for dataset in DATA_SETS:
            best_results[dataset] = {}
            for algorithm in ALGORITHMS:
                best_results[dataset][
                    algorithm
                ] = OutlierDetector.get_best_algorithm_runs(
                    algorithm,
                    dataset=dataset,
                    N=num_best,
                    display=False,
                    include_hash=include_hash,
                )

        # sort each dataset by best accuracy score
        for dataset in best_results:
            if include_hash:
                best_results[dataset] = sorted(
                    best_results[dataset].items(),
                    key=lambda x: float(x[1].split(" ")[-4]),
                    reverse=True,
                )
            else:
                best_results[dataset] = sorted(
                    best_results[dataset].items(),
                    key=lambda x: float(x[1].split(" ")[-2]),
                    reverse=True,
                )

        # save the results to a file in latex format if the output_file is not None
        # check that the output file format is .tex
        if output_file is not None:
            if not output_file.endswith(".tex"):
                output_file += ".tex"
            with open(output_file, "w") as f:
                if inLaTeX:
                    # print the latex preamble code
                    f.write("\\documentclass{article}\n")
                    f.write("\\usepackage[utf8]{inputenc}\n")
                    f.write("\\usepackage{booktabs}\n")
                    f.write("\\begin{document}\n")

                for dataset in best_results:
                    ds = dataset
                    # split the % sign from the dataset name and then join it back
                    # with a \% to make it visible in latex
                    if inLaTeX:
                        ds = dataset.split("%")[0] + "\%"
                        f.write(f"\\begin{{table}}[]\n")
                        f.write(f"\\centering")
                        if include_hash:
                            f.write(f"\\begin{{tabular}}{{c|c|c|c}}\n")
                            f.write(f"\\toprule\n")
                            f.write(f"Algorithm & Feature & Accuracy & hash \\\\\n")
                        else:
                            f.write(f"\\begin{{tabular}}{{c|c|c}}\n")
                            f.write(f"\\toprule\n")
                            f.write(f"Algorithm & Feature & Accuracy \\\\\n")
                        f.write(f"\\midrule\n")
                    for i in best_results[dataset]:
                        f.write(i[1])
                    if inLaTeX:
                        f.write(f"\\bottomrule\n")
                        f.write(f"\\end{{tabular}}\n")
                        f.write(f"\\caption{{Best results for {ds}}}\n")
                        f.write(f"\\label{{tab:my_label}}\n")
                        f.write(f"\\end{{table}}\n")
                if inLaTeX:
                    f.write("\\end{document}\n")
            f.close()

        if display:
            if inLaTeX:
                print("\\documentclass{article}")
                print("\\usepackage[utf8]{inputenc}")
                print("\\usepackage{booktabs}")
                print("\\begin{document}")

            # print the best results for each algorithm
            for dataset in best_results:
                # split the % sign from the dataset name and then join it back
                # with a \% to make it visible in latex
                ds = dataset
                if inLaTeX:
                    ds = dataset.split("%")[0] + "\%"
                    print(f"\\begin{{table}}[]")
                    print(f"\\centering")
                    if include_hash:
                        print(f"\\begin{{tabular}}{{c|c|c|c}}")
                        print(f"\\toprule")
                        print(f"Algorithm & Feature & Accuracy & hash \\\\")
                    else:
                        print(f"\\begin{{tabular}}{{c|c|c}}")
                        print(f"\\toprule")
                        print(f"Algorithm & Feature & Accuracy \\\\")
                    print(f"\\midrule")
                for i in best_results[dataset]:
                    print(i[1], end="")
                if inLaTeX:
                    print(f"\\bottomrule")
                    print(f"\\end{{tabular}}")
                    print(f"\\caption{{Best results for {ds}}}")
                    print(f"\\label{{tab:my_label}}")
                    print(f"\\end{{table}}")
            if inLaTeX:
                print("\\end{document}")

        if timing:
            print(f"generate_analysis_report took {time.time() - t0} seconds")

    @staticmethod
    def generate_analysis_report_stats(
        inLaTeX=True,
        display=True,
        output_file=None,
        include_hash=False,
        num_best=1,
        return_it=False,
        timing=False,
    ):
        """generates a report of the best results for each algorithm of the
        algorithms within each of the datasets. this method returns 3
        different reports, one for each of the datasets. Each report is in
        latex format and can easily be used to make a table in latex.
        Parameters
        ----------
        inLaTeX : bool
            if True, the results will be printed in latex format
        display : bool
            if True, the results will be printed
        output_file : str
            the name of the file to write the results to
        include_hash : bool
            if True, the hash of the run will be included in the results
        num_best : int
            the number of best runs per each algorithm to include in the report
        timing : bool
            if True, the timing results will be included in the report
        Returns
        -------
        str
            the report of the best results for each algorithm printed out to
            a file or to the screen or both
        """
        t0 = time.time()
        # get the best results for each algorithm
        best_results = {}
        for dataset in DATA_SETS:
            best_results[dataset] = {}
            for algorithm in ALGORITHMS:
                best_results[dataset][
                    algorithm
                ] = OutlierDetector.get_best_algorithm_runs_stats(
                    algorithm,
                    dataset=dataset,
                    N=num_best,
                    display=False,
                    include_hash=include_hash,
                )
        # check that the best_results does not contain keys with no values and if
        # it does, remove them
        # empty_keys = [k for k, v in best_results.items() if not v]
        # for k in empty_keys:
        #     del best_results[k]

        # sort each dataset by best accuracy score
        for dataset in best_results:
            best_results[dataset] = sorted(
                best_results[dataset].items(),
                key=lambda x: float(x[1].split(" ")[7]),
                reverse=True,
            )
        # save the results to a file in latex format if the output_file is not None
        # check that the output file format is .tex
        if output_file is not None:
            if not output_file.endswith(".tex"):
                output_file += ".tex"
            with open(output_file, "w") as f:
                if inLaTeX and display:
                    # print the latex preamble code
                    f.write("\\documentclass{article}\n")
                    f.write("\\usepackage[utf8]{inputenc}\n")
                    f.write("\\usepackage{booktabs}\n")
                    f.write("\\begin{document}\n")

                for dataset in best_results:
                    ds = dataset
                    # split the % sign from the dataset name and then join it back
                    # with a \% to make it visible in latex
                    if inLaTeX:
                        ds = dataset.split("%")[0] + "\%"
                        f.write(f"\\begin{{table}}[]\n")
                        f.write(f"\\centering")
                        if include_hash:
                            f.write(f"\\begin{{tabular}}{{c|c|c|c}}\n")
                            f.write(f"\\toprule\n")
                            f.write(
                                f"Algorithm & Norm. + Feature & Accuracy & hash \\\\\n"
                            )
                        else:
                            f.write(f"\\begin{{tabular}}{{c|c|c}}\n")
                            f.write(f"\\toprule\n")
                            f.write(f"Algorithm & Norm. + Feature & Accuracy \\\\\n")
                        f.write(f"\\midrule\n")
                    for i in best_results[dataset]:
                        f.write(i[1])
                    if inLaTeX:
                        f.write(f"\\bottomrule\n")
                        f.write(f"\\end{{tabular}}\n")
                        f.write(f"\\caption{{Best results for {ds}}}\n")
                        f.write(f"\\label{{tab:my_label}}\n")
                        f.write(f"\\end{{table}}\n")
                if inLaTeX and display:
                    f.write("\\end{document}\n")
            f.close()

        if return_it:
            return best_results

        if display:
            if inLaTeX:
                print("\\documentclass{article}")
                print("\\usepackage[utf8]{inputenc}")
                print("\\usepackage{booktabs}")
                print("\\begin{document}")

            # print the best results for each algorithm
            for dataset in best_results:
                # split the % sign from the dataset name and then join it back
                # with a \% to make it visible in latex
                # print(f'dataset: {best_results[dataset]}')
                # iterate over the algorithms in the dataset and print each
                # algorithm's best results

                ds = dataset
                if inLaTeX:
                    ds = dataset.split("%")[0] + "\%"
                    print(f"\\begin{{table}}[]")
                    print(f"\\centering")
                    if include_hash:
                        print(f"\\begin{{tabular}}{{c|c|c|c}}")
                        print(f"\\toprule")
                        print(f"Algorithm & Norm. + Feature & Accuracy & hash \\\\")
                    else:
                        print(f"\\begin{{tabular}}{{c|c|c}}")
                        print(f"\\toprule")
                        print(f"Algorithm & Norm. + Feature & Accuracy \\\\")
                    print(f"\\midrule")
                for i in best_results[dataset]:
                    print(i[1], end="")
                if inLaTeX:
                    print(f"\\bottomrule")
                    print(f"\\end{{tabular}}")
                    print(f"\\caption{{Best results for {ds}}}")
                    print(f"\\label{{tab:my_label}}")
                    print(f"\\end{{table}}")
            if inLaTeX:
                print("\\end{document}")
        if timing:
            print(f"generate_analysis_report_stats took {time.time() - t0} seconds")

    @staticmethod
    def generate_analysis_report_stats_MIDL(
        inLaTeX=False,
        display=True,
        output_file=None,
        include_hash=False,
        num_best=1,
        timing=False,
    ):
        """ generates a report of the best results for each algorithm of the
        algorithms within each of the datasets. this method returns 3
        different reports, one for each of the datasets. Each report is in
        latex format and can easily be used to make a table in latex. Table is
        the following format:
        \noindent\textbf{Results on Datasets}
        \begin{table}[!ht]
        % \renewcommand\thetable{1, 2, 3}
        % \includegraphics[width=\textwidth]{images/result_datasets_ABC_tables.pdf}
        \resizebox{\textwidth}{!}{
        \begin{tabular}{ccc|ccc|ccc}
          \toprule
          \multicolumn{3}{c}{\textbf{Dataset A}} & \multicolumn{3}{c}{\textbf{Dataset B}} & \multicolumn{3}{c}{\textbf{Dataset C}}  \\
          \textbf{Algorithm} & \textbf{Norm. + Feature}  & \textbf{Score} & \textbf{Algorithm} & \textbf{Norm. + Feature}  & \textbf{Score} & \textbf{Algorithm} & \textbf{Norm. + Feature}  & \textbf{Score} \\
                \midrule
      DeepDVDD & Gaussian + Histogram & $0.6667\pm0.1105$ & LOF & SIFT & $0.6923$ & COF & SIFT & $0.8333$ \\
      DeepDVDD & Gaussian + Histogram & $0.6667\pm0.1105$ & LOF & SIFT & $0.6923$ & COF & SIFT & $0.8333$ \\
    \bottomrule
    \end{tabular}
        Parameters
        ----------
        inLaTeX : bool
            if True, the results will be printed in latex format
        display : bool
            if True, the results will be printed
        output_file : str
            the name of the file to write the results to
        include_hash : bool
            if True, the hash of the run will be included in the results
        num_best : int
            the number of best runs per each algorithm to include in the report
        timing : bool
            if True, the timing results will be included in the report
        Returns
        -------
        str
            the report of the best results for each algorithm printed out to
            a file or to the screen or both
        """
        t0 = time.time()
        # get the best results for each algorithm
        best_results = {}
        for dataset in DATA_SETS:
            best_results[dataset] = {}
            for algorithm in ALGORITHMS:
                best_results[dataset][
                    algorithm
                ] = OutlierDetector.get_best_algorithm_runs_stats(
                    algorithm,
                    dataset=dataset,
                    N=num_best,
                    display=False,
                    include_hash=include_hash,
                )

        # sort each dataset by best accuracy score
        for dataset in best_results:
            best_results[dataset] = sorted(
                best_results[dataset].items(),
                key=lambda x: float(x[1].split(" ")[7]),
                reverse=True,
            )
        # save the results to a file in latex format if the output_file is not None
        # check that the output file format is .tex
        # to through the best_resutls and replace every \\ with & and remove any \n
        # for item in best_results:
        #     for i in range(len(best_results[item])):
        #         best_results[item][i] = (best_results[item][i][0],
        #                                  best_results[item][i][1].replace('\n', ''))

        if output_file is not None:
            if not output_file.endswith(".tex"):
                output_file += ".tex"
            with open(output_file, "w") as f:
                if inLaTeX and display:
                    # print the latex preamble code
                    f.write("\\documentclass{article}\n")
                    f.write("\\usepackage[utf8]{inputenc}\n")
                    f.write("\\usepackage{booktabs}\n")
                    f.write("\\begin{document}\n")

                # each row has results for 3 datasets
                # print the best results for each algorithm
                for i in range(0, len(best_results[DATA_SETS[0]]), 3):
                    # print the row
                    if inLaTeX:
                        f.write("\\noindent\\textbf{Results on Datasets}\n")
                        f.write("\\begin{table}[!ht]\n")
                        f.write("\\resizebox{\\textwidth}{!}{\n")
                        f.write("\\begin{tabular}{ccc|ccc|ccc}\n")
                        f.write("\\toprule\n")
                        f.write(
                            "\\multicolumn{3}{c}{textbf{Dataset A}} & multicolumn{3}{c}{textbf{Dataset B}} & multicolumn{3}{c}{textbf{Dataset C}}  \\\\\n"
                        )
                        f.write(
                            "\\textbf{Algorithm} & textbf{Norm. + Feature}  & textbf{Score} & textbf{Algorithm} & textbf{Norm. + Feature}  & textbf{Score} & textbf{Algorithm} & textbf{Norm. + Feature}  & textbf{Score} \\\\\n"
                        )
                        f.write("\\midrule\n")
                    for j in range(3):
                        if i + j < len(best_results[DATA_SETS[0]]):
                            for dataset in DATA_SETS:
                                f.write(best_results[dataset][i + j][1] + " & ")
                            f.write("\\\\\n")
                    if inLaTeX:
                        f.write("\\bottomrule\n")
                        f.write("\\end{tabular}\n")
                        f.write("\\end{table}\n")
                if inLaTeX and display:
                    f.write("\\end{document}\n")
        # print the results to the screen
        if display:
            if inLaTeX:
                print("\\noindent\\textbf{Results on Datasets}")
                print("\\begin{table}[!ht]")
                print("\\resizebox{\\textwidth}{!}{")
                print("\\begin{tabular}{ccc|ccc|ccc}")
                print("\\toprule")
                print(
                    "\\multicolumn{3}{c}{textbf{Dataset A}} & multicolumn{3}{c}{textbf{Dataset B}} & multicolumn{3}{c}{textbf{Dataset C}}  \\\\"
                )
                print(
                    "\\textbf{Algorithm} & textbf{Norm. + Feature}  & textbf{Score} & textbf{Algorithm} & textbf{Norm. + Feature}  & textbf{Score} & textbf{Algorithm} & textbf{Norm. + Feature}  & textbf{Score} \\\\"
                )
                print("\\midrule")
            for i in range(0, len(best_results[DATA_SETS[0]]), 3):
                # print the row
                for j in range(3):
                    if i + j < len(best_results[DATA_SETS[0]]):
                        for dataset in DATA_SETS:
                            print(best_results[dataset][i + j][1], end=" & ")
                        print()
                if inLaTeX:
                    print("\\bottomrule")
                    print("\\end{tabular}")
                    print("\\end{table}")
        if timing:
            print(f"generate_analysis_report_stats took {time.time() - t0} seconds")
        return best_results

        # print the best results for each algorithm  @staticmethod

    def get_configuration(cache_key, cache_path=CACHE_PATH, cache_file=CACHE_FILE):
        """loads the cache file as a json file, if the file does not exist
        it will return None"""
        # if the cache file exists, load it
        if os.path.exists(cache_path + cache_file):
            with open(cache_path + cache_file, "r") as f:
                cache = json.loads(f.read())
            # if the cache key exists, return the results
            if cache_key in cache:
                return cache[cache_key + "_config"]
            # if the cache key does not exist, return None
            else:
                return None
        # if the cache file does not exist, return None
        else:
            return None

    @staticmethod
    def get_config_stats(cache_key, cache_path=CACHE_PATH, cache_file=STATS_CACHE):
        """
        Loads the configuration settings from the cache stats file
        """
        # if the cache file exists, load it
        if os.path.exists(cache_path + cache_file):
            with open(cache_path + cache_file, "r") as f:
                cache = json.loads(f.read())
            # if the cache key exists, return the results
            if cache_key in cache:
                return cache[cache_key]["configuration"]
            # if the cache key does not exist, return None
            else:
                return None

    @staticmethod
    def get_all_config_stats(cache_path=CACHE_PATH, cache_file=STATS_CACHE):
        """
        Loads the configuration settings from the cache stats file
        """
        # if the cache file exists, load it
        if os.path.exists(cache_path + cache_file):
            with open(cache_path + cache_file, "r") as f:
                cache = json.loads(f.read())
            configurations = {}
            for key in cache:
                configurations[key] = cache[key]["configuration"]
            return configurations

    @staticmethod
    def get_values_for_config_parameter(
        config_parameter, cache_path=CACHE_PATH, cache_file=STATS_CACHE
    ):
        """
        Returs all the different values used for a given configuration parameter
        """
        # if the cache file exists, load it
        if os.path.exists(cache_path + cache_file):
            with open(cache_path + cache_file, "r") as f:
                cache = json.loads(f.read())
            values = []
            for key in cache:
                if cache[key]["configuration"][config_parameter] not in values:
                    values.append(cache[key]["configuration"][config_parameter])
            return values

    @staticmethod
    def get_all_values_for_all_config_parameters(
        cache_path=CACHE_PATH, cache_file=STATS_CACHE
    ):
        """
        Returs all the different values used for a given configuration parameter
        """
        # if the cache file exists, load it
        if os.path.exists(cache_path + cache_file):
            with open(cache_path + cache_file, "r") as f:
                cache = json.loads(f.read())
            values = {}
            for key in cache:
                for parameter in cache[key]["configuration"]:
                    if parameter not in values:
                        values[parameter] = []
                    if cache[key]["configuration"][parameter] not in values[parameter]:
                        values[parameter].append(cache[key]["configuration"][parameter])
            return values

    def get_error_log(cache_key, algorithm, print_log=False):
        """loads the cache file as a json file, if the file does not exist
        it will return None"""
        # looks in the LOG_DIR for file with name {cache_key}_{algorithm}.txt
        # if the file exists, load it and print the contents as well as return
        # the contents as a list
        # if the file does not exist, return None
        if os.path.exists(LOG_DIR + f"{cache_key}_{algorithm}.txt"):
            with open(LOG_DIR + f"{cache_key}_{algorithm}.txt", "r") as f:
                errors = f.readlines()
                # print in pretty format if print_log is True
                if print_log:
                    print(json.dumps(errors, indent=4))
                return errors
        else:
            print("No error log exists")
            return None

    @staticmethod
    def view(data, train_scores=None, hists=True, ncols=None):
        """View data
        Parameters
        ----------
        data : list
            data to be viewed
        train_scores : list
            train scores to be viewed
        hists : bool
            Whether to show histograms
        ncols : int
            Number of columns in the plot
        """
        O.DataHelper.view_histogram_grid(
            images=data, train_scores=train_scores, hists=hists, ncols=ncols
        )

    @staticmethod
    def detect_outliers(
        features,
        imgs,
        pyod_algorithm,
        accuracy_score=False,
        number_bad=None,
        timing=False,
        id_=None,
        **kwargs,
    ):
        """Detect outliers using pyod algorithms"""

        t0 = time.time()

        errors = {}
        t_scores = []
        t_labels = []
        accuracy = None
        # get the verbose flag from kwargs
        redirect_output = kwargs.get("redirect_output", False)
        return_decision_function = kwargs.get("return_decision_function", False)

        try:
            if redirect_output:
                print(f"Running {pyod_algorithm}...", end=" ")
                out_file = "outlier_output.txt"
                with open(out_file, "w") as f:
                    with contextlib.redirect_stdout(f):
                        if return_decision_function:
                            (
                                t_scores,
                                t_labels,
                                t_decision_function,
                            ) = OutlierDetector._detect_outliers(
                                features, pyod_algorithm=pyod_algorithm, **kwargs
                            )

                            accuracy = t_decision_function
                        else:
                            t_scores, t_labels = OutlierDetector._detect_outliers(
                                features, pyod_algorithm=pyod_algorithm, **kwargs
                            )
            else:
                print(f"Running {pyod_algorithm}...")

                if return_decision_function:
                    (
                        t_scores,
                        t_labels,
                        t_decision_function,
                    ) = OutlierDetector._detect_outliers(
                        features, pyod_algorithm=pyod_algorithm, **kwargs
                    )

                    accuracy = t_decision_function

                else:
                    t_scores, t_labels = OutlierDetector._detect_outliers(
                        features, pyod_algorithm=pyod_algorithm, **kwargs
                    )

        # if the algorithm causes an error, skip it and move on
        except Exception as e:
            print(f"Error running {pyod_algorithm}")
            errors[pyod_algorithm] = e
            accuracy = -1
            t_labels = None
            # print out only the error stack trace
            traceback.print_exc()

        # if len(t_scores) == len(imgs):
        #     print(f'len of imgs and t_scores are equal for {pyod_algorithm}')
        #     if accuracy_score:
        #         accuracy = OutlierDetector.accuracy(imgs,
        #                                             [t_scores],
        #                                             BAD_IMAGE_ID_PATH,
        #                                             number_bad)
        #
        #         print(f'{pyod_algorithm} accuracy: {accuracy}')

        kwargs_hash = dict_to_hash(kwargs)
        # get the date and time to use as file name
        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        # if the id is not None add it as part of the file name
        if id_ is not None:
            date_time = f"{date_time}_{str(id_)}"

        # put the feature type name with the number bad to use in the cache file
        filename = f"{date_time}_{pyod_algorithm}%_{kwargs_hash}"

        print(
            f"about to save and len of tscore and imgs is {len(t_scores)} and {len(imgs)}"
        )

        OutlierDetector._save_outlier_path_log(filename, imgs, t_scores)

        if timing:
            display_timing(t0, "running " + pyod_algorithm)

        return t_scores, t_labels, accuracy

    @staticmethod
    def accuracy(imgs, train_scores, bad_ids, number_bad, verbose=False, timing=False):
        """Calculates the accuracy of a pyod algorithm
        Parameters
        ----------
        imgs : list
            The images to be analyzed
        train_scores : list, np.ndarray
            The anomaly scores of the training data
        bad_ids : str, list
            The path to a text file with a list of bad ids or a list of bad ids
        number_bad : int
            The number of bad images
        verbose : bool, optional (default=False)
            Whether to display the results
        timing : bool, optional (default=False)
            If True, the time to calculate the accuracy is returned
        Returns
        -------
        accuracy : float
            The accuracy of the algorithm
        """
        t0 = time.time()
        if isinstance(bad_ids, str):
            with open(bad_ids, "r") as f:
                bad_ids = f.read().splitlines()
        elif isinstance(bad_ids, list):
            pass
        else:
            raise ValueError("bad_ids must be a string or a list")

        # add all the images and trainscores into the ImageList which is
        # automatically sorted by trainscore

        img_list = []
        for i, img in enumerate(imgs):
            img_list.append(SimpleNamespace(image=img, train_scores=train_scores[0][i]))

        # set the counter to 0
        counter = 0

        # loop through the first number_bad images in the list
        for i in range(number_bad):
            # get the SOPInstanceUID of the image
            uid = img_list[i].image.SOPInstanceUID
            # check if the uid is in the bad_ids list
            if uid in bad_ids:
                # if it is, increment the counter
                counter += 1

        # calculate the accuracy as a decimal
        accuracy = counter / number_bad

        if verbose:
            print("Accuracy: {:.10f}".format(accuracy))
        if timing:
            print("accuracy ...took: {}s".format(time.time() - t0))
        return accuracy

    @staticmethod
    def generate_sd_and_avg(
        n_runs,
        norm,
        feature,
        pyod_algorithm,
        dataset,
        display=False,
        timing=False,
        seed_test=False,
        **kwargs,
    ):
        """Generates the standard deviation and average of the accuracy of
        a pyod algorithm
        Parameters
        ----------
        imgs : list
            The images to be analyzed
        n_runs : int
            The number of times to run the algorithm
        norm : str
            The normalization type
        feature : str
            The feature type
        pyod_algorithm : str
            The pyod algorithm to be used
        dataset : int
            The dataset to be used which are as follows:
            1: is the dataset with 8% errors, or config_num=5
            2: is the dataset with 13% errors, or config_num=6
            3: is the dataset with 24% errors, or config_num=7
        display : bool, optional (default=False)
            Whether to display the results
        timing : bool, optional (default=False)
            If True, the time to calculate the accuracy is returned
        seed_test : bool, optional (default=False)
            If True, the seeds are iterated through and the accuracy is tested
        kwargs : dict
            Keyword arguments to be passed to the pyod algorithm
        Returns
        -------
        min_accuracy : float, max_accuracy : float, avg_accuracy : float, standard_deviation : float
            The minimum, maximum, average, and standard deviation of the
            accuracy of the algorithm
        """
        t0 = time.time()
        accuracy_list = []
        verbose = kwargs.get("verbose", False)
        print(f"Running {pyod_algorithm} {n_runs} times...")
        if dataset == 1:
            print(f"populating dataset with 8% errors...")
            imgs = O.DataHelper.get2D(N=100, config_num=5, randomize=True)
            num_bad = 8
        elif dataset == 2:
            print(f"populating dataset with 13% errors...")
            imgs = O.DataHelper.get2D(N=100, config_num=6, randomize=True)
            num_bad = 13
        elif dataset == 3:
            print(f"populating dataset with 24% errors...")
            imgs = O.DataHelper.get2D(N=100, config_num=7, randomize=True)
            num_bad = 24
        else:
            raise ValueError("dataset must be 1, 2, or 3")

        print(f"extracting features...")
        normalized_imgs = O.Normalize.get_norm(pixels=imgs, norm_type=norm)

        feature_vector = O.Features.get_features(normalized_imgs, feature_type=feature)
        for i in range(n_runs):
            if seed_test:
                kwargs["random_state"] = i
            print(f"run {i + 1} of {n_runs}")
            train_scores, train_labels, accuracy = OutlierDetector.detect_outliers(
                features=feature_vector,
                imgs=imgs,
                pyod_algorithm=pyod_algorithm,
                display=False,
                number_bad=num_bad,
                **kwargs,
            )
            if display:
                print(f"Accuracy: {accuracy}, with seed {i}")
            # update the seed in the kwargs which is under the name 'random_state'

            accuracy_list.append(accuracy)
        min_accuracy = min(accuracy_list)
        max_accuracy = max(accuracy_list)
        avg_accuracy = sum(accuracy_list) / len(accuracy_list)
        standard_deviation = statistics.stdev(accuracy_list)

        if display:
            print(f"min accuracy: {min_accuracy}")
            print(f"max accuracy: {max_accuracy}")
            print(f"Average accuracy: {avg_accuracy}")
            print(f"Standard deviation: {standard_deviation}")

        # get the hash from the kwargs and truncate it to 6 characters
        kwargs_hash = hashlib.md5(str(kwargs).encode("utf-8")).hexdigest()[:6]

        OutlierDetector._save_statistics_log(
            pyod_algorithm=pyod_algorithm,
            norm_type=norm,
            feature_type=feature,
            n_runs=n_runs,
            dataset=dataset,
            min_accuracy=min_accuracy,
            max_accuracy=max_accuracy,
            avg_accuracy=avg_accuracy,
            standard_deviation=standard_deviation,
            configuration=kwargs,
            hash_id=kwargs_hash,
        )

        if timing:
            print("generate_sd_and_avg ...took: {}s".format(time.time() - t0))
        return min_accuracy, max_accuracy, avg_accuracy, standard_deviation

    @staticmethod
    def generate_std_avg_all_algs(
        n_runs,
        norm,
        feature,
        dataset,
        display=False,
        timing=False,
        **kwargs,
    ):
        """Generates the standard deviation and average of the accuracy of
        all pyod algorithms
        Parameters
        ----------
        n_runs : int
            The number of times to run the algorithm
        norm : str
            The normalization type
        feature : str
            The feature type
        dataset : int
            The dataset to be used which are as follows:
            1: is the dataset with 8% errors, or config_num=5
            2: is the dataset with 13% errors, or config_num=6
            3: is the dataset with 24% errors, or config_num=7
        display : bool, optional (default=False)
            Whether to display the results
        timing : bool, optional (default=False)
            If True, the time to calculate the accuracy is returned
        kwargs : dict
            Keyword arguments to be passed to the pyod algorithm
        """
        t0 = time.time()
        print(f"Running all algorithms {n_runs} times...")

        for alg in ALGORITHMS:
            accuracy_list = []
            print(f"Running algorithm {alg}...")
            for i in range(n_runs):
                if dataset == 1:
                    print(f"populating dataset with 8% errors...")
                    imgs = O.DataHelper.get2D(N=100, config_num=5, randomize=True)
                    num_bad = 8
                elif dataset == 2:
                    print(f"populating dataset with 13% errors...")
                    imgs = O.DataHelper.get2D(N=100, config_num=6, randomize=True)
                    num_bad = 13
                elif dataset == 3:
                    print(f"populating dataset with 24% errors...")
                    imgs = O.DataHelper.get2D(N=100, config_num=7, randomize=True)
                    num_bad = 24
                else:
                    raise ValueError("dataset must be 1, 2, or 3")

                print(f"extracting features...")
                normalized_imgs = O.Normalize.get_norm(pixels=imgs, norm_type=norm)
                feature_vector = O.Features.get_features(
                    normalized_imgs, feature_type=feature
                )
                print(f"run {i + 1} of {n_runs}")
                train_scores, train_labels, accuracy = OutlierDetector.detect_outliers(
                    features=feature_vector,
                    imgs=imgs,
                    pyod_algorithm=alg,
                    display=display,
                    number_bad=num_bad,
                    accuracy_score=True,
                    **kwargs,
                )
                print(f"Accuracy: {accuracy}, with i {i}")
                accuracy_list.append(accuracy)

            # go thoough the accuracy list and get rid of any None values
            accuracy_list = [x for x in accuracy_list if x is not None]

            # print(f'accuracy list: {accuracy_list}')
            if len(accuracy_list) != 0:
                min_accuracy = min(accuracy_list)
                max_accuracy = max(accuracy_list)
                avg_accuracy = sum(accuracy_list) / len(accuracy_list)
                standard_deviation = statistics.stdev(accuracy_list)

            if display:
                print(f"min accuracy: {min_accuracy}")
                print(f"max accuracy: {max_accuracy}")
                print(f"Average accuracy: {avg_accuracy}")
                print(f"Standard deviation: {standard_deviation}")

            # get the hash from the kwargs and truncate it to 6 characters
            hash_ = hashlib.md5(str(kwargs).encode("utf-8")).hexdigest()[:6]

            OutlierDetector._save_statistics_log(
                pyod_algorithm=alg,
                norm_type=norm,
                feature_type=feature,
                n_runs=n_runs,
                dataset=dataset,
                min_accuracy=min_accuracy,
                max_accuracy=max_accuracy,
                avg_accuracy=avg_accuracy,
                standard_deviation=standard_deviation,
                configuration=kwargs,
                hash_id=hash_,
            )

        OutlierDetector._sort_statistics_log()

        if timing:
            print("generate_sd_and_avg ...took: {}s".format(time.time() - t0))

    @staticmethod
    def _save_statistics_log(
        min_accuracy,
        max_accuracy,
        avg_accuracy,
        standard_deviation,
        pyod_algorithm,
        norm_type,
        feature_type,
        dataset,
        n_runs,
        configuration,
        hash_id,
    ):
        """Saves the statistics to a json file in the logs folder
        Parameters
        ----------
        min_accuracy : float
            The minimum accuracy of the algorithm
        max_accuracy : float
            The maximum accuracy of the algorithm
        avg_accuracy : float
            The average accuracy of the algorithm
        standard_deviation : float
            The standard deviation of the accuracy of the algorithm
        pyod_algorithm : str
            The pyod algorithm to be used
        norm_type : str
            The normalization type
        feature_type : str
            The feature type
        dataset : int
            The dataset to be used which are as follows:
            1: is the dataset with 8% errors, or config_num=5
            2: is the dataset with 13% errors, or config_num=6
            3: is the dataset with 24% errors, or config_num=7
        n_runs : int
            The number of times to run the algorithm
        """
        print("saving statistics to log...")
        if dataset == 1:
            dataset = "8%-errors"
        elif dataset == 2:
            dataset = "13%-errors"
        elif dataset == 3:
            dataset = "24%-errors"
        else:
            raise ValueError("dataset must be 1, 2, or 3")

        # make a cache key to use
        cache_key = f"{pyod_algorithm}_{norm_type}_{feature_type}_{dataset}_{hash_id}"

        # open the json file if it exists and update it, otherwise create it
        if os.path.exists(CACHE_PATH + STATS_CACHE):
            print("log file exists, updating...")
            with open(CACHE_PATH + STATS_CACHE, "r") as f:
                log = json.loads(f.read())
        else:
            print("log file does not exist, creating...")
            log = {}

        # add the results to the log
        log[cache_key] = {
            "min_accuracy": min_accuracy,
            "max_accuracy": max_accuracy,
            "avg_accuracy": avg_accuracy,
            "standard_deviation": standard_deviation,
            "pyod_algorithm": pyod_algorithm,
            "norm_type": norm_type,
            "feature_type": feature_type,
            "n_runs": n_runs,
            "configuration": configuration,
        }

        # save the log
        with open(CACHE_PATH + STATS_CACHE, "w") as f:
            json.dump(log, f, indent=4)
            print(f"log saved to {CACHE_PATH + STATS_CACHE}")
        # close the file
        f.close()

    @staticmethod
    def _sort_statistics_log():
        """Sorts the statistics log best to worst by average accuracy"""
        print("sorting statistics log...")
        with open(CACHE_PATH + STATS_CACHE, "r") as f:
            log = json.loads(f.read())
        sorted_log = {}
        for key, value in sorted(
            log.items(), key=lambda item: item[1]["avg_accuracy"], reverse=True
        ):
            sorted_log[key] = value
        with open(CACHE_PATH + STATS_CACHE, "w") as f:
            json.dump(sorted_log, f, indent=4)
            print(f"log saved to {CACHE_PATH + STATS_CACHE}")

    @staticmethod
    def get_statistics_log():
        """Gets the statistics log
        Returns
        -------
        dict
            The statistics log
        """
        with open(CACHE_PATH + STATS_CACHE, "r") as f:
            log = json.loads(f.read())
        return log

    @staticmethod
    def _detect_outliers(data_x, pyod_algorithm, **kwargs):
        """Detect outliers using PyOD algorithm. Default algorithm is HBOS.
        See PYOD documentation to see which arguments are available for each
        algorithm and to get description of each algorithm and how the
        arguments work with each algorithm.
        link: https://pyod.readthedocs.io/en/latest/pyod.html
        """
        return_decision_function = kwargs.get("return_decision_function", False)

        # # make sure data_x is a 2d array and not a 1d array or 3D array
        if isinstance(data_x, np.ndarray):
            if len(data_x.shape) == 1:
                data_x = data_x.reshape(-1, 1)
            elif len(data_x.shape) == 3:
                data_x = data_x.reshape(data_x.shape[0], -1)
        elif isinstance(data_x, list):
            for i in range(len(data_x)):
                if len(data_x[i]) == 1:
                    data_x[i] = np.pad(data_x[i], (0, len(data_x[0]) - 1), "constant")
                if len(data_x[i]) == 3:
                    data_x[i] = np.pad(data_x[i], (0, len(data_x[0]) - 3), "constant")
            data_x = np.array(data_x)

        if pyod_algorithm == "ECOD":
            if DEBUG:
                print("In ECOD algorithm")
            from pyod.models.ecod import ECOD

            n_jobs = kwargs.get("n_jobs", 1)
            contamination = kwargs.get("contamination", 0.1)
            clf = ECOD(n_jobs=n_jobs, contamination=contamination)

        elif pyod_algorithm == "LOF":
            if DEBUG:
                print("In LOF algorithm")
            from pyod.models.lof import LOF

            if "LOF" in kwargs:
                clf = LOF(**kwargs["LOF"])
            else:
                n_neighbors = kwargs.get("n_neighbors", 20)
                algorithm = kwargs.get("algorithm", "auto")
                leaf_size = kwargs.get("leaf_size", 30)
                metric = kwargs.get("metric", "minkowski")
                p = kwargs.get("p", 2)
                novelty = kwargs.get("novelty", True)
                n_jobs = kwargs.get("n_jobs", 1)
                contamination = kwargs.get("contamination", 0.1)
                metric_params = kwargs.get("metric_params", None)
                clf = LOF(
                    n_neighbors=n_neighbors,
                    algorithm=algorithm,
                    leaf_size=leaf_size,
                    metric=metric,
                    p=p,
                    metric_params=metric_params,
                    contamination=contamination,
                    n_jobs=n_jobs,
                    novelty=novelty,
                )

        elif pyod_algorithm == "OCSVM":
            if DEBUG:
                print("In OCSVM algorithm")
            from pyod.models.ocsvm import OCSVM

            if "OCSVM" in kwargs:
                clf = OCSVM(**kwargs["OCSVM"])
            else:
                kernel = kwargs.get("kernel", "rbf")
                degree = kwargs.get("degree", 3)
                gamma = kwargs.get("gamma", "auto")
                coef0 = kwargs.get("coef0", 0.0)
                tol = kwargs.get("tol", 1e-3)
                nu = kwargs.get("nu", 0.5)
                shrinking = kwargs.get("shrinking", True)
                cache_size = kwargs.get("cache_size", 200)
                verbose = kwargs.get("verbose", False)
                max_iter = kwargs.get("max_iter", -1)
                contamination = kwargs.get("contamination", 0.1)
                clf = OCSVM(
                    kernel=kernel,
                    degree=degree,
                    gamma=gamma,
                    coef0=coef0,
                    tol=tol,
                    nu=nu,
                    shrinking=shrinking,
                    cache_size=cache_size,
                    verbose=verbose,
                    max_iter=max_iter,
                    contamination=contamination,
                )

        elif pyod_algorithm == "IForest":
            if DEBUG:
                print("In IForest algorithm")
            from pyod.models.iforest import IForest

            if "IForest" in kwargs:
                clf = IForest(**kwargs["IForest"])
            else:
                n_estimators = kwargs.get("n_estimators", 100)
                max_samples = kwargs.get("max_samples", "auto")
                contamination = kwargs.get("contamination", 0.1)
                max_features = kwargs.get("max_features", 1.0)
                bootstrap = kwargs.get("bootstrap", False)  # had  this as True
                n_jobs = kwargs.get("n_jobs", 1)
                behaviour = kwargs.get("behaviour", "old")
                verbose = kwargs.get("verbose", 0)
                random_state = kwargs.get("random_state", None)
                if DEBUG:
                    print("n_estimators: ", n_estimators)
                    print("max_samples: ", max_samples)
                    print("contamination: ", contamination)
                    print("max_features: ", max_features)
                    print("bootstrap: ", bootstrap)
                    print("n_jobs: ", n_jobs)
                    print("behaviour: ", behaviour)
                    print("verbose: ", verbose)
                    print("random_state: ", random_state)
                clf = IForest(
                    n_estimators=n_estimators,
                    max_samples=max_samples,
                    contamination=contamination,
                    max_features=max_features,
                    bootstrap=bootstrap,
                    n_jobs=n_jobs,
                    behaviour=behaviour,
                    random_state=random_state,
                    verbose=verbose,
                )

        elif pyod_algorithm == "CBLOF":
            if DEBUG:
                print("In CBLOF algorithm")
            from pyod.models.cblof import CBLOF

            if "CBLOF" in kwargs:
                clf = CBLOF(**kwargs["CBLOF"])
            else:
                n_clusters = kwargs.get("n_clusters", 8)
                contamination = kwargs.get("contamination", 0.1)
                alpha = kwargs.get("alpha", 0.9)
                beta = kwargs.get("beta", 5)
                use_weights = kwargs.get("use_weights", False)
                check_estimator = kwargs.get("check_estimator", False)
                n_jobs = kwargs.get("n_jobs", 1)
                random_state = kwargs.get("random_state", None)
                clustering_estimator = kwargs.get("clustering_estimator", None)
                clf = CBLOF(
                    n_clusters=n_clusters,
                    contamination=contamination,
                    clustering_estimator=clustering_estimator,
                    alpha=alpha,
                    beta=beta,
                    use_weights=use_weights,
                    check_estimator=check_estimator,
                    random_state=random_state,
                    n_jobs=n_jobs,
                )

        elif pyod_algorithm == "COPOD":
            if DEBUG:
                print("In COPOD algorithm")
            from pyod.models.copod import COPOD

            if "COPOD" in kwargs:
                clf = COPOD(**kwargs["COPOD"])
            else:
                contamination = kwargs.get("contamination", 0.1)
                n_jobs = kwargs.get("n_jobs", 1)
                clf = COPOD(contamination=contamination, n_jobs=n_jobs)

        elif pyod_algorithm == "MCD":
            if DEBUG:
                print("In MCD algorithm")
            from pyod.models.mcd import MCD

            if "MCD" in kwargs:
                clf = MCD(**kwargs["MCD"])
            else:
                contamination = kwargs.get("contamination", 0.1)
                store_precision = kwargs.get("store_precision", True)
                assume_centered = kwargs.get("assume_centered", False)
                support_fraction = kwargs.get("support_fraction", None)
                random_state = kwargs.get("random_state", None)
                clf = MCD(
                    contamination=contamination,
                    store_precision=store_precision,
                    assume_centered=assume_centered,
                    support_fraction=support_fraction,
                    random_state=random_state,
                )

        elif pyod_algorithm == "SOS":
            if DEBUG:
                print("In SOS algorithm")
            from pyod.models.sos import SOS

            if "SOS" in kwargs:
                clf = SOS(**kwargs["SOS"])
            else:
                contamination = kwargs.get("contamination", 0.1)
                perplexity = kwargs.get("perplexity", 4.5)
                metric = kwargs.get("metric", "euclidean")
                eps = kwargs.get("eps", 1e-5)
                if DEBUG:
                    print("contamination: ", contamination)
                clf = SOS(
                    contamination=contamination,
                    perplexity=perplexity,
                    metric=metric,
                    eps=eps,
                )

        elif pyod_algorithm == "KDE":
            if DEBUG:
                print("In KDE algorithm")
            from pyod.models.kde import KDE

            if "KDE" in kwargs:
                clf = KDE(**kwargs["KDE"])
            else:
                contamination = kwargs.get("contamination", 0.1)
                bandwidth = kwargs.get("bandwidth", 1.0)
                algorithm = kwargs.get("algorithm", "auto")
                leaf_size = kwargs.get("leaf_size", 30)
                metric = kwargs.get("metric", "minkowski")
                metric_params = kwargs.get("metric_params", None)
                clf = KDE(
                    contamination=contamination,
                    bandwidth=bandwidth,
                    algorithm=algorithm,
                    leaf_size=leaf_size,
                    metric=metric,
                    metric_params=metric_params,
                )

        elif pyod_algorithm == "Sampling":
            if DEBUG:
                print("In Sampling algorithm")
            from pyod.models.sampling import Sampling

            if "Sampling" in kwargs:
                clf = Sampling(**kwargs["Sampling"])
            else:
                contamination = kwargs.get("contamination", 0.1)
                subset_size = kwargs.get("subset_size", 20)
                metric = kwargs.get("metric", "minkowski")
                random_state = kwargs.get("random_state", None)
                metric_params = kwargs.get("metric_params", None)
                print(f"random_state: {random_state}")
                clf = Sampling(
                    contamination=contamination,
                    subset_size=subset_size,
                    metric=metric,
                    metric_params=metric_params,
                    random_state=random_state,
                )

        elif pyod_algorithm == "GMM":
            if DEBUG:
                print("In GMM algorithm")
            from pyod.models.gmm import GMM

            if "GMM" in kwargs:
                clf = GMM(**kwargs["GMM"])
            else:
                n_components = kwargs.get("n_components", 1)
                covariance_type = kwargs.get("covariance_type", "full")
                tol = kwargs.get("tol", 1e-3)
                reg_covar = kwargs.get("reg_covar", 1e-6)
                max_iter = kwargs.get("max_iter", 100)
                n_init = kwargs.get("n_init", 1)
                init_params = kwargs.get("init_params", "kmeans")
                contamination = kwargs.get("contamination", 0.1)
                weights_init = kwargs.get("weights_init", None)
                warm_start = kwargs.get(
                    "warm_start", False
                )  # warm start was None not False
                random_state = kwargs.get("random_state", None)
                precisions_init = kwargs.get("precisions_init", None)
                means_init = kwargs.get("means_init", None)
                clf = GMM(
                    n_components=n_components,
                    covariance_type=covariance_type,
                    tol=tol,
                    reg_covar=reg_covar,
                    max_iter=max_iter,
                    n_init=n_init,
                    init_params=init_params,
                    contamination=contamination,
                    weights_init=weights_init,
                    means_init=means_init,
                    precisions_init=precisions_init,
                    random_state=random_state,
                    warm_start=warm_start,
                )

        elif pyod_algorithm == "PCA":
            if DEBUG:
                print("In PCA algorithm")
            from pyod.models.pca import PCA

            if "PCA" in kwargs:
                clf = PCA(**kwargs["PCA"])
            else:
                contamination = kwargs.get("contamination", 0.1)
                copy = kwargs.get("copy", True)
                whiten = kwargs.get("whiten", False)
                svd_solver = kwargs.get("svd_solver", "auto")
                tol = kwargs.get("tol", 0.0)
                iterated_power = kwargs.get("iterated_power", "auto")
                weighted = kwargs.get("weighted", True)
                standardization = kwargs.get("standardization", True)
                random_state = kwargs.get("random_state", None)
                n_selected_components = kwargs.get("n_selected_components", None)
                n_components = kwargs.get("n_components", None)
                clf = PCA(
                    n_components=n_components,
                    n_selected_components=n_selected_components,
                    contamination=contamination,
                    copy=copy,
                    whiten=whiten,
                    svd_solver=svd_solver,
                    tol=tol,
                    iterated_power=iterated_power,
                    random_state=random_state,
                    weighted=weighted,
                    standardization=standardization,
                )

        elif pyod_algorithm == "LMDD":
            if DEBUG:
                print("In LMDD algorithm")
            from pyod.models.lmdd import LMDD

            if "LMDD" in kwargs:
                clf = LMDD(**kwargs["LMDD"])
            else:
                contamination = kwargs.get("contamination", 0.1)
                n_iter = kwargs.get("n_iter", 50)
                dis_measure = kwargs.get("dis_measure", "aad")
                random_state = kwargs.get("random_state", None)
                clf = LMDD(
                    contamination=contamination,
                    n_iter=n_iter,
                    dis_measure=dis_measure,
                    random_state=random_state,
                )

        elif pyod_algorithm == "COF":
            if DEBUG:
                print("In COF algorithm")
            from pyod.models.cof import COF

            if "COF" in kwargs:
                clf = COF(**kwargs["COF"])
            else:
                contamination = kwargs.get("contamination", 0.1)
                n_neighbors = kwargs.get("n_neighbors", 20)
                method = kwargs.get("method", "fast")
                clf = COF(
                    contamination=contamination, n_neighbors=n_neighbors, method=method
                )

        elif pyod_algorithm == "HBOS":
            if DEBUG:
                print("In HBOS algorithm")
            from pyod.models.hbos import HBOS

            if "HBOS" in kwargs:
                clf = HBOS(**kwargs["HBOS"])
            else:
                contamination = kwargs.get("contamination", 0.1)
                n_bins = kwargs.get("n_bins", 10)
                alpha = kwargs.get("alpha", 0.1)
                tol = kwargs.get("tol", 0.5)
                clf = HBOS(
                    contamination=contamination, n_bins=n_bins, alpha=alpha, tol=tol
                )

        elif pyod_algorithm == "KNN":
            if DEBUG:
                print("In KNN algorithm")
            from pyod.models.knn import KNN

            # look if kwargs["KNN"] is a dictionary value and if so use it
            if "KNN" in kwargs:
                clf = KNN(**kwargs["KNN"])
            else:
                contamination = kwargs.get("contamination", 0.1)
                n_neighbors = kwargs.get("n_neighbors", 5)
                method = kwargs.get("method", "largest")
                radius = kwargs.get("radius", 1.0)
                algorithm = kwargs.get("algorithm", "auto")
                leaf_size = kwargs.get("leaf_size", 30)
                metric = kwargs.get("metric", "minkowski")
                p = kwargs.get("p", 2)
                n_jobs = kwargs.get("n_jobs", 1)
                metric_params = kwargs.get("metric_params", None)
                clf = KNN(
                    contamination=contamination,
                    n_neighbors=n_neighbors,
                    method=method,
                    radius=radius,
                    algorithm=algorithm,
                    leaf_size=leaf_size,
                    metric=metric,
                    p=p,
                    metric_params=metric_params,
                    n_jobs=n_jobs,
                )

        elif pyod_algorithm == "AvgKNN":
            if DEBUG:
                print("In AvgKNN algorithm")
            from pyod.models.knn import KNN

            if "AvgKNN" in kwargs:
                clf = KNN(**kwargs["AvgKNN"])
            else:
                contamination = kwargs.get("contamination", 0.1)
                n_neighbors = kwargs.get("n_neighbors", 5)
                method = kwargs.get("method", "mean")
                radius = kwargs.get("radius", 1.0)
                algorithm = kwargs.get("algorithm", "auto")
                leaf_size = kwargs.get("leaf_size", 30)
                metric = kwargs.get("metric", "minkowski")
                p = kwargs.get("p", 2)
                n_jobs = kwargs.get("n_jobs", 1)
                metric_params = kwargs.get("metric_params", None)
                clf = KNN(
                    contamination=contamination,
                    n_neighbors=n_neighbors,
                    method=method,
                    radius=radius,
                    algorithm=algorithm,
                    leaf_size=leaf_size,
                    metric=metric,
                    p=p,
                    metric_params=metric_params,
                    n_jobs=n_jobs,
                )

        elif pyod_algorithm == "MedKNN":
            if DEBUG:
                print("In MedKNN algorithm")
            from pyod.models.knn import KNN

            if "MedKNN" in kwargs:
                clf = KNN(**kwargs["MedKNN"])
            else:
                contamination = kwargs.get("contamination", 0.1)
                n_neighbors = kwargs.get("n_neighbors", 5)
                method = kwargs.get("method", "median")
                radius = kwargs.get("radius", 1.0)
                algorithm = kwargs.get("algorithm", "auto")
                leaf_size = kwargs.get("leaf_size", 30)
                metric = kwargs.get("metric", "minkowski")
                p = kwargs.get("p", 2)
                n_jobs = kwargs.get("n_jobs", 1)
                metric_params = kwargs.get("metric_params", None)
                clf = KNN(
                    contamination=contamination,
                    n_neighbors=n_neighbors,
                    method=method,
                    radius=radius,
                    algorithm=algorithm,
                    leaf_size=leaf_size,
                    metric=metric,
                    p=p,
                    metric_params=metric_params,
                    n_jobs=n_jobs,
                )

        elif pyod_algorithm == "SOD":
            if DEBUG:
                print("In SOD algorithm")
            from pyod.models.sod import SOD

            if "SOD" in kwargs:
                clf = SOD(**kwargs["SOD"])
            else:
                contamination = kwargs.get("contamination", 0.1)
                n_neighbors = kwargs.get("n_neighbors", 20)
                ref_set = kwargs.get("ref_set", 10)
                if ref_set >= n_neighbors:
                    ref_set = n_neighbors - 1
                alpha = kwargs.get("alpha", 0.8)
                clf = SOD(
                    contamination=contamination,
                    n_neighbors=n_neighbors,
                    ref_set=ref_set,
                    alpha=alpha,
                )

        elif pyod_algorithm == "ROD":
            if DEBUG:
                print("In ROD algorithm")
            from pyod.models.rod import ROD

            if "ROD" in kwargs:
                clf = ROD(**kwargs["ROD"])
            else:
                contamination = kwargs.get("contamination", 0.1)
                parallel_execution = kwargs.get("parallel_execution", False)
                clf = ROD(
                    contamination=contamination, parallel_execution=parallel_execution
                )

        elif pyod_algorithm == "INNE":
            if DEBUG:
                print("In INNE algorithm")
            from pyod.models.inne import INNE

            if "INNE" in kwargs:
                clf = INNE(**kwargs["INNE"])
            else:
                n_estimators = kwargs.get("n_estimators", 200)
                max_samples = kwargs.get("max_samples", "auto")
                contamination = kwargs.get("contamination", 0.1)
                random_state = kwargs.get("random_state", None)
                clf = INNE(
                    n_estimators=n_estimators,
                    max_samples=max_samples,
                    contamination=contamination,
                    random_state=random_state,
                )

        elif pyod_algorithm == "FB":
            if DEBUG:
                print("In FB algorithm")
            from pyod.models.feature_bagging import FeatureBagging

            if "FB" in kwargs:
                clf = FeatureBagging(**kwargs["FB"])
            else:
                n_estimators = kwargs.get("n_estimators", 10)
                contamination = kwargs.get("contamination", 0.1)
                max_features = kwargs.get("max_features", 1.0)
                bootstrap_features = kwargs.get("bootstrap_features", False)
                check_detector = kwargs.get(
                    "check_detector", True
                )  # was False when should be True
                check_estimator = kwargs.get("check_estimator", False)
                n_jobs = kwargs.get("n_jobs", 1)
                combination = kwargs.get("combination", "average")
                verbose = kwargs.get("verbose", 0)
                random_state = kwargs.get("random_state", None)
                estimator_params = kwargs.get("estimator_params", None)
                base_estimator = kwargs.get("base_estimator", None)
                clf = FeatureBagging(
                    base_estimator=base_estimator,
                    n_estimators=n_estimators,
                    contamination=contamination,
                    max_features=max_features,
                    bootstrap_features=bootstrap_features,
                    check_detector=check_detector,
                    check_estimator=check_estimator,
                    n_jobs=n_jobs,
                    random_state=random_state,
                    combination=combination,
                    verbose=verbose,
                    estimator_params=estimator_params,
                )

        elif pyod_algorithm == "LODA":
            if DEBUG:
                print("In LODA algorithm")
            from pyod.models.loda import LODA

            if "LODA" in kwargs:
                clf = LODA(**kwargs["LODA"])
            else:
                contamination = kwargs.get("contamination", 0.1)
                n_bins = kwargs.get("n_bins", 10)
                n_random_cuts = kwargs.get("n_random_cuts", 100)
                clf = LODA(
                    contamination=contamination,
                    n_bins=n_bins,
                    n_random_cuts=n_random_cuts,
                )

        elif pyod_algorithm == "SUOD":
            if DEBUG:
                print("In SUOD algorithm")
            from pyod.models.suod import SUOD

            if "SUOD" in kwargs:
                clf = SUOD(**kwargs["SUOD"])
            else:
                contamination = kwargs.get("contamination", 0.1)
                combination = kwargs.get("combination", "average")
                rp_flag_global = kwargs.get("rp_flag_global", True)
                target_dim_frac = kwargs.get("target_dim_frac", 0.5)
                jl_method = kwargs.get("jl_method", "basic")
                bps_flag = kwargs.get("bps_flag", True)
                approx_flag_global = kwargs.get("approx_flag_global", True)
                verbose = kwargs.get("verbose", False)
                rp_ng_clf_list = kwargs.get("rp_ng_clf_list", None)
                rp_clf_list = kwargs.get("rp_clf_list", None)
                n_jobs = kwargs.get("n_jobs", None)
                cost_forecast_loc_pred = kwargs.get("cost_forecast_loc_pred", None)
                cost_forecast_loc_fit = kwargs.get("cost_forecast_loc_fit", None)
                base_estimators = kwargs.get("base_estimators", None)
                approx_ng_clf_list = kwargs.get("approx_ng_clf_list", None)
                approx_clf_list = kwargs.get("approx_clf_list", None)
                approx_clf = kwargs.get("approx_clf", None)
                if base_estimators is not None:
                    base_estimators = OutlierDetector._init_base_detectors(
                        base_estimators
                    )
                clf = SUOD(
                    base_estimators=base_estimators,
                    contamination=contamination,
                    combination=combination,
                    n_jobs=n_jobs,
                    rp_clf_list=rp_clf_list,
                    rp_ng_clf_list=rp_ng_clf_list,
                    rp_flag_global=rp_flag_global,
                    target_dim_frac=target_dim_frac,
                    jl_method=jl_method,
                    bps_flag=bps_flag,
                    approx_clf_list=approx_clf_list,
                    approx_ng_clf_list=approx_ng_clf_list,
                    approx_flag_global=approx_flag_global,
                    approx_clf=approx_clf,
                    cost_forecast_loc_fit=cost_forecast_loc_fit,
                    cost_forecast_loc_pred=cost_forecast_loc_pred,
                    verbose=verbose,
                )

        elif pyod_algorithm == "AE":
            if DEBUG:
                print("In AE algorithm")
            from pyod.models.auto_encoder import AutoEncoder
            from keras.losses import mean_squared_error

            if "AE" in kwargs:
                clf = AutoEncoder(**kwargs["AE"])
            else:
                hidden_activation = kwargs.get("hidden_activation", "relu")
                output_activation = kwargs.get("output_activation", "sigmoid")
                loss = kwargs.get("loss", mean_squared_error)
                optimizer = kwargs.get("optimizer", "adam")
                epochs = kwargs.get("epochs", 100)
                batch_size = kwargs.get("batch_size", 32)
                dropout_rate = kwargs.get("dropout_rate", 0.2)
                l2_regularizer = kwargs.get("l2_regularizer", 0.1)
                validation_size = kwargs.get("validation_size", 0.1)
                preprocessing = kwargs.get("preprocessing", True)
                verbose = kwargs.get("verbose", 1)
                contamination = kwargs.get("contamination", 0.1)
                random_state = kwargs.get("random_state", None)
                hidden_neurons = kwargs.get("hidden_neurons", None)
                clf = AutoEncoder(
                    hidden_neurons=hidden_neurons,
                    hidden_activation=hidden_activation,
                    output_activation=output_activation,
                    loss=loss,
                    optimizer=optimizer,
                    epochs=epochs,
                    batch_size=batch_size,
                    dropout_rate=dropout_rate,
                    l2_regularizer=l2_regularizer,
                    validation_size=validation_size,
                    preprocessing=preprocessing,
                    verbose=verbose,
                    random_state=random_state,
                    contamination=contamination,
                )

        elif pyod_algorithm == "VAE":
            if DEBUG:
                print("In VAE algorithm")
            from pyod.models.vae import VAE
            from keras.losses import mse

            if "VAE" in kwargs:
                clf = VAE(**kwargs["VAE"])
            else:
                latent_dim = kwargs.get("latent_dim", 2)
                hidden_activation = kwargs.get("hidden_activation", "relu")
                output_activation = kwargs.get("output_activation", "sigmoid")
                loss = kwargs.get("loss", mse)
                optimizer = kwargs.get("optimizer", "adam")
                epochs = kwargs.get("epochs", 100)
                batch_size = kwargs.get("batch_size", 32)
                dropout_rate = kwargs.get("dropout_rate", 0.2)
                l2_regularizer = kwargs.get("l2_regularizer", 0.1)
                validation_size = kwargs.get("validation_size", 0.1)
                preprocessing = kwargs.get("preprocessing", True)
                verbose = kwargs.get("verbose", 1)
                contamination = kwargs.get("contamination", 0.1)
                gamma = kwargs.get("gamma", 1.0)
                capacity = kwargs.get("capacity", 0.0)
                random_state = kwargs.get("random_state", None)
                encoder_neurons = kwargs.get("encoder_neurons", None)
                decoder_neurons = kwargs.get("decoder_neurons", None)
                clf = VAE(
                    encoder_neurons=encoder_neurons,
                    decoder_neurons=decoder_neurons,
                    latent_dim=latent_dim,
                    hidden_activation=hidden_activation,
                    output_activation=output_activation,
                    loss=loss,
                    optimizer=optimizer,
                    epochs=epochs,
                    batch_size=batch_size,
                    dropout_rate=dropout_rate,
                    l2_regularizer=l2_regularizer,
                    validation_size=validation_size,
                    preprocessing=preprocessing,
                    verbose=verbose,
                    random_state=random_state,
                    contamination=contamination,
                    gamma=gamma,
                    capacity=capacity,
                )

        elif pyod_algorithm == "ABOD":
            if DEBUG:
                print("In ABOD algorithm")
            from pyod.models.abod import ABOD

            if "ABOD" in kwargs:
                clf = ABOD(**kwargs["ABOD"])
            else:
                contamination = kwargs.get("contamination", 0.1)
                n_neighbors = kwargs.get("n_neighbors", 5)
                method = kwargs.get("method", "fast")
                clf = ABOD(
                    contamination=contamination, n_neighbors=n_neighbors, method=method
                )

        elif pyod_algorithm == "SOGAAL":
            if DEBUG:
                print("In SOGAAL algorithm")
            from pyod.models.so_gaal import SO_GAAL

            if "SOGAAL" in kwargs:
                clf = SO_GAAL(**kwargs["SOGAAL"])
            else:
                stop_epochs = kwargs.get("stop_epochs", 20)
                lr_d = kwargs.get("lr_d", 0.01)
                lr_g = kwargs.get("lr_g", 0.0001)
                decay = kwargs.get("decay", 1e-6)
                momentum = kwargs.get("momentum", 0.9)
                contamination = kwargs.get("contamination", 0.1)
                clf = SO_GAAL(
                    stop_epochs=stop_epochs,
                    lr_d=lr_d,
                    lr_g=lr_g,
                    decay=decay,
                    momentum=momentum,
                    contamination=contamination,
                )

        elif pyod_algorithm == "MOGAAL":
            if DEBUG:
                print("In MOGAAL algorithm")
            from pyod.models.mo_gaal import MO_GAAL

            if "MOGAAL" in kwargs:
                clf = MO_GAAL(**kwargs["MOGAAL"])
            else:
                stop_epochs = kwargs.get("stop_epochs", 20)
                lr_d = kwargs.get("lr_d", 0.01)
                lr_g = kwargs.get("lr_g", 0.0001)
                decay = kwargs.get("decay", 1e-6)
                momentum = kwargs.get("momentum", 0.9)
                contamination = kwargs.get("contamination", 0.1)
                clf = MO_GAAL(
                    stop_epochs=stop_epochs,
                    lr_d=lr_d,
                    lr_g=lr_g,
                    decay=decay,
                    momentum=momentum,
                    contamination=contamination,
                )

        elif pyod_algorithm == "DeepSVDD":
            if DEBUG:
                print("In DeepSVDD algorithm")
            from pyod.models.deep_svdd import DeepSVDD

            if "DeepSVDD" in kwargs:
                clf = DeepSVDD(**kwargs["DeepSVDD"])
            else:
                use_ae = kwargs.get("use_ae", False)
                hidden_activation = kwargs.get("hidden_activation", "relu")
                output_activation = kwargs.get("output_activation", "sigmoid")
                optimizer = kwargs.get("optimizer", "adam")
                epochs = kwargs.get("epochs", 100)
                batch_size = kwargs.get("batch_size", 32)
                dropout_rate = kwargs.get("dropout_rate", 0.2)
                l2_regularizer = kwargs.get("l2_regularizer", 0.1)
                validation_size = kwargs.get("validation_size", 0.1)
                preprocessing = kwargs.get("preprocessing", True)
                verbose = kwargs.get("verbose", 1)
                contamination = kwargs.get("contamination", 0.1)
                random_state = kwargs.get("random_state", None)
                hidden_neurons = kwargs.get("hidden_neurons", None)
                c = kwargs.get("c", None)
                clf = DeepSVDD(
                    c=c,
                    use_ae=use_ae,
                    hidden_neurons=hidden_neurons,
                    hidden_activation=hidden_activation,
                    output_activation=output_activation,
                    optimizer=optimizer,
                    epochs=epochs,
                    batch_size=batch_size,
                    dropout_rate=dropout_rate,
                    l2_regularizer=l2_regularizer,
                    validation_size=validation_size,
                    preprocessing=preprocessing,
                    verbose=verbose,
                    random_state=random_state,
                    contamination=contamination,
                )

        elif pyod_algorithm == "AnoGAN":
            if DEBUG:
                print("In AnoGAN algorithm")
            from pyod.models.anogan import AnoGAN

            if "AnoGAN" in kwargs:
                clf = AnoGAN(**kwargs["AnoGAN"])
            else:
                activation_hidden = kwargs.get("activation_hidden", "tanh")
                dropout_rate = kwargs.get("dropout_rate", 0.2)
                latent_dim_G = kwargs.get("latent_dim_G", 2)
                G_layers = kwargs.get("G_layers", [20, 10, 3, 10, 20])
                verbose = kwargs.get("verbose", 0)
                D_layers = kwargs.get("D_layers", [20, 10, 5])
                index_D_layer_for_recon_error = kwargs.get(
                    "index_D_layer_for_recon_error", 1
                )
                epochs = kwargs.get("epochs", 500)
                preprocessing = kwargs.get("preprocessing", False)
                learning_rate = kwargs.get("learning_rate", 0.001)
                learning_rate_query = kwargs.get("learning_rate_query", 0.01)
                epochs_query = kwargs.get("epochs_query", 20)
                batch_size = kwargs.get("batch_size", 32)
                contamination = kwargs.get("contamination", 0.1)
                output_activation = kwargs.get("output_activation", None)
                clf = AnoGAN(
                    activation_hidden=activation_hidden,
                    dropout_rate=dropout_rate,
                    latent_dim_G=latent_dim_G,
                    G_layers=G_layers,
                    verbose=verbose,
                    D_layers=D_layers,
                    index_D_layer_for_recon_error=index_D_layer_for_recon_error,
                    epochs=epochs,
                    preprocessing=preprocessing,
                    learning_rate=learning_rate,
                    learning_rate_query=learning_rate_query,
                    epochs_query=epochs_query,
                    batch_size=batch_size,
                    output_activation=output_activation,
                    contamination=contamination,
                )

        elif pyod_algorithm == "CD":
            if DEBUG:
                print("In CD algorithm")
            from pyod.models.cd import CD

            if "CD" in kwargs:
                clf = CD(**kwargs["CD"])
            else:
                contamination = kwargs.get("contamination", 0.1)
                whitening = kwargs.get("whitening", True)
                rule_of_thumb = kwargs.get("rule_of_thumb", False)
                clf = CD(
                    contamination=contamination,
                    whitening=whitening,
                    rule_of_thumb=rule_of_thumb,
                )

        elif pyod_algorithm == "XGBOD":
            if DEBUG:
                print("In XGBOD algorithm")
            from pyod.models.xgbod import XGBOD

            if "XGBOD" in kwargs:
                clf = XGBOD(**kwargs["XGBOD"])
            else:
                max_depth = kwargs.get("max_depth", 3)
                learning_rate = kwargs.get("learning_rate", 0.1)
                n_estimators = kwargs.get("n_estimators", 100)
                silent = kwargs.get("silent", True)
                objective = kwargs.get("objective", "binary:logistic")
                booster = kwargs.get("booster", "gbtree")
                n_jobs = kwargs.get("n_jobs", 1)
                gamma = kwargs.get("gamma", 0)
                min_child_weight = kwargs.get("min_child_weight", 1)
                max_delta_step = kwargs.get("max_delta_step", 0)
                subsample = kwargs.get("subsample", 1)
                colsample_bytree = kwargs.get("colsample_bytree", 1)
                colsample_bylevel = kwargs.get("colsample_bylevel", 1)
                reg_alpha = kwargs.get("reg_alpha", 0)
                reg_lambda = kwargs.get("reg_lambda", 1)
                scale_pos_weight = kwargs.get("scale_pos_weight", 1)
                base_score = kwargs.get("base_score", 0.5)
                standardization_flag_list = kwargs.get(
                    "standardization_flag_list", None
                )
                random_state = kwargs.get("random_state", None)
                nthread = kwargs.get("nthread", None)
                estimator_list = kwargs.get("estimator_list", None)
                clf = XGBOD(
                    estimator_list=estimator_list,
                    standardization_flag_list=standardization_flag_list,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    silent=silent,
                    objective=objective,
                    booster=booster,
                    n_jobs=n_jobs,
                    nthread=nthread,
                    gamma=gamma,
                    min_child_weight=min_child_weight,
                    max_delta_step=max_delta_step,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    colsample_bylevel=colsample_bylevel,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    scale_pos_weight=scale_pos_weight,
                    base_score=base_score,
                    random_state=random_state,
                )

        else:
            raise ValueError("Algorithm not supported")

        clf.fit(data_x)

        # print(clf.decision_function)

        if return_decision_function:
            return clf.decision_scores_, clf.labels_, clf.decision_function

        return clf.decision_scores_, clf.labels_

    @staticmethod
    def _save_outlier_path_log(filename, imgs, t_scores):
        """Saves the outlier path log
        Parameters
        ----------
        filename : str
            The cache key
        t_scores : list, np.array
            The training scores
        imgs : list
            The images
        """
        # validate the length of the scores and images
        OutlierDetector._validate_lengths(imgs, t_scores)

        # the path to write the log to will be the LOG_DIR with the cache key
        # appended to it
        path = os.path.join(LOG_DIR, filename + ".txt")

        # create the log directory if it doesn't exist with all the parent directories
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        # this will put all the data into order from highest to lowest score
        image_list = []
        if isinstance(t_scores, np.ndarray):
            for i in range(len(imgs)):
                image_list.append(SimpleNamespace(image=imgs[i], score=t_scores[i]))
                image_list = sorted(image_list, key=lambda x: x.score, reverse=True)
        elif isinstance(t_scores, list):
            for i in range(len(imgs)):
                for j in range(len(t_scores)):
                    image_list.append(
                        SimpleNamespace(image=imgs[i], score=t_scores[j][i])
                    )
                    image_list = sorted(image_list, key=lambda x: x.score, reverse=True)

        # write the paths in order they are in the image_list
        with open(path, "w") as f:
            for i in range(len(image_list)):
                f.write(image_list[i].image.filePath + "\n")

        # sort the scores from highest to lowest and then write them to
        # another file with the same name as the path but with _scores.txt
        # appended to it
        path = os.path.join(LOG_DIR, filename + "_scores.txt")
        with open(path, "w") as f:
            for i in range(len(image_list)):
                f.write(str(image_list[i].score) + "\n")

    @staticmethod
    def _validate_lengths(data_x, labels):
        """Validates the lengths of the data
        Parameters
        ----------
        data_x : list
            The data to validate
        labels : list, np.ndarray
            The labels to validate
        """
        if isinstance(labels, list):
            for label in labels:
                if len(data_x) != len(label):
                    raise ValueError(
                        "LengthError: The length of the data and labels must be the same"
                    )
        else:
            if len(data_x) != len(labels):
                raise ValueError("The length of the data and labels must be the same")

    @staticmethod
    def _init_base_detectors(base_detectors):
        """Initializes the base detectors
        Parameters
        ----------
        base_detectors : list
            The base detectors to use
        **kwargs : dict
            The kwargs to pass to the base detectors
        Returns
        -------
        base_detectors : list
            The initialized base detectors
        """
        base_temp = []
        for detector in base_detectors:
            if detector == "ECOD":
                from pyod.models.ecod import ECOD

                base_temp.append(ECOD())
            elif detector == "LOF":
                from pyod.models.lof import LOF

                base_temp.append(LOF())
            elif detector == "OCSVM":
                from pyod.models.ocsvm import OCSVM

                base_temp.append(OCSVM())
            elif detector == "IForest":
                from pyod.models.iforest import IForest

                base_temp.append(IForest())
            elif detector == "CBLOF":
                from pyod.models.cblof import CBLOF

                base_temp.append(CBLOF())
            elif detector == "COPOD":
                from pyod.models.copod import COPOD

                base_temp.append(COPOD())
            elif detector == "SOS":
                from pyod.models.sos import SOS

                base_temp.append(SOS())
            elif detector == "KDE":
                from pyod.models.kde import KDE

                base_temp.append(KDE())
            elif detector == "Sampling":
                from pyod.models.sampling import Sampling

                base_temp.append(Sampling())
            elif detector == "GMM":
                from pyod.models.gmm import GMM

                base_temp.append(GMM())
            elif detector == "PCA":
                from pyod.models.pca import PCA

                base_temp.append(PCA())
            elif detector == "MCD":
                from pyod.models.mcd import MCD

                base_temp.append(MCD())
            elif detector == "LMDD":
                from pyod.models.lmdd import LMDD

                base_temp.append(LMDD())
            elif detector == "COF":
                from pyod.models.cof import COF

                base_temp.append(COF())
            elif detector == "HBOS":
                from pyod.models.hbos import HBOS

                base_temp.append(HBOS())
            elif detector == "KNN":
                from pyod.models.knn import KNN

                base_temp.append(KNN())
            elif detector == "AvgKNN":
                from pyod.models.knn import KNN

                base_temp.append(KNN(method="mean"))
            elif detector == "MedKNN":
                from pyod.models.knn import KNN

                base_temp.append(KNN(method="median"))
            elif detector == "SOD":
                from pyod.models.sod import SOD

                base_temp.append(SOD())
            elif detector == "INNE":
                from pyod.models.inne import INNE

                base_temp.append(INNE())
            elif detector == "FB":
                from pyod.models.feature_bagging import FeatureBagging

                base_temp.append(FeatureBagging())
            elif detector == "LODA":
                from pyod.models.loda import LODA

                base_temp.append(LODA())
            elif detector == "SUOD":
                from pyod.models.suod import SUOD

                base_temp.append(SUOD())
            elif detector == "AE":
                from pyod.models.auto_encoder import AutoEncoder

                base_temp.append(AutoEncoder())
            elif detector == "VAE":
                from pyod.models.vae import VAE

                base_temp.append(VAE())
            elif detector == "SOGAAL":
                from pyod.models.so_gaal import SO_GAAL

                base_temp.append(SO_GAAL())
            elif detector == "DeepSVDD":
                from pyod.models.deep_svdd import DeepSVDD

                base_temp.append(DeepSVDD())
            elif detector == "AnoGAN":
                from pyod.models.anogan import AnoGAN

                base_temp.append(AnoGAN())
            else:
                continue

        return base_temp


def dict_to_hash(d):
    """Converts a dictionary to a hash and returns it as a string
    Parameters
    ----------
    d : dict
        The dictionary to be converted
    Returns
    -------
    hash : str
        The hash of the dictionary
    """
    return hashlib.md5(str(d).encode("utf-8")).hexdigest()


def display_timing(t0, label: str):
    """Prints the time it takes to perform a certain action

    Parameters
    ----------
    t0 : float
        the time when the action was performed
    label : str
        the label of the action
    """
    print(
        "{:<25s}{:<10s}{:>10f}{:^5s}".format(
            label, "...took ", time.time() - t0, " seconds"
        )
    )
