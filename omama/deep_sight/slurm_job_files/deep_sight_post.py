import json
import time
from datetime import datetime
import getpass
import uuid
import pydicom as dicom
import omama.data as d
from omama.algorithms import Algorithms
import os

# _____________________ variables below need to get set: _______________________
ERROR_CODE_CSV_PATH = r"/raid/mpsych/acceptance_criteria.csv"

DEEPSIGHT_SCRIPT = os.path.dirname(os.path.realpath(__file__)) + "/deepsight2.sh"

OUTPUT_DIRECTORY = r"/raid/mpsych/deepsight_out/"

CACHE_PATH = r"/raid/mpsych/cache_files/predictions_cache.json"

CASELIST_FILE_NAME = r"caselist.txt"


# _____________________ end of variables to set _______________________________!


class DeepSightPost(Algorithms):
    """
    DeepSight class is used to run the DeepSight classifier.
    """

    sop_to_path_dict = {}

    @staticmethod
    def _generate_error_codes_dict(csv_path=None):
        """Generate a dictionary of error codes and their corresponding
        descriptions.
        Parameters
        ----------
        csv_path: str
            path to the csv file containing the error codes
        Returns
        -------
        error_codes_dict: dict
            dictionary of error codes and their corresponding descriptions
        """
        error_codes_dict = {}
        if csv_path is None:
            csv_path = ERROR_CODE_CSV_PATH
        with open(csv_path, "r") as csv_file:
            lines = csv_file.readlines()
            for line in lines:
                values = line.split(",")
                error_code = values[0]
                description = values[1]
                error_codes_dict[error_code] = description

        return error_codes_dict

    # --------------------------------------------------------------------------
    @staticmethod
    def _make_caselist_file(cases, path, filename):
        """Make a file with the list of cases to be processed
        Parameters
        ----------
        cases: list, str
            list of cases to be processed or a path to a caselist file
        path: str
            path to the directory where the file will be created
        filename: str
            name of the file to be created
        Returns
        -------
        path: str
            path to the file that was created
        """
        if isinstance(cases, str):
            if cases.endswith(".txt"):
                print("path is ", path)
                if not os.path.exists(path):
                    os.makedirs(path)
                with open(cases, "r") as f:
                    lines = f.readlines()
                with open(os.path.join(path, filename), "w") as f:
                    for line in lines:
                        f.write(line)
                return os.path.join(path, filename)
            else:
                # is a path to a dicom so make this into a caselist file
                if not os.path.exists(path):
                    os.makedirs(path)
                with open(os.path.join(path, filename), "w") as f:
                    f.write(cases)
                return os.path.join(path, filename)
        # else is a list of strings
        elif isinstance(cases, list):
            if isinstance(cases[0], str):
                if not os.path.exists(path):
                    os.makedirs(path)
                with open(os.path.join(path, filename), "w") as f:
                    for case in cases:
                        if case != cases[-1]:
                            f.write(case + "\n")
                        else:
                            f.write(case)
                return os.path.join(path, filename)
            else:
                with open(path + filename, "w") as f:
                    for case in cases:
                        if case != cases[-1]:
                            f.write(case.filePath + "\n")
                        else:
                            f.write(case.filePath)
                caselist_file = path + filename
                return caselist_file
        else:
            with open(path + filename, "w") as f:
                if cases[-1] != "\n":
                    f.write(cases.filePath + "\n")
                else:
                    f.write(cases.filePath)
            caselist_file = path + filename
            return caselist_file

    # --------------------------------------------------------------------------
    @staticmethod
    def _make_study_uid_dict(cases):
        """Make a dict of StudyInstanceUIDs
        Parameters
        ----------
        cases: list, str, SimpleNamespace
            list of cases to be processed
        Returns
        -------
        study_uid_dict: dict
            dict of StudyInstanceUIDs
        """
        study_uid_dict = {}

        if isinstance(cases, str):
            print("make study uid dict, string case")
            if cases.endswith(".txt"):
                print("end with .txt")
                i = 0
                with open(cases, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("/"):
                            path = line
                            ds = dicom.dcmread(path, stop_before_pixels=True)
                            study_uid = os.path.split(os.path.split(path)[0])[1]
                            sop = ds.SOPInstanceUID
                            DeepSightPost.sop_to_path_dict[sop] = path
                            study_uid_dict = DeepSightPost._modify_study_dict(
                                study_uid_dict, study_uid, sop
                            )
                        i += 1
            else:
                # is a path to a single dicom so read the dicom and get the
                # study uid and sop uid
                ds = dicom.dcmread(cases, stop_before_pixels=True)
                study_uid = os.path.split(os.path.split(cases)[0])[1]
                sop = ds.SOPInstanceUID
                DeepSightPost.sop_to_path_dict[sop] = cases
                study_uid_dict = DeepSightPost._modify_study_dict(
                    study_uid_dict, study_uid, sop
                )
        elif isinstance(cases, list):
            # is a list of strings
            if isinstance(cases[0], str):
                for case in cases:
                    ds = dicom.dcmread(case, stop_before_pixels=True)
                    study_uid = os.path.split(os.path.split(case)[0])[1]
                    sop = ds.SOPInstanceUID
                    DeepSightPost.sop_to_path_dict[sop] = case
                    study_uid_dict = DeepSightPost._modify_study_dict(
                        study_uid_dict, study_uid, sop
                    )
            else:  # is a list of Data.get_image objects
                for case in cases:
                    study_uid = case.StudyInstanceUID
                    path = case.filePath
                    sop = case.SOPInstanceUID
                    DeepSightPost.sop_to_path_dict[sop] = path
                    study_uid_dict = DeepSightPost._modify_study_dict(
                        study_uid_dict, study_uid, sop
                    )
        else:
            # is a Data.get_image -> SimpleNamespace object
            study_uid = cases.StudyInstanceUID
            path = cases.filePath
            sop = cases.SOPInstanceUID
            DeepSightPost.sop_to_path_dict[sop] = path
            study_uid_dict = DeepSightPost._modify_study_dict(
                study_uid_dict, study_uid, sop
            )
        print(len(study_uid_dict))
        return study_uid_dict

    # --------------------------------------------------------------------------
    @staticmethod
    def _modify_study_dict(study_dict, uid, sop_uid):
        """Check if the study dict contains the value of the study uid. If
        it does, add the sop uid to the list of sops. If it doesn't, add the
        study uid to the dict and add the sop uid as the studies first sop.
        Parameters
        ----------
        study_dict: dict
            dict of study uids
        uid: str
            per-study identifier
        sop_uid: str
            per-image identifier
        Returns
        -------
        study_dict: dict
            dict of updated study uids
        """
        if uid not in study_dict:
            study_dict[uid] = [sop_uid]
        else:
            prev = study_dict[uid]
            prev.append(sop_uid)
            study_dict[uid] = prev

        return study_dict

    # --------------------------------------------------------------------------
    @staticmethod
    def _make_log_file(t0, path, filename, task_num):
        """Make a file with log information
        Parameters
        ----------
        t0: float
            time when the script started
        path: str
            path to the directory where the file will be created
        filename: str
            name of the file to be created
        task_num: int
            number to identify the task that was run for cases of multiple tasks
        """
        # make the log file using mkdir -p
        os.system("touch " + path + filename)
        with open(path + filename, "w") as f:
            f.write("username: " + getpass.getuser() + "\n")
            f.write("date_time: " + str(datetime.now()) + "\n")
            # add the total time the script took to run
            f.write("total_time: " + str(time.time() - t0) + "\n")
            # add the current path to the log file
            f.write("log_path: " + path + "\n")
            if task_num is not None:
                f.write("task_num: " + str(task_num) + "\n")

    # --------------------------------------------------------------------------
    @staticmethod
    def _generate_unique_filename():
        """Generate a unique filename using the uuid module
        Returns
        -------
        filename: str
            unique filename
        """
        # # make sure the tempfile will not be deleted
        # folder = tempfile.mkdtemp(suffix=None, prefix=None,
        #                           dir=output_path)
        # # only get the last part of the path
        # folder = os.path.split(folder)[1]
        return str(uuid.uuid4().hex)

    # --------------------------------------------------------------------------
    @staticmethod
    def _validate_unique_filename(path, filename):
        """Validate that the filename is unique
        Parameters
        ----------
        path: str
            path to the directory where the file will be created
        filename: str
            name of the file to be created
        Returns
        -------
        bool
            True if the filename is unique, False otherwise
        """
        count = 0
        for f in os.listdir(path):
            if f.endswith(filename):
                count += 1
        if count > 1:
            return False
        else:
            return True

    # --------------------------------------------------------------------------
    @staticmethod
    def _parse_log_to_dictionary(log):
        """Parse the log file into a dictionary
        Parameters
        ----------
        log: str
            log file to be parsed
        Returns
        -------
        log_dict: dict
            dictionary of log information
        """
        log_dict = {}
        for line in log.split("\n"):
            val = line.split(": ")
            if len(val) > 1:
                log_dict[val[0]] = val[1]
        return log_dict

    # --------------------------------------------------------------------------
    @staticmethod
    def _get_all_logs(timing=False):
        """Get all the logs in the path and returns a dictionary of all the
        appended logs in the path
        Parameters
        ----------
        timing: bool
            True if the timing information should be included, False otherwise
        Returns
        -------
        log_dict: dict
            dictionary of all the logs in the path
        """
        t0 = time.time()
        path = OUTPUT_DIRECTORY
        if not os.path.exists(path):
            print("Path does not exist")
            return None
        logs = {}
        counter = 0
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.startswith("log"):
                    with open(os.path.join(root, file), "r") as f:
                        key = file.replace(".txt", "") + "_" + str(counter)
                        contents = f.read()
                        case = DeepSightPost._parse_log_to_dictionary(contents)
                        logs[key] = case
                    counter += 1
        if timing:
            print("Time to get all logs: ", time.time() - t0)
        return logs

    # --------------------------------------------------------------------------
    @staticmethod
    def parse_sop_uid_from_paths(paths, substrs_to_remove=None, timing=False):
        """Parse the SOPInstanceUID from a list of paths and build a dictionary
        of SOPInstanceUIDs to paths.
        Parameters
        ----------
        paths : list
            list of paths to dicom files
        substrs_to_remove : str
            strings to remove from the beginning of the file name
        timing : bool
            (default is False) If true will time execution of method,
            else will not
        Returns
        -------
        sop_uids : list
            list of SOPInstanceUIDs
        """
        t0 = d.time.time()
        sop_uids = []
        sop_to_path_dict = {}
        if substrs_to_remove is None:
            substrs_to_remove = ["DXm.", "BT."]
        for path in paths:
            sop_uid = os.path.basename(path)
            for substr in substrs_to_remove:
                sop_uid = sop_uid.replace(substr, "")
            sop_to_path_dict[sop_uid] = path
            sop_uids.append(sop_uid)
        if timing is True:
            print("Time to parse SOP UIDs: ", d.time.time() - t0)
        return sop_uids, sop_to_path_dict

    # --------------------------------------------------------------------------
    @staticmethod
    def _create_errors_txt(path, sop_to_path_dict, preds):
        """Create a text file with the path of each dicom that had errors that
        occurred during the processing of the DICOM files.
        Parameters
        ----------
        path : str
            path to the directory where the file will be created
        sop_to_path_dict : dict
            dictionary of SOPInstanceUIDs to paths
        preds : dict
            dictionary of SOPInstanceUIDs to predictions
        """
        with open(path + "errors.txt", "w") as f:
            # for every sop in the preds dictionary that has a score of -1,
            # get the path from the sop_to_path_dict and write it to the file
            for k, v in preds.items():
                if v["score"] == -1:
                    if k in sop_to_path_dict:
                        f.write(sop_to_path_dict[k] + "\n")

    # --------------------------------------------------------------------------
    @staticmethod
    def _update_predictions_cache(predictions, cache_path, timing=False):
        """Update the predictions cache by appending the new predictions to the
        existing cache json file.
        Parameters
        ----------
        predictions : dict
            dictionary of predictions
        cache_path : str
            path to the cache file
        """
        t0 = d.time.time()
        with open(cache_path, "r") as f:
            cache = json.load(f)
        # add the new predictions to the cache dictionary
        cache.update(predictions)
        # write the new cache to the cache file
        with open(cache_path, "w") as f:
            json.dump(cache, f)
        if timing is True:
            print("Time to update cache: ", d.time.time() - t0)

    # --------------------------------------------------------------------------
    @staticmethod
    def _check_predictions_cache(caselist_file, cache_path=None, timing=False):
        """Check if the deepsight cache exists for the given SOP instance
        Parameters
        ----------
        caselist_file: str
            path to the caselist file
        cache_path: str
            path to the cache file
        Returns
        -------
        dict, caselist_file
            predictions dict and altered caselist_file
        """
        t0 = time.time()
        if cache_path is None:
            cache_path = CACHE_PATH
        # load the caselist file into a list
        with open(caselist_file, "r") as f:
            caselist = f.read().splitlines()

        # parse the caselist list into a list of SOP instance UIDs and
        # dictionary of SOP instance UIDs to paths
        sop_uids, sop_to_path_dict = DeepSightPost.parse_sop_uid_from_paths(
            caselist, timing=timing
        )
        # load the json file into a dictionary
        pred_cache = json.loads(open(cache_path, "r").read())
        predictions_dict = {}

        # check if the sop_uids are in the cache
        for sop_uid in sop_uids:
            if sop_uid in pred_cache:
                predictions_dict[sop_uid] = pred_cache[sop_uid]
                # remove the path that corresponds to the sop_uid from the
                # caselist_dict
                if sop_uid in sop_to_path_dict:
                    del sop_to_path_dict[sop_uid]

        run_deepsight = False
        # if there are any SOP UIDs in the caselist_dict that were not found
        # in the cache, then we need to run DeepSight on them so write only
        # them paths to the caselist file overwriting the previous caselist
        if len(sop_to_path_dict) > 0:
            run_deepsight = True
            caselist = []
            for sop_uid in sop_to_path_dict:
                caselist.append(sop_to_path_dict[sop_uid])
            with open(caselist_file, "w") as f:
                for line in caselist:
                    f.write(line + "\n")
        if timing:
            print("Time to check cache: ", time.time() - t0)
        # close any open files
        f.close()
        return predictions_dict, caselist_file, run_deepsight

    # --------------------------------------------------------------------------
    @staticmethod
    def get_logs(username=None, date=None, task_num=None, timing=False):
        """Get the log files based on matching the username, the date or both
        and returns a dictionary of the logs
        Parameters
        ----------
        username: str
            username to match
        date: str
            date to match
        timing: bool
            if True, print the time it took to get the logs
        task_num: int
            task number to match
        Returns
        -------
        log_dict: dict
            dictionary of all the logs in the path
        """
        t0 = time.time()
        # set path to the output directory
        path = OUTPUT_DIRECTORY
        if username is None and date is None:
            # will get all logs
            return DeepSightPost._get_all_logs(timing=timing)

        logs = {}
        counter = 0
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.startswith("log"):
                    with open(os.path.join(root, file), "r") as f:
                        log = f.read()
                        key = file.replace(".txt", "") + "_" + str(counter)
                    if username is not None and date is not None:
                        if username in log and date in log:
                            case = DeepSightPost._parse_log_to_dictionary(log)
                            logs[key] = case
                            counter += 1
                    elif username is not None:
                        if username in log:
                            case = DeepSightPost._parse_log_to_dictionary(log)
                            logs[key] = case
                            counter += 1
                    elif date is not None:
                        if date in log:
                            case = DeepSightPost._parse_log_to_dictionary(log)
                            logs[key] = case
                            counter += 1
                    elif task_num is not None:
                        if str(task_num) in log:
                            case = DeepSightPost._parse_log_to_dictionary(log)
                            logs[key] = case
                            counter += 1
        if timing:
            print("Time to get logs: ", time.time() - t0)
        return logs

    # --------------------------------------------------------------------------
    @staticmethod
    def get_predictions(folder_name):
        """Get the predictions from the specified folder
        Parameters
        ----------
        folder_name: str
            folder where the predictions are stored
        Returns
        -------
        predictions: dict
            dictionary of the predictions
        """
        predictions = {}
        # walk the output directory looking for the folder with the folder_name
        # and when found look inside for the predictions.json file and load
        for root, dirs, files in os.walk(OUTPUT_DIRECTORY):
            if folder_name in dirs:
                predictions_path = os.path.join(root, folder_name, "predictions.json")
                with open(predictions_path, "r") as f:
                    predictions = json.load(f)
        return predictions

    # --------------------------------------------------------------------------
    @staticmethod
    def run(
        caselist_file_loc,
        output_dir=None,
        deepsight_out=None,
        deepsight_script_path=None,
        caselist_file_name=None,
        ignore_checks=None,
        output_in_terminal=False,
        task_num=None,
        pred_cache_path=None,
        timing=False,
    ):
        """Run the DeepSight algorithm on the cases
        Parameters:
        ----------
            cases : list
                cases to be processed
            output_dir: str
                path to the output directory
            deepsight_script_path: str
                path to the DeepSight script
            caselist_file_name: str
                name of the caselist file
            ignore_checks : list
                ignore checks
            output_in_terminal: bool
                output the deepsight generated text to the terminal
            timing : bool
                if True, print the time it took to run the algorithm
        Returns: dict
        ----------
            predictions : dictionary of classifier predictions
        """
        t0 = time.time()

        if pred_cache_path is None:
            pred_cache_path = CACHE_PATH

        if deepsight_script_path is None:
            deepsight_script_path = DEEPSIGHT_SCRIPT

        # set the input nad output paths
        if output_dir is None:
            output_dir = OUTPUT_DIRECTORY

        if caselist_file_name is None:
            caselist_file_name = CASELIST_FILE_NAME

        output_location = output_dir  # output_dir + unique_filename + '/'
        deepsight_out = deepsight_out  # + date_time

        # make the error code dictionary
        error_codes_dict = (
            DeepSightPost._generate_error_codes_dict()
        )  # !!! Full set of error codes

        case_list_contents = open(caselist_file_loc).read().splitlines()
        # check for cache and get the flag to run the classifier
        predictions, caselist_file, run_flag = DeepSightPost._check_predictions_cache(
            caselist_file_loc, timing=timing
        )

        study_uid_dict = DeepSightPost._make_study_uid_dict(caselist_file_loc)

        # if the classifier needs to be run
        if run_flag:
            run_size = len(case_list_contents) - len(predictions)
            print(f"Running DeepSight on {run_size} cases, please be patient...")
            unfound_folders = []

            # get a list of folders in the output directory
            folders = sorted(os.listdir(deepsight_out))

            # for each folder, get the list of files
            for folder in folders:
                # make sure it is a directory first
                if os.path.isdir(os.path.join(deepsight_out, folder)):
                    # with folder as key for study_uid_dict get the
                    # SOPInstanceUID
                    if folder in study_uid_dict:
                        sop_instance_uid_list = study_uid_dict[folder]

                        # read-in classifier results output
                        f = open(deepsight_out + "/" + folder + "/results_full.json")
                        json_result = json.load(f)  # load the json file
                        # prediction results are present
                        if json_result["results_raw"] is not None:
                            # get the prediction with the highest score
                            for sopuid in sop_instance_uid_list:
                                if (
                                    sopuid
                                    in json_result["results_raw"]["dicom_results"]
                                ):
                                    coords = json_result["results_raw"][
                                        "dicom_results"
                                    ][sopuid]["none"][0]["coords"]
                                    score = json_result["results_raw"]["dicom_results"][
                                        sopuid
                                    ]["none"][0]["score"]
                                    if (
                                        "slice"
                                        in json_result["results_raw"]["dicom_results"][
                                            sopuid
                                        ]["none"][0]
                                    ):
                                        slice = json_result["results_raw"][
                                            "dicom_results"
                                        ][sopuid]["none"][0]["slice"]
                                    else:
                                        slice = 0
                                    case = {
                                        "coords": coords,
                                        "score": score,
                                        "slice": slice,
                                        "errors": None,
                                    }
                                    # add the prediction to the dictionary
                                    predictions[sopuid] = case
                        errors_dict = {}
                        id_to_sop_dict = {}
                        metadata_dict = json.loads(json_result["_metadata"])
                        for k, v in metadata_dict["SOPInstanceUID"].items():
                            if metadata_dict["failed_checks"][k] is not None:
                                id_to_sop_dict[int(k)] = metadata_dict[
                                    "SOPInstanceUID"
                                ][k]
                                for code in metadata_dict["failed_checks"][k]:
                                    if id_to_sop_dict[int(k)] in errors_dict:
                                        errors_dict[id_to_sop_dict[int(k)]].append(
                                            code
                                            + ": "
                                            + error_codes_dict[code].replace("\n", "")
                                        )
                                    else:
                                        errors_dict[id_to_sop_dict[int(k)]] = [
                                            code
                                            + ": "
                                            + error_codes_dict[code].replace("\n", "")
                                        ]
                                id_to_sop_dict[int(k)] = metadata_dict[
                                    "SOPInstanceUID"
                                ][k]
                                if errors_dict[id_to_sop_dict[int(k)]] is not None:
                                    case = {
                                        "coords": None,
                                        "score": -1,
                                        "slice": -1,
                                        "errors": errors_dict[id_to_sop_dict[int(k)]],
                                    }
                                    predictions[id_to_sop_dict[int(k)]] = case
                        f.close()
                    else:
                        unfound_folders.append(folder)
                else:
                    print(f"{folder} is not a directory")
            print(f"{len(unfound_folders)} folders were not found")
            print(f"folders: {unfound_folders}")

            # update the predictions cache file
            DeepSightPost._update_predictions_cache(
                predictions, pred_cache_path, timing=timing
            )
        # create the errors.txt file
        DeepSightPost._create_errors_txt(
            output_location, DeepSightPost.sop_to_path_dict, predictions
        )
        # create the local predictions.json file for this run
        with open(output_location + "predictions.json", "w") as fp:
            json.dump(predictions, fp)
        # add a log file to the output directory
        DeepSightPost._make_log_file(t0, output_location, "log.txt", task_num)
        # write the original caselist file back to the caselist file
        with open(output_location + "caselist.txt", "w") as f:
            for line in case_list_contents:
                # check if it is the last line and if it is, don't add a \n
                if line != case_list_contents[-1]:
                    f.write(line + "\n")
                else:
                    f.write(line)
        if timing:
            print(f"...took " + str(time.time() - t0))
        if len(predictions) == 0:
            raise ValueError(
                f"No predictions made, please check file located at"
                f" {output_location}deepsight_out.txt for possible errors"
            )
        return predictions


def build_predictions_cache(root_directory, location_directory):
    """
    Walks a directory and all its subdirectories and builds a master dictionary
    of all the predictions made by DeepSight to use in deepsight api calls.
    """
    predictions_cache = {}
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file == "predictions.json":
                # open the predictions.json file
                with open(os.path.join(root, file), "r") as f:
                    predictions = json.load(f)
                    for k, v in predictions.items():
                        predictions_cache[k] = v
    with open(location_directory + "predictions_cache.json", "w") as fp:
        json.dump(predictions_cache, fp)
    return predictions_cache
