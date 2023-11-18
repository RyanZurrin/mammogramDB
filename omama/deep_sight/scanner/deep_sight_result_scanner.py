# %load '/home/p.bendiksen001/deephealth/omama/omama/deep_sight/deep_sight_result_scanner.py'
import json
import os
import pickle
import sys
import matplotlib.pyplot as plt
import csv
from collections import Counter

sys.path.insert(0, "../../..")
import omama as O


paths = []
aggregate_preds_dict = {}
aggregate_errors_dict = {}
aggregate_errors = []
num_lines = 0
aggregate_num_lines = [0]
total_counter = [0]
total_counter_err = [0]


class Scanner:
    @staticmethod
    def list_pred_files(filepath, filetype):
        depth_counter = 0
        for root, dirs, files in os.walk(filepath):
            depth_counter += 1
            # print(f"root: {root}; dirs : {dirs} files: {files}")
            if "2d_out" not in root:
                continue
            if "log.txt" not in files and "predictions.json" not in files:
                continue
            for name in files:
                # print(f"counter {depth_counter}; name {name}")
                if name.lower().endswith(filetype.lower()):
                    json_path = os.path.join(root, name)
                    # print(f"2: {json_path}")
                    with open(json_path) as file:
                        preds_dict = json.load(file)
                    # print(f"preds: {preds_dict}")
                    inner_total_counter = 0
                    inner_error_counter = 0
                    for key in preds_dict:
                        inner_total_counter += 1
                        updated_tot_count = total_counter[0] + 1
                        total_counter.insert(0, updated_tot_count)
                        total_counter.pop(1)
                        if preds_dict[key]["errors"] != None:
                            aggregate_errors_dict[key] = preds_dict[key]["errors"]
                            inner_error_counter += 1
                            updated_tot_count = total_counter_err[0] + 1
                            total_counter_err.insert(0, updated_tot_count)
                            total_counter_err.pop(1)
                        else:
                            aggregate_preds_dict[key] = {}
                            aggregate_preds_dict[key]["coords"] = preds_dict[key][
                                "coords"
                            ]
                            aggregate_preds_dict[key]["score"] = preds_dict[key][
                                "score"
                            ]
                    print(
                        f"error % for {name} is {inner_error_counter/inner_total_counter}"
                    )
                    # paths.append(os.path.join(root, file))
                elif name == "caselist.txt":
                    with open(os.path.join(root, name), "r") as fp:
                        line_count = sum(1 for line in fp if line.rstrip())
                        total_line_count = aggregate_num_lines[0] + line_count
                        aggregate_num_lines.insert(0, total_line_count)
                        aggregate_num_lines.pop(1)
                        # print('Total lines:', aggregate_num_lines)  # 8
                elif name == "errors.txt":
                    error_list = open(os.path.join(root, name)).read().splitlines()
                    aggregate_errors.append(error_list)

        error_values = aggregate_errors_dict.values()
        error_values = [item for sublist in error_values for item in sublist]
        error_values = [item.rsplit(":")[0] for item in error_values]
        print(f"total error count : {total_counter_err[0]}")
        print(f"total error %: {total_counter_err[0] / total_counter[0]}")
        # n, bins, patches = plt.hist(error_values)
        # plt.xlabel("Values")
        # plt.ylabel("Frequency")
        # plt.title("Histogram")
        # plt.show()
        len_preds = len(aggregate_preds_dict.keys())
        len_errors = len(aggregate_errors_dict.keys())
        len_preds_dict = {"case_count": len_preds, "total": len_preds + len_errors}
        len_preds_dict.update(aggregate_preds_dict)
        len_errors_dict = {"case_count": len_errors, "total": len_errors + len_preds}
        len_errors_dict.update(aggregate_errors_dict)
        aggregate_error_paths = [
            item for sublist in aggregate_errors for item in sublist
        ]
        # with open('/raid/mpsych/deepsight_run/2D_whitelist_results/2D_error_paths.pkl', 'w') as file:
        #     file.write(json.dumps(aggregate_error_paths))  # use `json.loads` to do the reverse
        # file.close()
        with open(
            "/raid/mpsych/whitelists/whitelists_results/2D_whitelist_errors_dicts.pkl",
            "wb",
        ) as file:
            pickle.dump(
                aggregate_errors_dict, file
            )  # use `json.loads` to do the reverse
        with open(
            "/raid/mpsych/whitelists/whitelists_results/2D_whitelist_errors_paths.pkl",
            "wb",
        ) as file_2:
            pickle.dump(
                aggregate_error_paths, file_2
            )  # use `json.loads` to do the reverse
        with open(
            "/raid/mpsych/whitelists/whitelists_results/2D_whitelist_preds_dict.pkl",
            "wb",
        ) as file_3:
            pickle.dump(
                aggregate_preds_dict, file_3
            )  # use `json.loads` to do the reverse
        set_error_paths = set(aggregate_error_paths)
        whitelist = (
            open("/raid/mpsych/deepsight_run/whitelists/whitelist_2d.txt")
            .read()
            .splitlines()
        )
        whitelist = set(whitelist)
        final_whitelist = whitelist - set_error_paths
        final_whitelist = list(final_whitelist)
        textfile = open(
            "/raid/mpsych/deepsight_run/2D_whitelist_results/2D_whitelist_final.txt",
            "w",
        )
        # for elem in final_whitelist:
        #     textfile.write(elem + "\n")
        # textfile.write(json.dumps(final_whitelist))  # use `json.loads` to do the reverse
        textfile.close()
        print(f"total cases: {aggregate_num_lines[0]}")
        return

    @staticmethod
    def get_label_by_sops(sop_to_label_dict):
        for sop in sop_to_label_dict.keys():
            if sop in aggregate_preds_dict.keys():
                aggregate_preds_dict[sop]["label"] = sop_to_label_dict[sop]

    @staticmethod
    def get_label_by_sops_2(df_omama, data):
        for sop in aggregate_preds_dict.keys():
            label = data.get_image(dicom_name=sop, pixels=False).label
            aggregate_preds_dict[sop][label] = label
        return aggregate_preds_dict

    @staticmethod
    def test_label_by_sops(df_omama):
        df_omama = df_omama[
            df_omama["SOPInstanceUID"].isin(aggregate_preds_dict.keys())
        ]
        return df_omama

    @staticmethod
    def cross_matrix(threshold):
        tp = fp = tn = fn = 0
        counter = 0
        cancer_count = non_cancer_count = 0
        for sop in aggregate_preds_dict.keys():
            if aggregate_preds_dict[sop]["label"] != "NonCancer":
                cancer_count += 1
            else:
                non_cancer_count += 1
            counter += 1
            if (
                aggregate_preds_dict[sop]["score"] >= threshold
                and aggregate_preds_dict[sop]["label"] != "NonCancer"
            ):
                tp += 1
            elif (
                aggregate_preds_dict[sop]["score"] >= threshold
                and aggregate_preds_dict[sop]["label"] == "NonCancer"
            ):
                fp += 1
            elif (
                aggregate_preds_dict[sop]["score"] < threshold
                and aggregate_preds_dict[sop]["label"] == "NonCancer"
            ):
                tn += 1
            elif (
                aggregate_preds_dict[sop]["score"] < threshold
                and aggregate_preds_dict[sop]["label"] != "NonCancer"
            ):
                fn += 1
            else:
                print("YOU SHOULD NOT BE HERE!!!")

        print(
            f"tp: {tp}, fp: {fp}, tn: {tn}, fn: {fn}\n"
            f" cancer count: {cancer_count}, noncancer count: {non_cancer_count}\n"
            f"Real tp: {tp/cancer_count}, Real fp: {fp/cancer_count}, Real tn: {tn/non_cancer_count}, Real fn: "
            f"{fn/non_cancer_count}\nCounts: "
            f"tp: {tp}, "
            f"fp: {fp},"
            f" tn: {tn}, fn: {fn} "
        )

    if __name__ == "__main__":
        # paths = []
        # aggregate_preds_dict = {}
        # aggregate_errors_dict = {}
        # num_lines=0
        # aggregate_num_lines = [0]
        # total_counter = [0]
        # total_counter_err = [0]
        list_pred_files("/raid/mpsych/deepsight_run", ".json")
        # sop_to_pred_label = find_matching_rows("/raid/data01/deephealth/labels", aggregate_preds_dict)
        # print(f"len: {len(sop_to_pred_label)}")
        get_label_by_sops()
        cross_matrix(0.5)
