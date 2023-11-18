import sys
import os
import _pickle as cPickle

sys.path.insert(0, "/home/ryan.zurrin001/Projects/omama/")
import omama as O


def per_task_execution(task_num, start_num, end_num):
    with open(
        r"/raid/mpsych/kaggle_mammograms/kaggle_caselist.pkl", "rb"
    ) as input_file:
        list_2d = cPickle.load(input_file)
    print("file opened")
    input_file.close()
    print("file closed")
    print(start_num)
    if end_num > len(list_2d):
        list_2d_subset = list_2d[start_num : len(list_2d) + 1]
    else:
        list_2d_subset = list_2d[start_num : end_num + 1]
    task_num = str(task_num)
    O.DeepSight.run(
        list_2d_subset,
        output_dir="/raid/mpsych/deepsight_out/kaggle_processed/",
        timing=True,
        task_num=task_num,
    )
    print(f"task number is: {task_num}")
    return


if __name__ == "__main__":
    TASK_NUM = sys.argv[1]
    TASK_NUM = int(TASK_NUM)
    PER_TASK = 3208
    START_NUM = PER_TASK * (TASK_NUM - 1)
    END_NUM = START_NUM + (PER_TASK - 1)
    per_task_execution(TASK_NUM, START_NUM, END_NUM)
