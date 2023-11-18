import sys
import os
import _pickle as cPickle

sys.path.insert(0, "/home/ryan.zurrin001/Projects/omama/")
import omama as O


def per_task_execution(task_num, start_num, end_num):
    with open(
        r"/home/ryan.zurrin001/Projects/omama/omama/deep_sight/toylist" r".pkl", "rb"
    ) as input_file:
        list_3d = cPickle.load(input_file)
        # list_3d = [elem.decode("utf-8") for elem in list_3d]
        if len(list_3d) == 0:
            print("ERROR!!!")
        print(f"len of list is : {len(list_3d)}")

    print("file opened")
    input_file.close()
    print("file closed")
    print(start_num)
    if start_num == len(list_3d):
        print("returning from per_task_execution without classifier execution")
        return
    elif end_num > len(list_3d):
        print(f"hit if statement for {task_num}")
        list_3d_subset = list_3d[start_num : len(list_3d)]
    else:
        list_3d_subset = list_3d[start_num : end_num + 1]
    print(f"****{len(list_3d_subset)}****")
    task_num = str(task_num)
    print(f"List is : {list_3d_subset}")
    O.DeepSight.run(
        list_3d_subset,
        output_dir="/raid/mpsych/deepsight_out" "/3D_whitelist_toy1/",
        timing=True,
        task_num=task_num,
    )
    print(f"task number is: {task_num}")
    return


if __name__ == "__main__":
    TASK_NUM = sys.argv[1]
    TASK_NUM = int(TASK_NUM)
    #     PER_TASK = 3377
    PER_TASK = 3
    START_NUM = PER_TASK * (TASK_NUM - 1)
    END_NUM = START_NUM + (PER_TASK - 1)
    per_task_execution(TASK_NUM, START_NUM, END_NUM)
