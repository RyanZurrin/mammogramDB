import os
import subprocess


def test_odm_run_tests():
    # specify the path to the script
    script_path = os.path.join(os.path.dirname(__file__), "..", "odm", "run_tests.py")

    # specify the working directory
    work_dir = os.path.join(os.path.dirname(__file__), "..", "odm")

    # run the script
    result = subprocess.run(
        ["python", script_path], capture_output=True, text=True, cwd=work_dir
    )

    # check if the script ran successfully
    assert (
        result.returncode == 0
    ), f"Script failed with output:\n{result.stdout}\n{result.stderr}"
