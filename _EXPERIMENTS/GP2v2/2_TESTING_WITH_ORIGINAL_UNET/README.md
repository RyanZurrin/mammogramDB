# STEP 2: RUNNING GP2 USING ORIGINAL UNET

This folder contains notebooks for running the GP2 framework using the original U-Net architecture, 
which is being developed as part of the Omama-DB and CS410 UMB Software Engineering Project.

## Overview

The notebooks in this folder are designed to test the performance of the GP2 framework's original U-Net classifier, 
a hardcoded Keras U-Net model, on different datasets. The tests involve using normalized and unnormalized data, 
as well as changing the weights of the data distribution, which affects the amount of data in the different training, 
validation, and test sets of A, B, and Z.

## Dependencies

Install the required dependencies by createing a new Anaconda environment using the `GP2.yml` file
in the root directory of the project:
```bash
conda env create -f GP2.yml
```
Activate the environment using:
```bash
conda activate GP2
```

## Notebooks

The notebooks in this folder are responsible for:

1. Loading the preprocessed image and mask data.
2. Setting up the data for the GP2 framework using different weights for data distribution.
3. Running multiple iterations of the classifier and discriminator.
4. Relabeling the data based on the classifier's and discriminator's results.
5. Evaluating and plotting the performance of the classifier and discriminator.

## Usage

To use the notebooks in this folder, please make sure the required dependencies are installed and 
then follow these steps:

1. Load the preprocessed image and mask data from the appropriate dataset directory.
2. Adjust the weights for data distribution, if necessary.
3. Run the notebook cells to perform the tests with the GP2 framework's original U-Net classifier.
4. Review and analyze the performance metrics and plots generated by the notebook.

For other datasets, modify the notebook accordingly to load the specific dataset and ensure that the 
data is preprocessed correctly.

## Contact

For any questions, issues, or suggestions related to this folder or the project in general, please contact the project team members.