import os, sys


class Classifier:
    def __init__(self, verbose=True, workingdir="/tmp"):
        self.verbose = verbose
        self.workingdir = workingdir

        pass

    def build(self):
        pass

    def train(self, X_train, y_train, X_val, y_val):
        pass
