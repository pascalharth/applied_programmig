
import itertools
import pandas as pd
import os
import pickle


from sklearn.model_selection import GridSearchCV
from .classifier import DepressedClassifier

def train_grid_search(dclf: DepressedClassifier, 
                    hyper_params: dict, 
                    k=4, 
                    restore_path: str=None) -> GridSearchCV:
    """Restores a GridSearchCV with DepressedClassifiers. 
        If restore_path or if file is not found is None GridSearchCV will be executed again.
        Object is saved to restore_path if training was needed. 

        # Parameter
        - dclf: DepressedClassifier object with fixed parameters set
        - hyper_params: hyperparameter that should be altered in dclf
        - k: number of folds that GridSearchCV should use
        - restore_path: path from where the model should be restored. 
          If path doesn't exist a new GridSearchCV is calculated and saved under this path
    """
    # is model saved and can be restored?
    if os.path.isfile(restore_path) and restore_path:
        with open(restore_path, "rb") as f:
            clf = pickle.load(f)
    else:
        # we must calculate again...
        clf = GridSearchCV(dclf, hyper_params, cv=k, n_jobs=1)
        clf.fit(X=dclf.train_vec, y=dclf.train_labels)

        # save file
        if restore_path:
            with open(restore_path, "wb") as f:
                pickle.dump(clf, f)

    return clf
        


def get_all_combinations(in_list: list, 
                        basic_combinations: list=[], 
                        flattern: bool=False,
                        keep_empty: bool=True) -> list:
    """ Builds a from a list a list of lists which contains 
        all possible combinatiosn with the values in the input list.
        Uses itertools.combinations

        # Parameter
        - in_list: input list one or two dimension
        - basic_combination: Values that should be in every combination list
        - flattern: should list be flattered to 1d. 
            Each list in 2nd dimension is considered as a fixed combination which cannot be split.
        - keep_empty: if true [] is returned as one possible combination

        # Returns
        - out_list: list (2D) which contains lists with all combinations
    """
    if flattern:
        basic_combinations = [basic_combinations]

    out_list = [] # create empty output if empty list was passed
    for r in range(len(in_list) + 1):
        out_list += [list(i) + basic_combinations for i in itertools.combinations(in_list, r)]
    
    if flattern:
        out_list = [list(itertools.chain(*l)) for l in out_list]

    if not keep_empty:
        out_list.remove([])

    return out_list
