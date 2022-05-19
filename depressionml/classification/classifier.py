from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import warnings
import inspect
from collections import defaultdict
from typing import Union, Any

class DepressedClassifier(BaseEstimator):
    """
    Class for preparing the vectors for the training. 
    If fitted it can be used for transforming values for prediction or transform values from prediciton.
    This class can be used in GridSearchCV and mutate Data. The Class will do the test/train split as well. 

    # Usage:
    >>> from sklearn.svm import LinearSVC
    >>> from sklearn.model_selection import GridSearchCV
    >>> dclf = DepressedClassifier(LinearSVC, data=df, 
    ...     scaler_columns=["current_cigarettes_per_day"], 
    ...     keep_columns=["asthma"]
    ...     remove_values=["Missing"])
    >>> hyper_params = {"C": [0.25, 0.5, 0.75, 1.], binary_columns=[["gender", "marital_status"], "gender"]}
    >>> clf = GridSearchCV(dclf, hyper_params, cv = k, n_jobs=1)
    >>> clf.fit(X=clf.train_vec, y=clf.train_labels)
    >>> "Magic will happen"
    """
    binary_vecorizer_ = None
    scaler_ = None
    train_vec = None
    test_vec = None
    train_labels = None
    test_labels = None
    label_encoder_ = None
    test_predictions_ = None

    def __init__(self,
                classifier_object: RandomForestClassifier, 
                data: pd.DataFrame, 
                binarizer_columns: Union[str, list]=[], 
                scaler_columns: Union[str, list]=[],
                keep_columns:   Union[str, list]=[],
                remove_values:  Union[str, list]=[],
                label_column: str="depression", 
                test_quota: float=0.2, 
                classifier_kwargs: dict={}) -> None:
        """ Creates BeerVectorizer and transforms the given data to vectors and will set them as attributes. \n
        # Parameter
        - classifier_object: An classifier object that provides a fit and predict method
        - data: DataFrame from which the vector should be build
        - binarizer_column: Column which will be preprocessed with MultiLabelBinarizer, if more than one column they will be appended
        - scaler_columns: Columns which will be preprocessed with MinMaxScaler
        - keep_columns: columns that will taken as they are as np.array for in the vector
        - remove_values: Values for which the row will be removed if they occur in it
        - label_column: column that should be predicted
        - test_quota: How much of data should be used for test the model. (percentage as float 1. = 100%)
        - classifier_kwargs: since every classifier delivers different values this argument takes the other key word arguments 
        and matches them against those which are available in the choosen classifier object and uses the matching parameters for object creation
        """
        
        # Save all arguments as class properties (needed for Estimators)
        args, varargs, varkw, values = inspect.getargvalues(inspect.currentframe())
        
        values.pop("self") # Remove self, since we cannot set self

        for arg, val in values.items():
            setattr(self, arg, val)

        self.__update_data__()


    def __update_data__(self):
        """This function updates the data when the class is initiated or 
            the `set_params` function is called.
        """
        
        # Get paramets from the current attributes that are valid for the classifier
        # get params if they are in GridSearch
        classifier_params = {param: vars(self)[param] 
                            for param in vars(self) 
                            if param in self.classifier_object.__init__.__code__.co_varnames}
        # Update with dict from parameters
        classifier_params.update(self.classifier_kwargs)

        self.clf = self.classifier_object(**classifier_params)

        # Transform binarizer and scaler columns to list if its just one
        if type(self.binarizer_columns) == str:
            self.binarizer_columns = [self.binarizer_columns]
        
        if type(self.scaler_columns) == str:
            self.scaler_columns = [self.scaler_columns]

        if type(self.keep_columns) == str:
            self.keep_columns = [self.keep_columns]

        if type(self.remove_values) == str:
            self.remove_values = [self.remove_values]

        # copy data:
        df = self.data.copy()

        # filter values for each column, so we don't filter wrong columns
        if self.remove_values:
            old_len = len(df)
            for col in self.binarizer_columns + self.scaler_columns + self.keep_columns:
                df = df[~df[col].isin(self.remove_values)]

            if old_len > len(df):
                warnings.warn(f"{old_len-len(df)} rows removed", UserWarning)
        
        # Too much filtered, stop
        if len(df)<=0 or len(df[self.label_column].unique())<2:
            self.train_vec = np.array([[0], [0], [0], [0]])
            self.test_vec = np.array([[0], [0], [0], [0]])
            self.train_labels = np.array([1, 1, 0, 0])
            self.test_labels = np.array([0, 0, 1, 1])
            return

        # Copy data without reseted index
        df = df.reset_index(drop=True)

        # Do Binarizer preprocessing
        self.binarizer_list = []
        binary_vector = np.array([[] for i in df.index])
        first_bin = True
        for col in self.binarizer_columns:
            self.binarizer_list.append(MultiLabelBinarizer())
            df[col] = df[col].astype(str).str.split("TRANSFORM_TO_LIST")

            binary_vector = np.concatenate((binary_vector, self.binarizer_list[-1].fit_transform(df[col])), axis=1)
                

        # Do scaler preprocessing
        self.scaler_ = MinMaxScaler()
        if len(self.scaler_columns) > 0:
            scaler_vec = self.scaler_.fit_transform(df[self.scaler_columns])
        else:
            # No column passed, build empty for concat
            scaler_vec = np.array([[] for i in range(0, len(df.index))])

        # Build train vector
        self.train_vec = np.concatenate((binary_vector, scaler_vec, df[self.keep_columns].to_numpy()), axis=1)

        # Encode labels
        self.label_encoder_ = LabelEncoder()
        label_vec = self.label_encoder_.fit_transform(df[self.label_column])

        self.train_vec, self.test_vec, self.train_labels, self.test_labels = train_test_split(self.train_vec, label_vec, test_size=self.test_quota, random_state=7)

        # Too much filtered, not enough labels
        if len(np.unique(self.train_labels)) < 2 or  len(np.unique(self.test_labels)) < 2:
            self.train_vec = np.array([[0], [0], [0], [0]])
            self.test_vec = np.array([[0], [0], [0], [0]])
            self.train_labels = np.array([0, 0, 1, 1])
            self.test_labels = np.array([0, 0, 1, 1])
            return


    def plot(self, title: str="", suptitle: str="") -> plt:
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 10)
        cm = confusion_matrix(self.test_labels, self.test_predictions_, labels=self.clf.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.label_encoder_.classes_).plot(ax=ax)
        #fig.tight_layout()

        if title:
            ax.set_title(title, fontsize=16)

        if suptitle:
            fig.suptitle(suptitle, fontsize=12, y=0.1, x=0.15)
        
        return plt


    # Functions for classification #######################################################
    def predict(self, X) -> Any:
        """Predict an array of features"""
        try:
            # Return predict from the classifier object
            return self.clf.predict(X)
        except:
            # When data is updated in parallelism we need to run fit again
            try:
                if (self.train_vec.shape == np.array([[0], [0], [0], [0]]).shape):
                    warnings.warn("Predicting not possible. Not enough data.", UserWarning)
                    return np.array([2 for i in self.test_labels])
            except:
                pass

            self.clf.fit(self.train_vec, self.train_labels)
            return self.clf.predict(X)


    def fit(self, X, y, **kwargs) -> Any:
        """Fits the passed"""
        if (self.train_vec.shape == np.array([[0], [0], [0], [0]]).shape):
            warnings.warn("Fitting not possible. Not enough data.", UserWarning)
            return self
        self.clf.fit(X, y)
        return self


    def inverse_transform_label(self, label: Any) -> Any:
        """ Inverse transform the label by means of the label encoder property
        # Parameter:
        - label: Label which gets predicted.

        # Returns:
        Human readable label
        """
        return self.label_encoder_.inverse_transform(label)

    def predict_one(self, **kwargs) -> Any:
        """ Predicts the class of the given input. \n
            Input are all the features as keyword arguments, which were given in input as binarizer_columns and label_columns
            # Parameter:
            - kwargs: each column which was named in initiation as scaler and binary column must be added.
        """
        vec = self.transform_one(**kwargs)
        
        label = self.clf.predict(vec)
        return self.inverse_transform_label(label)[0]


    def transform_one(self, **kwargs) -> np.ndarray:
        """ Transforms single values to a input vector for predict
            # Parameter:
            - kwargs: each column which was named in initiation as scaler and binary column must be added.
        """
        # Get keyword arguments from input and map them to class properties
        scaler_input = []
        for col in self.scaler_columns:
            if type(kwargs[col]) == list:
                scaler_input += kwargs[col]
            else:
                scaler_input.append(kwargs[col])

        binary_input = []
        for i, col in enumerate(self.binarizer_columns):
            if len(binary_input) == 0:
                binary_input = self.binarizer_list[i].transform(np.array([kwargs[col]]))
            else:
                binary_input = np.concatenate((binary_input, self.binarizer_list[i].transform(np.array([[kwargs[col]]]))), axis=1)

        keep_input = []
        for col in self.keep_columns:
            if type(kwargs[col]) == list:
                keep_input += kwargs[col]
            else:
                keep_input.append(kwargs[col])

        # Is scaler column available for transform?        
        if len(scaler_input) > 0:
            scaler_vec = self.scaler_.transform(np.array([scaler_input]))
        else:
            scaler_vec = np.array([[]])
        

        return np.concatenate((binary_input, scaler_vec, np.array([keep_input])), axis=1)


    @property
    def accuracy(self):
        """Property that returns the accuracy"""
        if (self.train_vec.shape == np.array([[0], [0], [0], [0]]).shape):
            return np.nan

        if str(type(self.test_predictions_)) == "<class 'NoneType'>":
            self.test_predictions_ = self.predict(self.test_vec)

        return accuracy_score(self.test_labels, self.test_predictions_)
    
    # Functions for BaseEstimator ########################################################

    def score(self, *args):
        """Return score of estimator"""
        return self.accuracy

    def get_params(self, deep=True):
        """ Class which is needed as estimator. \n
            # Parameter:
            - deep: Attribute of BaseEstimator, not used in current version
        """
        
        out = {}
        
        for key in self._get_param_names():
            value = getattr(self, key)
            out[key] = value

        return out


    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator. Same as in BaseEstimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "scikit-learn estimators should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature)
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])
    

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        This function does not work on nested objects.  
        GridSearchCV is using this function to update parameters.
        The function uses the internal `__update_data__` function.

        # Parameters:
        **params : dict
            Estimator parameters.

        # Returns
        self : estimator instance
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self

        for arg, val in params.items():
            setattr(self, arg, val)
        self.__update_data__()

        return self


    # standard object functions ##########################################################
    def __lt__(self, other: object) -> bool:
        """Strandard method for comparing items (less than)"""
        try:
            return self.accuracy < other.accuracy
        except Exception:
            raise Exception("Cannot compare unfitted classifiers.")

    def __gt__(self, other: object) -> bool:
        """Strandard method for comparing items (greater than)"""
        try:
            return self.accuracy > other.accuracy
        except Exception:
            raise Exception("Cannot compare unfitted classifiers.")
    
    def __str__(self) -> str:
        """Standard function for stringify object"""
        start_string = "<class DepressedClassifier>\n"
        if str(type(self.test_predictions_)) == "<class 'NoneType'>":
            try:
                self.test_predictions_ = self.predict(self.test_vec)
            except:
                return start_string
        return  start_string + classification_report(self.test_labels, self.test_predictions_)