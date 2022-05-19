from msilib.schema import Property
from turtle import color
import matplotlib
import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
from typing import Union



class DepressedDataFrame(pd.DataFrame):
    """This class provides all Function of pandas.DataFrame, 
    but adds more functions and enhances functions"""

    def __init__(self, csv_file_path: str = None, **kwargs) -> None:
        """Initiates the DataFrame. 
            
            # Parameter
            - csv_file_path: path to csv which contains the data
            - kwargs: standard parameters for DataFrame, if DataFrame
            should not be created from file.
        """
        
        if csv_file_path:
            if os.path.isfile(csv_file_path) and csv_file_path.endswith(".csv"):
                super().__init__(pd.read_csv(csv_file_path))
                return
            else:
                warnings.warn("Given file is not available. DataFrame is created with keyword arguments",
                    UserWarning)
        pd.DataFrame().info
        super().__init__(**kwargs)
  

    def info(self, max_columns: int=None, max_unique_values: int=5, old_version: bool=True, **kwargs) -> None:
        """ Prints information about the DataFrame.
            # Parameter
            - max_columns: How many columns should be analysed?
            - max_unique_values: How many unique values should be displayed?
            - old_version: if true the standard Dataframe info() is used with kwargs
            - kwargs: Keyword arguments for using old version of info()
        """
        if old_version:
            super().info(**kwargs)
        else:
            # Print information about columns, rows and memory usage
            pd.io.formats.info.DataFrameInfo(self, memory_usage=True).render(buf=None, max_cols=None, verbose=None, show_counts=True)
            if not max_columns:
                max_columns = len(self.index)

            # Set max column length, so that we can pretty print the result
            max_col_length = max([len(c) for c in self.columns[:max_columns]]) + 4
            print('{:<{max_col_length}s}{:<14s}{:<8s}{}'.format(
                "Column Name", "Unique values", "Type", "Column values", max_col_length=max_col_length))
            for col in self.columns[:max_columns]:
                # calculate unique values and the percentage of their occurance
                s = self[col].unique()

                unique_values = [str(val) + " (" +  
                    str(round(len(self[self[col]==val].index) / len(self[col].index)*100, 2)) + "%)"
                    for val in 
                        self.copy().groupby([col]).count().sort_values(
                            ascending=False, 
                            by=list(self.columns)[list(self.columns).index(col)-1]).index[:max_unique_values]
                    ]
                if len(self[col].unique())>5:
                    unique_values.append(" ...")

                print('{:<{max_col_length}s}{:<13d}{:<8s}{}'.format(
                    col, 
                    len(self[col].unique()), 
                    str(self[col].dtype), 
                    str(unique_values).replace("[", "").replace("]", "").replace("'", ""),
                    max_col_length=max_col_length))

    def plot_binary_columns(self, 
                            columns: list, 
                            select_true: bool=True, 
                            title: str="Percent of depressed", 
                            remove_str_in_plot_of_labels: str="", 
                            suptitle=None):
        """ Plots a binary (0/1 or Yes/No) column to a grouped bar plot. 
            The value "Missing" is removed from the df to show the distribution for all rows with available data.
            The plot always shows depressed and not depressed group bars and the percentage of persons who answered
            with yes/1 or with no/0 if parameter select_true is False. 

            The bar could be read like that: 
            of persons who are depressed/not depressed also have column_name. (if select true) 

            # Parameter
            - columns: list of column names as str. Each column is plotted as one row (with two groups) in the bar chart.
            - select_true: if true 1 or Yes is displayed in percentage. If False, 0 or No is displayed.
            - title: Title which will be displayed above chart (font size 16)
            - remove_str_in_plot_of_labels: A string that will be removed from the label from column names
            - suptitle: Suptitle below the chart (font size 12)

            # Returns
            - matplotlib.pyplot: so you can directly enter after line .draw() or .show() to show chart

            # Example:
            >>> df = DepressedDataFrame(data={"col1": [1, 0, 1, 1], "depression": ["Depressed", "Not Depressed", ...]})
            >>> df.plot_binary_columns(columns=["col1"]).draw()

            You could as well use pyplot in you source code to change the plot.
            >>> import matplotlib.pyplot as plt
            >>> df = DepressedDataFrame(data={"col1": [1, 0, 1, 1], "depression": ["Depressed", "Not Depressed", ...]})
            >>> df.plot_binary_columns(columns=["col1"])
            >>> plt.show()
        """
        not_depressed = self[self["depression"]=="Not Depressed"]
        depressed = self[self["depression"]=="Depressed"]

        bar_labels = []
        dep_percent = []
        not_dep_percent = []

        for col in columns:
            # identify the yes label. It's either 1/0 or yes/no/missing
            if select_true:
                select_label = 1
                if 1 not in self[col].values and "Yes" in self[col].values:
                    select_label = "Yes"
            else:
                select_label = 0
                if 0 not in self[col].values and "No" in self[col].values:
                    select_label = "No"
            
            bar_labels.append(col.replace(remove_str_in_plot_of_labels, ""))
            # For yes/no label the Missing value must be removed
            dep_percent.append(round(len(depressed[depressed[col]==select_label]) / len(depressed[depressed[col]!="Missing"].index) * 100, 2))
            not_dep_percent.append(round(len(not_depressed[not_depressed[col]==select_label]) / len(not_depressed[not_depressed[col]!="Missing"].index) * 100, 2))

        # Fill plot
        x = np.arange(len(bar_labels)) 
        width = 0.35
        fig, ax = plt.subplots()
        fig.set_size_inches(11.5, 0.3*len(bar_labels)+5)

        rects1 = ax.barh(x - width/2, dep_percent, width, label='Depressed')
        rects2 = ax.barh(x + width/2, not_dep_percent, width, label='Not Depressed')

        # Add text for labels, title and custom x-axis tick labels, etc.
        ax.set_xlabel('Percent')
        ax.set_title(title)
        ax.set_yticks(x, bar_labels)
        ax.legend()

        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)

        fig.tight_layout()

        if suptitle:
            fig.suptitle(suptitle, fontsize=12, y=0, x=0.11+len(suptitle)*0.002)

        return plt

    def plot_categorical_columns(self, 
                columns: Union[str, list], 
                value_filter: Union[list, dict]=None, 
                title: str="Percent of depressed", 
                remove_str_in_plot_of_labels: Union[str,list]=[], 
                keep_column_name: bool=True,
                suptitle=None):
        """ Plots one ore more categorial columns. 
            Each element in the list will be displayed as it's own grouped bar (depressed/not depressed). 

            # Parameter:
            - columns: one column (str) or more columns (list) of the df.
            - value_filter: what values should not be displayed in bar chart? 
                - if list: The values in the list will be removed from all columns
                - if dict: key must be the column name and value is a list of values that should be removed
            - title: Title which will be displayed above chart (font size 16)
            - remove_str_in_plot_of_labels: A string that will be removed from the label from column names
                - if str: the str is removed from all column names
                - if list: list must have same len as columns and based on index the str in list 
                    will be replaced on the index in the columns list
            - keep_column_name: if False, the column name is removed from the label and only the attribute name remains
            - suptitle: Suptitle below the chart (font size 12) 

            # Returns
            - matplotlib.pyplot: so you can directly enter after line .draw() or .show() to show chart

            # Example:
            >>> df = DepressedDataFrame(data={"col1": ["a", "b", "b", "x"], "depression": ["Depressed", "Not Depressed", ...]})
            >>> df.plot_categorical_columns(columns="col1").draw()

            You could as well use pyplot in you source code to change the plot.
            >>> import matplotlib.pyplot as plt
            >>> df = DepressedDataFrame(data={"col1": ["a", "b", "b", "x"], "depression": ["Depressed", "Not Depressed", ...]})
            >>> df.plot_categorical_columns(columns="col1")
            >>> plt.show()
        """
        # Check parameters
        if value_filter:
            if type(columns) == str and type(value_filter) != list:
                if columns in value_filter:
                    value_filter = value_filter[columns]
                else:
                    raise TypeError("When columns is type str value_filter must be list")
            elif type(columns) == list and type(value_filter) != dict:
                value_filter = {col: value_filter for col in columns}

        if type(remove_str_in_plot_of_labels) == str:
            remove_str_in_plot_of_labels = [remove_str_in_plot_of_labels]

        if type(columns) == str:
            value_filter = {columns: value_filter}
            columns = [columns]

        # Separate in two dfs and apply value filter if exits
        not_depressed_main = self[self["depression"]=="Not Depressed"]
        depressed_main = self[self["depression"]=="Depressed"]

        bar_labels = []
        dep_percent = []
        not_dep_percent = []
        
        for col in columns:
            not_depressed = not_depressed_main.copy()
            depressed = depressed_main.copy()
            # apply filter 
            if value_filter:
                if col in value_filter:
                    for v_filter in value_filter[col]:
                        not_depressed = not_depressed[not_depressed[col]!=v_filter]
                        depressed = depressed[depressed[col]!=v_filter]
                    # Set array filter dependent if value filter is set
                    compare_array = (np.setdiff1d(np.array(self[col].unique()),
                                        np.array(value_filter[col])) 
                                    if value_filter 
                                    else self[col].unique())
                else:
                    compare_array = np.array(self[col].unique())

            for val in compare_array:
                if keep_column_name:
                    bar_label = f"{col}_{val}"
                else:
                    bar_label = val

                for rml in  remove_str_in_plot_of_labels:
                    bar_label = bar_label.replace(rml, "")
                bar_labels.append(bar_label)
                # For yes label the Missing value must be removed. If there are no values add 0
                try:
                    dep_percent.append(round(len(depressed[depressed[col]==val]) / len(depressed.index) * 100, 2))
                except ZeroDivisionError:
                    dep_percent.append(0.)
                try:
                    not_dep_percent.append(round(len(not_depressed[not_depressed[col]==val]) / len(not_depressed.index) * 100, 2))
                except ZeroDivisionError:
                    not_dep_percent.append(0.)

        # Fill plot
        x = np.arange(len(bar_labels)) 
        width = 0.35
        fig, ax = plt.subplots()
        fig.set_size_inches(11.5, 0.3*len(bar_labels)+5)

        rects1 = ax.barh(x - width/2, dep_percent, width, label='Depressed')
        rects2 = ax.barh(x + width/2, not_dep_percent, width, label='Not Depressed')

        # Add text for labels, title and custom x-axis tick labels, etc.
        ax.set_xlabel('Percent')
        ax.set_title(title)
        ax.set_yticks(x, bar_labels)
        ax.legend()

        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)

        fig.tight_layout()

        if suptitle:
            fig.suptitle(suptitle, fontsize=12, y=0, x=0.11+len(suptitle)*0.002)

        return plt


    def depression_hist(self, 
                        column: str, 
                        title: Union[str, list]="", 
                        suptitle: str=None, 
                        axes_in_rows: bool=False) -> plt:
        """ Plots three histograms for a numerical column. 
            1st will be histogram of all data, 
            2nd histogram only for depressed,
            3rd will be only for not depressed. 

            # Parameter
            - column: column name of df that should be plot.
            - title: Title which will be displayed above chart (font size 16)
            - suptitle: Suptitle below the chart (font size 12)
            - axes_as_rows: if true each histogram will be one row (3 rows in total), 
                if false the histograms will be in columns (1 row) 

                        # Returns
            - matplotlib.pyplot: so you can directly enter after line .draw() or .show() to show chart

            # Example:
            >>> df = DepressedDataFrame(data={"col1": [1, 5, 3, 1], "depression": ["Depressed", "Not Depressed", ...]})
            >>> df.depression_hist(columns="col1").draw()

            You could as well use pyplot in you source code to change the plot.
            >>> import matplotlib.pyplot as plt
            >>> df = DepressedDataFrame(data={"col1": [1, 5, 3, 1], "depression": ["Depressed", "Not Depressed", ...]})
            >>> df.depression_hist(columns="col1", axes_in_rows=True) # three rows :O
            >>> plt.show()
        """
        # Plot in rows or in columns?
        if axes_in_rows:
            fig, ax = plt.subplots(3, 1, figsize=(30,10))
        else:
            fig, ax = plt.subplots(1, 3, figsize=(30,10))
        # Transform title or is it already a list of titles?
        if type(title) == str:
            title = [title + " overall",
                    title + " of depressed",
                    title + "  of not depressed"]
        elif type(title) != list and len(title) != 3:
            raise Exception("Title must be string or a list of 3 titles. \n\t 1. Title for overall\n\t 2. Title for depressed\n\t 3. Title for not depressed")

        # build bins
        start = self[column].min()-0.5
        end = self[column].max()+1
        step = int(end//50)+1
        bins = np.arange(-0.5*step, end, step)# range(start, end, step)

        # Plot three histograms in the axes
        self.hist(column, ax=ax[0], bins=bins, color="black")
        ax[0].set_title(title[0])
        self[self["depression"]=="Depressed"].hist(column, ax=ax[1], bins=bins)
        ax[1].set_title(title[1])
        self[self["depression"]=="Not Depressed"].hist(column, ax=ax[2], bins=bins, color="tab:orange")
        ax[2].set_title(title[2])
        if suptitle:
            fig.suptitle(suptitle, fontsize=12, y=0, x=0.11+len(suptitle)*0.002)

        return plt

    def depression_box_plot(self, 
                            column: str, 
                            title: str=None, 
                            suptitle: str=None) -> plt:
        """ Plots three boxes in boxplot for a numerical column. 
            1st will be boxplot of all data, 
            2nd boxplot only for depressed,
            3rd will be only for not depressed. 
        
            # Parameter
            - column: column name of df that should be plot.
            - title: Title which will be displayed above chart (font size 16)
            - suptitle: Suptitle below the chart (font size 12)

            # Returns
            - matplotlib.pyplot: so you can directly enter in line .draw(), .show(), ...

            # Example:
            >>> df = DepressedDataFrame(data={"col1": [1, 5, 3, 1], "depression": ["Depressed", "Not Depressed", ...]})
            >>> df.depression_box_plot(columns="col1").draw()

            You could as well use pyplot in you source code to change the plot.
            >>> import matplotlib.pyplot as plt
            >>> df = DepressedDataFrame(data={"col1": [1, 5, 3, 1], "depression": ["Depressed", "Not Depressed", ...]})
            >>> df.depression_hist(columns="col1")
            >>> plt.show()

        """
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 10)
    
        ax.boxplot(self[column], positions=[1],
                    notch=True, patch_artist=True,
                    boxprops={"facecolor": "black", "color": "black"},
                    medianprops={"color": "white"})
        
        ax.boxplot(self[self["depression"]=="Not Depressed"][column], positions=[2],
                    notch=True, patch_artist=True,
                    boxprops={"facecolor": "tab:orange", "color": "tab:orange"},
                    medianprops={"color": "white"})
        ax.boxplot(self[self["depression"]!="Not Depressed"][column], positions=[3],
                    notch=True, patch_artist=True,
                    boxprops={"facecolor": "tab:blue", "color": "tab:blue"},
                    medianprops={"color": "white"})

        ax.set_xticks([1,2,3], 
                      [column + " overall", column + " not depressed", column + " depressed"])

        if suptitle:
            fig.suptitle(suptitle, fontsize=12, y=0, x=0.11+len(suptitle)*0.002)
        if title:
            plt.title(title, fontsize=16)

        return plt

