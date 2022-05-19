# applied_programmig
Repository for applied programmig 

# Set up
To excecute the code without errors you will need **python version 10.0** or higher.  

Please install the requirements before opening the Jupiter Notebook. Open console navigate to this repository and activate your virtual environment (if needed or create one):  
```BASH
python -m venv venv 
# activation depends on os and terminal. e.g. cmd:
venv/Scripts/activate.bat
```
Then install all requirements from the requirements file:  
> pip install -r requirements.txt

# Execution
Open `nootbook.ipynb` in you preferred IDE or in the browser Version. **Don't move the notebook outside of this folder.**  
Model calculation in this notebook could take a while, if there are models to restore saved as pickle file they will be used.  
All models to be restored can be found in `./models/`. If you want to do a recalculation delete them or change the parameter `restore_path` in the notebook.ipynb
