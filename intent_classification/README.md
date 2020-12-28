# The structure of the  Project

- The `src` folder contains a main source scripts in the project

```
nbs           nbs folder contains all the notebooks(Note: These notebooks are standalone notebooks meaning no need  to import 				any of the required scripts)
models        The models folder contains all the model scripts
engine.py     Main logic of the problem you want to solve.
utils.py      All utility functions required.
config.py     Configurable parameters or re-usable global variables of this project.
data.py       Utility script for loading data.
train.py      Script for training the models
predict.py    Script for  predicting the classes 
```

- It is important to include `__init__.py` folder otherwise the package will not be able to import
functions / classes.

- In the `__init__.py` folder import stuff as you need. E.g. `from project.src.app import *`

- Use imports from `project` try to avoid relative imports. This is a better practice.

