# echoAI-PET-measurements
## System requirement
### The codes were tested in the following environment
OS: Ubuntu 18.04

Required software
```
python 3.7.7
tensorflow 2.2.0
tensorflow-addons 0.10.0
```
For a complete list of python packages see `requirements.txt` file
## How to run
Installation of python dependencies
```
pip install -r requirements.txt
```
Installation of the echoAI-PET-measurements module
into the virtual environment (e.g. conda, virtualenv or pipenv)
```
python setup.py develop
```
Now the main function that runs the model can be imported 
```
import numpy as np
from echoai_pet_measurements.runCFRModel import runCFRModel
```
An echocardiography video can be loaded into the memory as a numpy array
```
# Load video data into memory
with open(datafile, 'rb') as f:
    data = np.load(f)

# Provide frame time (in ms) and image scaling factors
frame_time_ms = 30
deltaX = 0.03667
deltaY = 0.03367
```
Inference on the echocardiography video by calling the runCFRModel function
```
predictions = runCFRModel(data=data,
                          frame_time_ms=frame_time_ms,
                          deltaX=deltaX,
                          deltaY=deltaY)
```
Estimated time for 
running the code on a single echo input: 5.4 seconds.