# echoAI-PET-measurements
## System requirement
### The codes were tested in the following environment
OS: Ubuntu 18.04

Required software
```
python 3.7.7
tensorflow 2.2.0
pandas 1.0.3
```
## How to run

```
# Install echoAI-PET-measurements module into a current activated environment (created e.g. by anaconda, virtualenv, pipenv)

python setup.py develop

# Within the environment, start the python interpreter

# Import runCFRModel function
import numpy as np
from echoai_pet_measurements.runCFRModel import runCFRModel

# Load video data into memory
with open(datafile, 'rb') as f:
    data = np.load(f)

# Add frame time (in ms) and scaling factors
deltaX = 0.03667
deltaY = 0.03367
frame_time_ms = 30

# Run inference
predictions = runCFRModel(data=data,
                          frame_time_ms=frame_time_ms,
                          deltaX=deltaX,
                          deltaY=deltaY)
```

## Estimated time for running the code on a single echo input: 5.4 seconds