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
_, predictions = runCFRModel(data_array_list=[data],
                             frame_time_ms_list=[frame_time_ms],
                             deltaX_list=[deltaX],
                             deltaY_list=[deltaY])
```
Estimated time for 
running the code on a single echo input: 5.4 seconds.

If no weights are available, a new model with random weights is initialized.
However, checkpoint files can be provided to the runCFRModel function in the
form of a python dictionary {'model_response_variable': 'path_to_checkpoint_file'}.
Furthermore, lists of numpy arrays can be provided to predict on a larger
number of videos:

```
qualified_index_list, predictions = runCFRModel(data_array_list, 
                                                frame_time_ms_list, 
                                                deltaX_list, 
                                                deltaY_list, 
                                                checkpoint_dict,
                                                batch_size=1)
```

The returned list *qualified_index_list* contains the video indices that
of the echo videos that satisfied the minimum requirements, i.e. maximum frame_time
and minimum length.

