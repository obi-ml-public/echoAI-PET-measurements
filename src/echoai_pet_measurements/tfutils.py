""""
This file is part of the echoAI-PET-measurements project.
"""

import os
import tensorflow as tf

def use_gpu_devices(gpu_device_string):

    use_device_string = gpu_device_string
    use_device_idx = list(range(len(use_device_string.split(','))))

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = use_device_string

    physical_devices = tf.config.list_physical_devices('GPU')
    device_list = [physical_devices[idx].name.replace('physical_device:', '') for idx in use_device_idx]

    print('AVAILABLE GPUs:')
    print(*physical_devices, sep='\n')
    print('TRAIN DEVICE LIST:')
    print(*device_list, sep='\n')

    try:
        for dev in physical_devices:
            tf.config.experimental.set_memory_growth(dev, True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    return physical_devices, device_list
