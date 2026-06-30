import os

PATH_HOME = './'  # path in HPC

PATH_TO_NETWORK_FILE = os.path.join(PATH_HOME, 'datasets')
PATH_TO_RESULTS = os.path.join(PATH_HOME, 'results')

if not os.path.exists(PATH_HOME):
    os.makedirs(PATH_HOME, exist_ok=True)

if not os.path.exists(PATH_TO_RESULTS):
    os.makedirs(PATH_TO_RESULTS, exist_ok=True)