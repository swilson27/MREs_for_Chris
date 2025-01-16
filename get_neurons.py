# %%
import logging
import os
import pickle
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Iterable, List, Tuple, TypeVar
import json
import matplotlib.pyplot as plt

import navis
import numpy as np
import pandas as pd
import pymaid
from navis.core.core_utils import make_dotprops
from navis.nbl.smat import LookupDistDotBuilder
from pymaid.core import Dotprops
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent

logger = logging.getLogger(__name__)
logger.info("Using navis version %s", navis.__version__)
CACHE_DIR = HERE / "cache"
OUT_DIR = HERE / "output"

STRAHLER = ()
RESAMPLE = 1000

creds_path = os.environ.get("PYMAID_CREDENTIALS_PATH", HERE / "seymour.json")
with open(creds_path) as f:
    creds = json.load(f)
rm = pymaid.CatmaidInstance(**creds)
# Loads stored catmaid credentials

DEFAULT_SEED = 1998

def get_neurons(annotation = False):
    """
    Get a respective CatmaidNeuronList for left and right pairs. Lists are each in arbitrary order.

    if annotation = False: load neurons from downloaded csv (paired_neurons.csv)
        N.B. CatmaidNeuronLists must be unique. If there are any duplicated neurons in your pair list, 
        they will be removed for subsequent separate analysis

    if annotation = True: load neurons from catmaid annotation (subset of paired_neurons), lacks duplicates

    neurons stored as pickle object, will be loaded if in cache folder. N.B. cached neurons must be renamed or deleted if you wish to analyse a different set
    
    Returns:
        {l/r}_neurons: tuple of CmNLs for left and right side respectively

        duplicates: extracted (last) left_skid duplicates (2 in seymour_paired_neurons.csv), for separate analysis (CMNs have to be unique) 
    """    
    pairs = pd.read_csv(HERE / "paired_neurons.csv")
    pairs.drop('region', axis=1, inplace=True)  
    pairs_ascending = pairs.sort_values(by=["leftid"], ascending=True) 

    l_neurons = list(pairs_ascending["leftid"])
    r_neurons = list(pairs_ascending["rightid"])

    l_neurons = pymaid.get_neuron(l_neurons)
    r_neurons = pymaid.get_neuron(r_neurons)

    return l_neurons, r_neurons


# %%
if __name__ == "__main__":
    logging.basicConfig()
    with logging_redirect_tqdm():
        left, right = get_neurons(annotation = False)
        print('neurons obtained')

        navis.write_swc(left, OUT_DIR / 'left_neurons/{neuron.name}.swc')
        print('left neurons written to SWC files')
        navis.write_swc(right, OUT_DIR / 'right_neurons/{neuron.name}.swc')
        print('right neurons written to SWC files')
        
