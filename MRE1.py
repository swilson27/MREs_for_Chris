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

STRAHLER = () # Strahler indices to prune. Currently no pruning
RESAMPLE = 1000 # Integer value, nodes will be resampled to one per every N (RESAMPLE) units of cable
# Commented out resampling later on, for now

creds_path = os.environ.get("PYMAID_CREDENTIALS_PATH", HERE / "seymour.json")
with open(creds_path) as f:
    creds = json.load(f)
rm = pymaid.CatmaidInstance(**creds)
# Loads stored catmaid credentials

DEFAULT_SEED = 1998



### Functions ###



# this one would not have been needed, were I to be able to get the SWC files working. Unfortunately that's not the case
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



def get_landmarks(landmarks):
    """
    Generates landmark coordinates from downloaded CSV of L and R VNC hemispheres
    Returns:
        numpy ndarray: x, y, z coordinates of L and R VNC hemispheres
    """    
    if landmarks == 'brain':
        df = pd.read_csv(HERE / "brain_landmarks.csv", index_col=False, sep=", ")
    if landmarks == 'vnc':
        df = pd.read_csv(HERE / "VNC_landmarks.csv", index_col=False, sep=",")
    if landmarks == 'cns':
        df = pd.read_csv(HERE / "CNS_landmarks.csv", index_col=False, sep=",")
    
    counts = Counter(df["landmark_name"])
    l_xyz = []
    r_xyz = []
    for name, count in counts.items():
        if count != 2:
            continue
        left, right = df.loc[df["landmark_name"] == name][[" x", " y", " z"]].to_numpy()
        if left[0] < right[0]:
            left, right = right, left

        l_xyz.append(left)
        r_xyz.append(right)

    return np.asarray(l_xyz), np.asarray(r_xyz)


def transform_neuron(tr: navis.transforms.base.BaseTransform, nrn: navis.TreeNeuron):
    """
    Applies selected transformation (tr, type: BT) to neuron (nrn, type: TN)
    Args:
        tr (navis.transforms.base.BaseTransform): _description_
        nrn (navis.TreeNeuron): _description_
    Returns:
        TreeNeuron: Transformed TN/s
    """    
    nrn = nrn.copy(True)
    dims = ["x", "y", "z"]
    nrn.nodes[dims] = tr.xform(nrn.nodes[dims])
    if nrn.connectors is not None:
        nrn.connectors[dims] = tr.xform(nrn.connectors[dims])
    return nrn


def get_transformed_neurons(landmarks):
    """
    Obtains transformed (moving least squares) neurons, taking left pairs and applying L:R mirror flip via 'bhem' landmarks
    Also outputs right pairs, with no transformation applied
    Returns:
        two lists: first will be transformed left, second right; both as class: TN
        stores as pickle, if already generated it will simply load this
    """
    fpath = HERE / "transformed_paired.pickle"

    if not os.path.isfile(fpath):
        neurons_l, neurons_r = get_neurons()

        # create zipped list of paired neuron names
        by_name_l = dict(zip(neurons_l.name, neurons_l))
        by_name_r = dict(zip(neurons_r.name, neurons_r))
        paired_names = list(zip(neurons_l.name, neurons_r.name))

        l_xyz, r_xyz = get_landmarks(landmarks)
        transform = navis.transforms.MovingLeastSquaresTransform(l_xyz, r_xyz)
        left_transform = []
        right_raw = []
        for l_name, r_name in paired_names:
            left_transform.append(transform_neuron(transform, by_name_l[l_name]))
            right_raw.append(by_name_r[r_name])
        
        with open(fpath, "wb") as f:
            pickle.dump((left_transform, right_raw), f, 5)
    else:
        with open(fpath, "rb") as f:
            left_transform, right_raw = pickle.load(f)

    return left_transform, right_raw #, duplicates



def make_dotprop(neuron, prune_strahler = STRAHLER, resample = RESAMPLE):
    """
    First prunes by specified strahler index and resamples neuron to given resolution
    Subsequently applies navis.make_dotprops to this neuron (k = 5, representing appropriate # of nearest neighbours [for tangent vector calculation] for the sparse point clouds of skeletons)
    Args:
        neuron (TN)
        prune_strahler: prune TNs by strahler index, defaults to no pruning
        resample (int): resamples to # of nodes per every N units of cable? Defaults to 1000, but commented out for now
    Returns:
        Dotprops of pruned & resampled neuron
    """    
    nrn = neuron.prune_by_strahler(list(prune_strahler))
    if not len(nrn.nodes):
        logger.warning('pruned %s to nothing', nrn.id)
        return None
    nrn.tags = {}
    # nrn.resample(resample, inplace=True)
    return make_dotprops(nrn, 5)



def make_dps(neurons: navis.TreeNeuron, prune_strahler = STRAHLER, resample = RESAMPLE):
    """
    Applies make_dotprop to list of TreeNeurons, utilising multiprocessing to speed up
    Args:
        neurons (TN)
        prune_strahler
        resample (int): defaults to 1000
    Returns:
        Dotprops of pruned & resampled neurons
    """    

    out = []

    fn = partial(make_dotprop, prune_strahler = prune_strahler, resample = resample)

    with ProcessPoolExecutor(os.cpu_count()) as p:
        out = [
            n
            for n in tqdm(
                p.map(fn, neurons, chunksize=50), "making dotprops", len(neurons)
            )
        ]

    return out



def get_dps(landmarks = 'cns', prune_strahler = STRAHLER, resample = RESAMPLE):
    """
    Obtains left and right pairs from prior functions.
    Transforms left pairs (via 'landmarks' — defaults to whole CNS), makes dot products for both these and right pairs and outputs. Loaded from pickle if already ran
    Args:
        prune_strahler: prune TN by strahler index, defaults to lowest order only (-1, None)
        resample (int): resamples to # of nodes per every N units of cable? Defaults to 1000
    Returns:
        list: dotproducts for l_trans and r
    """    
  
    l_trans, r = get_transformed_neurons(landmarks)
    out = tuple(make_dps(ns, prune_strahler, resample) for ns in [l_trans, r])
    dps_transformed_left = []
    dps_right = []
    for l, r in zip(*out):
        if l is not None and r is not None:
            dps_transformed_left.append(l)
            dps_right.append(r)

    return dps_transformed_left, dps_right 
   


### Run MRE ###



#%%
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    with logging_redirect_tqdm():
        left, right = get_dps('cns') 
        print('dotprop conversion complete')