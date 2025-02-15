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



# the first 7 functions are identical to those in MRE1. I've marked where the new ones are
# if you are able to

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


T = TypeVar("T")


def split_training_testing(items: List[T], n_partitions=5, seed=DEFAULT_SEED
    ) -> Iterable[Tuple[List[T], List[T]]]:
    """ Splits items into training and testing sets (default n = 5) to generate custom score matrices ahead of cross-validated nblast
    Args:
        items: list of zipped L and R dotprops
        n_partitions (int): # of partitions for cross-validation, defaults to 5.
        seed (int): defaults to specified DEFAULT_SEED for reproducibility
    Yields:
        Iterator[Iterable[Tuple[List[T], List[T]]]]: iteratively yields training and testing sets for n_partitions
    """
    items_arr = np.array(items, dtype=object)
    # Converts items (zipped list of L and R dotprops) into np.ndarray
    partition_idxs = np.arange(len(items_arr)) % n_partitions
    # creates partition indexes as modulus of # of neurons and # of partitions
    rng = np.random.default_rng(seed)
    rng.shuffle(partition_idxs)
    # randomly generates and shuffles partition indexes, based on seed for reproducibility
    for idx in range(n_partitions):
        training = list(items_arr[partition_idxs != idx])
        testing = list(items_arr[partition_idxs == idx])
        yield training, testing
    # iteratively yields training and testing subsets of items for custom score matrices, based on n_partitions
    # each iteration will contain one of the partioned subsets as testing, with remaining partitions (n-1) used for training



def train_nblaster(dp_pairs: List[Tuple[Dotprops, Dotprops]], threads=8
    ) -> navis.nbl.nblast_funcs.NBlaster:
    """ Takes pairs of dotprops, constructs matching_lists (to build score_mat) from pairs, trains nblaster for each
    Args:
        dp_pairs (List[Tuple[Dotprops, Dotprops]]): Dotproducts from the L-R pairs
        threads (int): Defaults to 8.
    Returns:
        blaster : NBLAST algorithm (version 2) trained on input dp_pairs, ready to apply to testing set
    """    
    dps = []
    matching_lists = []
    for pair in dp_pairs:
        matching_pair = []
        for item in pair:
            matching_pair.append(len(dps))
            dps.append(item)
        matching_lists.append(matching_pair)
    # Iterates through pairs in dp_pairs, and both items in pair. Appends indices of matching_pairs and dotprops to separate lists 

    builder = LookupDistDotBuilder(
        dps, matching_lists, use_alpha=True, seed = DEFAULT_SEED
    ).with_bin_counts([21, 10])
    logger.info("Training...")
    score_mat = builder.build(threads)
    # build score matrix across # of threads
    logger.info("Trained")
    df = score_mat.to_dataframe()
    # print(df)
    # df.to_csv(HERE / "smat.csv")
    # Option to output score_mat as CSV

    blaster = navis.nbl.nblast_funcs.NBlaster(True, True, None)
    blaster.score_fn = score_mat

    return blaster



def split_training_testing(items: List[T], n_partitions=5, seed=DEFAULT_SEED
    ) -> Iterable[Tuple[List[T], List[T]]]:
    """ Splits items into training and testing sets (default n = 5) to generate custom score matrices ahead of cross-validated nblast

    Args:
        items: list of zipped L and R neurons
        n_partitions (int): # of partitions for cross-validation, defaults to 5.
        seed (int): defaults to specified DEFAULT_SEED for reproducibility

    Yields:
        Iterator[Iterable[Tuple[List[T], List[T]]]]: iteratively yields training and testing sets for n_partitions
    """
    items_arr = np.array(items, dtype=object)
    # Converts items (zipped list of L and R neurons) into np.ndarray
    partition_idxs = np.arange(len(items_arr)) % n_partitions
    # creates partition indexes as modulus of # of neurons and # of partitions
    rng = np.random.default_rng(seed)
    rng.shuffle(partition_idxs)
    # randomly generates and shuffles partition indexes, based on seed for reproducibility
    for idx in range(n_partitions):
        training = list(items_arr[partition_idxs != idx])
        testing = list(items_arr[partition_idxs == idx])
        yield training, testing
    # iteratively yields training and testing subsets of items for custom score matrices, based on n_partitions
    # each iteration will contain one of the partioned subsets as testing, with remaining partitions (n-1) used for training




def cross_validation(dp_pairs: List[Tuple[Dotprops, Dotprops]], n_partitions=5, seed=DEFAULT_SEED):
    """ Takes zipped list of left and right dotprops, partitions data (training/testing) for cross-validated nblaster training and applies these functions to testing sets
        Scores are calculated as mean of transformed_left-to-right and right-to-transformed_left comparisons
    Args:
        dp_pairs (List[Tuple[Dotprops, Dotprops]]): _description_
        n_partitions (int, optional): _description_. Defaults to 5.
        seed (_type_, optional): _description_. Defaults to DEFAULT_SEED.
    Returns:
        _type_: _description_
    """
    headers = [
        "left_skid_transformed",
        "right_skid_raw",
        "left_name",
        "right_name",
        "mean_normalized_nblast",
        "partition"
        ]
    dtypes = [int, str, int, str, float, int]
    rows = []
    for partition, (training, testing) in tqdm(
        enumerate(split_training_testing(dp_pairs, n_partitions, seed)),
        "cross validation partitions",
        n_partitions,
    ):
        nblaster = train_nblaster(training, threads=None)
        # creates nblast algorithm from training set
        for left, right in tqdm(testing, f"testing {partition}"):
            l_idx = nblaster.append(left)
            r_idx = nblaster.append(right)
            result = nblaster.single_query_target(l_idx, r_idx, scores="mean")
            rows.append([left.id, right.id, left.name, right.name, result, partition])
        # iteratively applies this across testing set, result output as mean of l-to-r and r-to-l scores

    df = pd.DataFrame(rows, columns=headers)
    for header, dtype in zip(headers, dtypes):
        df[header] = df[header].astype(dtype)
    
    df.sort_values(headers[0], axis=0, inplace=True)
    df.sort_values(headers[3], axis=0, inplace=True)
    
    return df



### Run MRE ###



if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    with logging_redirect_tqdm():
        left, right = get_dps('cns')      
        paired = list(zip(left, right))
        df = cross_validation(paired)
        print(df)

        df.to_csv(OUT_DIR / f"nblast_analysis_results.csv", index=False)



