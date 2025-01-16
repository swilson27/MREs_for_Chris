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



# TO DO #
# either use an approach like this first function, where duplicate pairs are filtered out as a separate dataframe and then an identical 'get_neurons' function is applied to both the duplicate and original dataframe (minus those duplicates)
# or,
# use the 'get_neurons' function to output separate CatmaidNeuronLists for the left and right neurons of any duplicate pairs

def check_duplicates():
    '''
    CatmaidNeuronLists must be unique. If there are any duplicated neurons in your list of skeleton IDs (skids), they will be extracted out as a separate CSV for parallel downstream analysis
    
    !!! Currently this function would need to be recursively applied, if there are any cases of skids duplicated >2 times (not in Seymour, however)
    Args:
        csv: CSV file, which contains the skids of each left/right paired neurons as rows
            First column should be the left neuron, and the second the right
    
    Returns:
        pairs_unique: the same dataframe, with any duplicated pairs/rows removed (after the first occurence of that duplicated skid)

        pairs_dups: if any duplicates are detected, these rows (e.g. those pairs after the first occurence of a duplicate) are extracted
    '''

    pairs = pd.read_csv(HERE / "paired_neurons.csv")
    pairs.drop('region', axis=1, inplace=True)  
    pairs_ascending = pairs.sort_values(by=["leftid"], ascending=True)

    # CatmaidNeuron objects must be unique, and thus we must check for any duplicated (dup) skids

    pairs_dups = []
    # make empty list for duplicates, which will be overwritten with a dataframe if these exist

    has_duplicate = pairs_ascending.duplicated(subset=['leftid'])
    dup_left = pairs_ascending[has_duplicate]
    
    has_duplicate = pairs_ascending.duplicated(subset=['rightid'])
    dup_right = pairs_ascending[has_duplicate]
    
    # in Seymour, left side has 2 neurons (4985759 and 8700125) with duplicated pairs owing to developmental phenomenon
    # the second occurence of these pairs are filtered out (as CmNs must be unique), with parallel analysis applied and scores appended at the end

    if len(dup_left) or len(dup_right) > 0:

        print("DUPLICATES DETECTED")

        print("duplicates on left:")
        print(dup_left)
        print("duplicates on right:")
        print(dup_right)

        pairs_dups = pairs_ascending.duplicated(subset=['leftid']) + pairs_ascending.duplicated(subset=['rightid'])
    
    else:

        pairs_unique = pairs_ascending.drop_duplicates(subset='leftid')
        pairs_unique = pairs_unique.drop_duplicates(subset='rightid')

    return pairs_unique, pairs_dups


def get_neurons(csv, annotation = False):
    '''
    Get a respective CatmaidNeuronList for left and right pairs.
    Args:
        csv: CSV file, which contains the skids of each left/right paired neurons as rows
            First column should be the left neuron, and the second the right
        
    
    CatmaidNeuronLists must be unique. If there are any duplicated neurons in your pair list, they will be extracted for parallel downstream analysis

    if annotation = True: load neurons from catmaid annotation (subset of paired_neurons), lacks duplicates
    if annotation = False: load neurons from downloaded csv (paired_neurons)

    neurons stored as pickle object, will be loaded if in cache folder. N.B. cached neurons must be renamed or deleted if you wish to analyse a different set
    
    Returns:
        {l/r}_neurons: tuple of CmNLs for left and right side respectively
    '''    
    pairs = pd.read_csv(HERE / "paired_neurons.csv")
    pairs.drop('region', axis=1, inplace=True)  
    pairs_ascending = pairs.sort_values(by=["leftid"], ascending=True) 

    # CatmaidNeuron objects must be unique, and thus we must check for any duplicated skids

    has_duplicate = pairs_ascending.duplicated(subset=['leftid'])
    dup_left = pairs_ascending[has_duplicate]
    print("duplicates on left:")
    print(dup_left)

    has_duplicate = pairs_ascending.duplicated(subset=['rightid'])
    dup_right = pairs_ascending[has_duplicate]
    print("duplicates on right:")
    print(dup_right)

    # in Seymour, left side has 2 neurons (4985759 and 8700125) with duplicated pairs owing to developmental phenomenon
    # the second occurence of these pairs are filtered out (as CmNs must be unique), with parallel analysis applied and scores appended at the end


    # similar code as 'check_duplicates' function below, just potentially within a single function here

    dup_l_neurons = []
    dup_r_neurons = []
    # make empty lists for duplicates, which will be overwritten with a dataframe if these exist

    has_duplicate = pairs_ascending.duplicated(subset=['leftid'])
    dup_left = pairs_ascending[has_duplicate]
    
    has_duplicate = pairs_ascending.duplicated(subset=['rightid'])
    dup_right = pairs_ascending[has_duplicate]

    if len(dup_left) or len(dup_right) > 0:

        print("DUPLICATES DETECTED")

        print("duplicates on left:")
        print(dup_left)
        print("duplicates on right:")
        print(dup_right)
    
        dup_l_neurons = list(pairs_ascending.duplicated(subset=['leftid']))
        dup_r_neurons = list(pairs_ascending.duplicated(subset=['rightid']))

        dup_l_neurons = pymaid.get_neuron(dup_l_neurons)
        dup_r_neurons = pymaid.get_neuron(dup_r_neurons)

        l_neurons = pairs_ascending.drop_duplicates(subset='leftid')
        r_neurons = pairs_ascending.drop_duplicates(subset='rightid')

        l_neurons = pymaid.get_neuron(l_neurons)
        r_neurons = pymaid.get_neuron(r_neurons)

    else:

        print("NO DUPLICATES DETECTED")

        l_neurons = list(pairs_ascending["leftid"])
        r_neurons = list(pairs_ascending["rightid"])

        l_neurons = pymaid.get_neuron(l_neurons)
        r_neurons = pymaid.get_neuron(r_neurons)

    return l_neurons, r_neurons, dup_l_neurons, dup_r_neurons

    # Rather than potentially returning unnecessary, duplicate CmNLs, could handle these via a dict?
    # return_dict = {
    #         "left": {
    #             "original": l_neurons,
    #             "dup": dup_l_neurons
    #         }
    #         ,
    #         "right": {
    #             "original": r_neurons,
    #             "dup": dup_r_neurons
    #         }
    #     }

    

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

    # will run into issues if duplicates are present (as more than ! 
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
        

    return left_transform, right_raw




def make_dotprop(neuron, prune_strahler = STRAHLER, resample = RESAMPLE):
    """
    First prunes by specified strahler index and resamples neuron to given resolution
    Subsequently applies navis.make_dotprops to this neuron (k = 5, representing appropriate # of nearest neighbours [for tangent vector calculation] for the sparse point clouds of skeletons)
    Args:
        neuron (TN)
        prune_strahler: prune TN by strahler index, defaults to lowest order only (-1, None)
        resample (int): resamples to # of nodes per every N units of cable? Defaults to 1000
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



def get_dps(landmarks, prune_strahler = STRAHLER, resample = RESAMPLE):
    """
    Obtains left and right pairs from prior functions.
    Transforms left pairs, makes dot products for both these and right pairs and outputs. Loaded from pickle if already ran
    Args:
        prune_strahler: prune TN by strahler index, defaults to lowest order only (-1, None)
        resample (int): resamples to # of nodes per every N units of cable? Defaults to 1000
    Returns:
        list: dotproducts for l_trans and r
    """    
    
    l_trans, r = get_transformed_neurons(landmarks)
    out = tuple(make_dps(ns, prune_strahler, resample) for ns in [l_trans, r])
    filtered_left = []
    filtered_right = []
    for l, r in zip(*out):
        if l is not None and r is not None:
            filtered_left.append(l)
            filtered_right.append(r)

    return filtered_left, filtered_right


# TO DO #


# dependent on whether option 1 or 2 is taken for initial duplicate filtering, this function takes as arguments either a CSV or CmNLs as arguments
def duplicate_prepare(left_duplicate_neurons, right_duplicate_neurons, prune_strahler = STRAHLER, resample = RESAMPLE):
    """ Takes the outputted duplicate skids, performs identical preparatory steps and returns dotprops in form ready for cross_validation()
    Args:
        duplicates (pandas DF): DF of the duplicate skids, outputted from original CSV
    Returns:
        paired: list of the zipped dotprops (left, right), ready for cross_validation 
    """  
    # either take the duplicates as the subsetted CSV (if using the first function), or directly as CmNLs (if using get_neurons directly)
    l_neurons = list(duplicates["leftid"])
    r_neurons = list(duplicates["rightid"])

    neurons_l = pymaid.get_neuron(l_neurons)
    neurons_r = pymaid.get_neuron(r_neurons)

    by_name_l = dict(zip(neurons_l.name, neurons_l))
    by_name_r = dict(zip(neurons_r.name, neurons_r))
    paired_names = list(zip(neurons_l.name, neurons_r.name))

    l_xyz, r_xyz = get_landmarks('cns')
    transform = navis.transforms.MovingLeastSquaresTransform(l_xyz, r_xyz)
    l_trans = []
    r_raw = []

    for l_name, r_name in paired_names:
        l_trans.append(transform_neuron(transform, by_name_l[l_name]))
        r_raw.append(by_name_r[r_name])
    
    left, right = tuple(make_dps(ns, prune_strahler, resample) for ns in [l_trans, r_raw])
    paired = list(zip(left, right))

    return paired


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


def cross_validation_original(
    dp_pairs: List[Tuple[Dotprops, Dotprops]], n_partitions = 5, seed = DEFAULT_SEED
):
    headers = [
        "skeleton_id-transformed-L",
        "skeleton_id-raw-R",
        "mean_normalized_alpha_nblast",
        "partition",
    ]
    dtypes = [int, int, int, float, str]
    rows = []
    for partition, (training, testing) in tqdm(
        enumerate(split_training_testing(dp_pairs, n_partitions, seed)),
        "cross validation partitions",
        n_partitions,
    ):
        nblaster = train_nblaster(training, threads=None)
        for left, right in tqdm(testing, f"testing {partition}"):
            l_idx = nblaster.append(left)
            r_idx = nblaster.append(right)
            result = nblaster.single_query_target(l_idx, r_idx, scores="mean")
            rows.append([left.id, right.id, result, partition])

    df = pd.DataFrame(rows, columns=headers)
    for header, dtype in zip(headers, dtypes):
        df[header] = df[header].astype(dtype)

    df.sort_values(headers[-1], axis=0, inplace=True)

    return df


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

def cross_validation3(dp_pairs: List[Tuple[Dotprops, Dotprops]], n_partitions=5, seed=DEFAULT_SEED):
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



### Run analysis ###



#%%
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    with logging_redirect_tqdm():
        # analysis for most pairs
        left, right = get_dps('cns')      
        paired = list(zip(left, right))
        df = cross_validation(paired)
        print(df)

        ### TO DO ###
        # ensure below runs and merges the output results into a single DF

        # identical analysis done separately for duplicate skids, as CatmaidNeuron objects can't be repeated
        # paired2 = duplicate_prepare(dups)
        # df2 = cross_validation(paired2)
        # print(df2)

        # concat_df = pd.concat([df,df2])
        df.to_csv(OUT_DIR / f"manalysis_results_new.csv", index=False)
        # merge both dataframes and export as CSV