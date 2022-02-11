"""Use docstrings at the top of the file to document scripts' purpose."""
import json
import logging
import pickle
import datetime as dt

import navis
import pymaid

from utils.constants import CACHE_DIR, CREDENTIALS_DIR, OUTPUT_DIR

logger = logging.getLogger(__name__)

# change "example.json" to match your credentials file's name
with open(CREDENTIALS_DIR / "example.json") as f:
    CREDS = json.load(f)

catmaid = pymaid.CatmaidInstance(**CREDS)


def example_function_with_cache(some_arg, another_arg, force=False):
    """Use docstrings like this to describe functions.

    Here, some_arg and another_arg would be something like a name (str) or number
    which inform a query from CATMAID.
    You may want to cache "raw" data which takes a long time to fetch or can change beneath your feet.
    Where possible, don't match cached and fresh data.

    force=True ignores the cache and generates (then caches) new data.
    """
    cache_file = CACHE_DIR / f"example_{some_arg}_{another_arg}.pickle"
    if cache_file.is_file() and not force:
        logger.info("Loading cached output from %s", cache_file)
        return pickle.loads(cache_file.read_bytes())

    # do whatever data wrangling you were planning on...
    data = example_function(some_arg, another_arg)

    logger.info("Writing result to cache at %s", cache_file)
    cache_file.write_bytes(pickle.dumps(data))
    return data


# write your output to a timestamped file like
with open(OUTPUT_DIR / f"output_{dt.date.today().isoformat()}.txt", "w") as f:
    f.write(my_serialised_output)
