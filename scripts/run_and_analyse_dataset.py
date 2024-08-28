#!/usr/bin/env python3

import os
from analysis_tools import analyse_dataset
from DatasetInfo import add_dataset_parsing, read_dataset_list
import subprocess
import argparse
import time


def run_over_dataset(ds_info: 'DatasetInfo', config_fname: str, verbose: bool = False, simvis=False):

    eqvio_command = ds_info.eqvio_command(config_fname, simvis)

    t0 = time.time()
    try:
        sts = subprocess.run(eqvio_command, shell=True, capture_output=True, text=True)
    except KeyboardInterrupt as e:
        print("COMMAND USED:")
        print(eqvio_command)
        raise e
    t1 = time.time()
    if verbose:
        print(sts.stdout)
        print("EQVIO Took {} seconds.".format(t1-t0))
    ds_info.timing = t1 - t0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run and analyse EqF VIO in the given directory.")
    parser.add_argument("-d", "--display", action='store_true',
                        help="Set this flag to display the output plots.")
    parser.add_argument("-v", "--verbose", action='store_true',
                        help="Set this flag to give verbose output.")
    parser.add_argument("--simvis", action='store_true',
                        help="Simulate vision measurements")
    parser.add_argument("config", type=str, help="The config file.")
    parser = add_dataset_parsing(parser)
    args = parser.parse_args()

    datasets_info = read_dataset_list(args.directory, args.mode)

    assert os.path.isfile(args.config), "Configuration file not found at {}".format(args.config)

    for ds_info in datasets_info:
        print("Running EqVIO and analysing: {}".format(ds_info.short_name()))
        run_over_dataset(ds_info, args.config, args.verbose, args.simvis)
        analyse_dataset(ds_info, save=True)
