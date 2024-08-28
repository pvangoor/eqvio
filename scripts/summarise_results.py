import argparse
import yaml
import os
import datetime
from DatasetInfo import *


def nested_get(d: dict, l: list):
    if d is None:
        return 1.0e6
    # Given a list l of keys, retrieve an item from d recursively
    dn = d
    for k in l:
        dn = dn[k]
    return dn


def collect_results(datasets_info: list):
    # Summarise all the EqVIO results from a given directory.
    all_results = {}
    for ds_info in datasets_info:
        dataset_name = ds_info.short_name()
        if os.path.exists(ds_info.results_yaml()):
            with open(ds_info.results_yaml(), 'r') as f:
                results_dict = yaml.safe_load(f)
            all_results[dataset_name] = results_dict
        else:
            print("No results available for {}".format(dataset_name))

    return all_results


def summarise_results(all_results, nested_key_list):
    sum_results = {}
    for nested_key in nested_key_list:
        nk_name = nested_key[0]
        if len(nested_key) > 1:
            for k in nested_key[1:]:
                nk_name = nk_name + "/" + k
        sum_results[nk_name] = {}

        for results_key in all_results:
            r = nested_get(all_results[results_key], nested_key)
            sum_results[nk_name][results_key] = r

    return sum_results


def add_all_results_by_key(all_results, nested_key):
    total = 0.0
    for results_key in all_results:
        r = nested_get(all_results[results_key], nested_key)
        total += float(r)

    return total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Summarise the EqVIO results from a directory containing several results directories.")
    parser = add_dataset_parsing(parser)
    args = parser.parse_args()

    datasets_info = read_dataset_list(args.directory, args.mode)

    all_results = collect_results(datasets_info)

    summary_keys = [
        ["position (m)", "rmse"],
        ["attitude (d)", "rmse"],
        ["velocity (m/s)", "rmse"],
        ["scale"],
        ["mean time (ms)"],
        ["Early Finish flag"],
        ["NaN flag"],
        ["Trajectory length"],
    ]

    sum_results = summarise_results(all_results, summary_keys)
    total_position_rmse = add_all_results_by_key(
        all_results, ["position (m)", "rmse"])

    print("The mean of position RMSE (cm) is {:.2f}".format(
        100.0*total_position_rmse / len(datasets_info)))

    head_dir_name = get_common_directory(datasets_info)
    summary_fname = head_dir_name + \
        "results_summary_{}.yaml".format(
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    with open(summary_fname, 'w') as f:
        yaml.safe_dump(sum_results, f)
        print("Summary saved to "+summary_fname)
