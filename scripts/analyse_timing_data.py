import argparse
import csv
import matplotlib.pyplot as plt
# plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('lines', linewidth=1.0)
import math
import numpy as np

flamegraph_presets = {
    'eqvio': [
        "features",
        "preprocessing",
        "propagation",
        "correction",
    ],
    'openvins': [
        "tracking",
        "propagation",
        "msckf update",
        "slam update",
        "slam delayed",
        "re-tri & marg",
    ]
}


def collect_timing_info(fname, ignore_entries=0):
    with open(fname, 'r') as f:
        reader = csv.reader(f)
        labels = [label.strip() for label in next(reader)]

        for _ in range(ignore_entries):
            next(reader)  # skip header and first few entries

        timing_dict = {label: [] for label in labels}
        for row in reader:
            for (label, entry) in zip(labels, row):
                timing_dict[label].append(float(entry)*1.0e3)

    return labels, timing_dict


def plot_timing_histograms(labels, timing_dict):
    cols = math.ceil(len(labels)**0.5)
    rows = math.ceil(len(labels)/cols)
    fig, axs = plt.subplots(rows, cols)

    for (i, label) in enumerate(labels):
        ax = axs.ravel()[i]
        ax.hist(timing_dict[label], 100)
        # ax.boxplot(timing_dict[label])
        ax.set_xlabel("Timing (ms)")
        ax.set_ylabel("Count")
        ax.set_title(label.capitalize())

    fig.tight_layout()

    return {"timing_histograms": fig}


def plot_flame_graph(labels, timing_dict, keys):
    flame_len = None
    for k in keys:
        assert(k in labels)

    flame_len = min(len(timing_dict[k]) for k in keys)

    # Set up colours
    my_cmap = plt.get_cmap("rainbow")

    fig, ax = plt.subplots()
    cumulative_flames = np.zeros(flame_len)
    frames = np.arange(flame_len)
    for (i, k) in enumerate(keys):
        timing_array = np.array(timing_dict[k])
        new_cum_flames = cumulative_flames + timing_array
        ax.fill_between(frames, cumulative_flames, new_cum_flames,
                        label=k.capitalize(), color=my_cmap(i/(len(keys)-1)), linewidth=0.0)
        cumulative_flames = new_cum_flames

    mean_time = np.mean(cumulative_flames)
    ax.axhline(mean_time, c='k', ls=":")

    # Reverse the legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels))

    ax.set_xlabel("Frame number")
    ax.set_ylabel("Time taken (ms)")
    ax.set_title("Processing Time per Frame")
    ax.set_xlim(0, flame_len-1)
    ax.set_ylim(0, None)

    return {"mean time (ms)": float(mean_time)}, {"timing_flamegraph": fig}


def plot_timing_boxplots(labels, timing_dict):
    fig, axs = plt.subplots(1, 1)
    axs.boxplot(list(timing_dict[lab]
                for lab in labels), tick_labels=labels, sym="")
    # axs.set_xlabel(timing_dict.keys())

    return {"timing_boxplots": fig}


def plot_timing_data(fname, flamegraph_type='eqvio'):
    labels, timing_dict = collect_timing_info(fname, 10)

    if flamegraph_type in flamegraph_presets:
        flame_graph_keys = flamegraph_presets[flamegraph_type]
    else:
        flame_graph_keys = labels

    figures_dict = {}
    figures_dict.update(plot_timing_histograms(flame_graph_keys, timing_dict))
    figures_dict.update(plot_timing_boxplots(flame_graph_keys, timing_dict))
    time_results, fg_fig = plot_flame_graph(
        labels, timing_dict, flame_graph_keys)
    figures_dict.update(fg_fig)

    return time_results, figures_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Analyse timing information.")
    parser.add_argument("file", metavar='f', type=str,
                        help="The file containing timing data")
    parser.add_argument("--type", type=str, help="The type of timing file. 'eqvio' (default) or 'openvins'", default='eqvio')
    args = parser.parse_args()

    if args.type is None:
        plot_timing_data(args.file)
    else:
        plot_timing_data(args.file, args.type)

    plt.show()
