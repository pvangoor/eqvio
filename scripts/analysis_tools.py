import argparse
from analyse_timing_data import plot_timing_data
import os
from pylie import R3, SE3, SIM3, analysis, Trajectory
import numpy as np
import yaml
import matplotlib.pyplot as plt
from DatasetInfo import *
import sys

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('lines', linewidth=1.0)
papersize = False


def plot_feature_information(ds_info):
    times, feature_information = ds_info.get_features_info()

    # Analyse feature stats etc
    fig_features, ax_features = plt.subplots(2, 1)

    # Number of features over time
    num_features = [len(feat_info) for feat_info in feature_information]
    a_num_feat = ax_features.ravel()[0]
    a_num_feat.plot(times, num_features)
    a_num_feat.set_title("Number of tracked features")
    a_num_feat.set_title("Features (\#)")
    a_num_feat.set_xlabel("Time (s)")
    a_num_feat.set_xlim(times[0], times[-1])
    a_num_feat.set_ylim(0, None)

    # Average Lifetime of features over time
    feature_lifetimes = []
    # TODO: this is inefficient
    for (i, info) in enumerate(feature_information):
        current_features = info.keys()
        current_feature_lifetimes = [sum([cf in feature_information[i-j] for j in range(i)]) for cf in current_features]
        feature_lifetimes.append(current_feature_lifetimes)

    mean_feature_lifetimes = np.hstack([np.mean(fl) for fl in feature_lifetimes])
    a_mean_lt = ax_features.ravel()[1]
    a_mean_lt.plot(times, mean_feature_lifetimes)
    a_mean_lt.set_title("Mean Feature Lifetime")
    a_mean_lt.set_ylabel("Lifetime (frames)")
    a_mean_lt.set_xlabel("Time (s)")
    a_mean_lt.set_xlim(times[0], times[-1])
    a_mean_lt.set_ylim(0, None)

    fig_features.tight_layout()
    return {"features": fig_features}


def plot_camera_offset(ds_info):
    est_cam_offset, nanflag = ds_info.get_est_camera()

    fig_offset, ax_offset = plot_trajectory(est_cam_offset, 'r--')
    ax_offset[0, 0].set_title("IMU-Camera Offset Rotation")
    ax_offset[0, 1].set_title("IMU-Camera Offset Position")
    fig_offset.suptitle("Estimated IMU-Camera Offset Trajectory")
    fig_offset.tight_layout()

    return {"camera_offset": fig_offset}


def plot_biases(ds_info: DatasetInfo):
    bias_times, bias_gyr, bias_acc = ds_info.get_est_bias()

    fig_bias, ax_bias = plt.subplots(3, 2)
    plot_stack(ax_bias[:, 0], bias_times, bias_gyr, "Gyr b. {} (rad/s)", 'r--')
    plot_stack(ax_bias[:, 1], bias_times, bias_acc, "Acc b. {} (m/s/s)", 'r--')

    ax_bias[0, 0].set_title("IMU Gyr. Bias")
    ax_bias[0, 1].set_title("IMU Acc. Bias")
    fig_bias.suptitle("Estimated IMU Biases")

    fig_bias.set_size_inches(11.69, 8.27)
    if papersize:
        fig_bias.set_size_inches(6.0, 6.0 * 8.27 / 11.69)
        fig_bias.tight_layout()

    return {"biases": fig_bias}


def analyse_imu_trajectory(ds_info: DatasetInfo, make_plots=True):

    # -------------------------------
    # Read in the ground truth and estimated trajectories and velocities
    # -------------------------------
    has_ground_truth = ds_info.has_groundtruth()
    tru_nan_flag = False
    if has_ground_truth:
        tru_trajectory, tru_nan_flag = ds_info.get_groundtruth()

    est_trajectory, est_nan_flag = ds_info.get_est_imu()
    assert (len(est_trajectory) > 0), "The estimated trajectory is empty."
    est_velocities = ds_info.get_est_vel()

    nan_flag = tru_nan_flag or est_nan_flag

    # The early finish flag indicates the estimated trajectory ends more than 10% before the end of the true trajectory.
    # This indicates the algorithm has failed and renders the results questionable
    if has_ground_truth:
        if ds_info.stop is not None:
            early_finish_flag = (est_trajectory.get_times()[-1] <= 0.9 * (ds_info.stop + tru_trajectory.get_times()[0]))
        else:
            early_finish_flag = (est_trajectory.get_times()[-1] <= 0.9 * (tru_trajectory.get_times()[-1] - tru_trajectory.get_times()[0]) + tru_trajectory.get_times()[0])
    else:
        early_finish_flag = False

    # -------------------------------
    # Align the trajectories
    # -------------------------------
    if has_ground_truth:
        min_time = max(est_trajectory.begin_time(), tru_trajectory.begin_time())
        max_time = min(est_trajectory.end_time(), tru_trajectory.end_time())
        est_trajectory.truncate(min_time, max_time)
        tru_trajectory.truncate(min_time, max_time)

        if len(tru_trajectory.get_times()) < len(est_trajectory.get_times()):
            comp_times = tru_trajectory.get_times()
            est_trajectory = est_trajectory[comp_times]
        else:
            comp_times = est_trajectory.get_times()
            tru_trajectory = tru_trajectory[comp_times]
        assert (len(est_trajectory) > 0) and (len(tru_trajectory) > 0), "The trajectories do not overlap."
        
        tru_velocities = Trajectory([R3(tru_trajectory.get_velocity(t)[3:6]) for t in comp_times], comp_times)

        est_trajectory, alignment = analysis.align_trajectory(est_trajectory, tru_trajectory, ret_params=True)
    else:
        alignment = SIM3.identity()
        comp_times = est_trajectory.get_times()
    est_velocities = est_velocities[comp_times]

    # -------------------------------
    # Compute gravity, velocity, and trajectory errors
    # -------------------------------
    # Get true and estimated gravity and velocity

    gravity_vector = np.array([[0], [0], [-9.81]])
    est_gravity = np.hstack([pose.R().inv() * gravity_vector for pose in est_trajectory.get_elements()])
    est_velocities = np.stack([vel.as_vector() for vel in est_velocities.get_elements()]).T

    if has_ground_truth:
        tru_gravity = np.hstack([pose.R().inv() * gravity_vector for pose in tru_trajectory.get_elements()])
        tru_velocities = np.stack([vel.as_vector() for vel in tru_velocities.get_elements()]).T

        # Trajectory errors
        err_attitude = 180.0 / np.pi * np.hstack([np.reshape((tru.R() * est.R().inv()).log(), (3, 1))
            for (est, tru) in zip(est_trajectory.get_elements(), tru_trajectory.get_elements())])
        err_position = np.stack([tru.x().as_vector() - est.x().as_vector() 
            for (est, tru) in zip(est_trajectory.get_elements(), tru_trajectory.get_elements())]).T
        err_velocity = tru_velocities - est_velocities

    # -------------------------------
    # Compute statistics
    # -------------------------------

    if has_ground_truth:
        tru_positions = np.stack([p.x().as_vector() for p in tru_trajectory.get_elements()]).T
        traj_length = float(np.sum(np.linalg.norm(np.diff(tru_positions, axis=1), axis=0)))
        
        absolute_position_error_stats = computeStatistics(np.linalg.norm(err_position, axis=0))
        absolute_attitude_error_stats = computeStatistics(np.linalg.norm(err_attitude, axis=0))
        velocity_error_stats = computeStatistics(np.linalg.norm(err_velocity, axis=0))
    else:
        est_positions = np.stack([p.x().as_vector() for p in est_trajectory.get_elements()]).T
        traj_length = float(np.sum(np.linalg.norm(np.diff(est_positions, axis=1), axis=0)))
        absolute_position_error_stats = computeStatistics(0.0)
        absolute_attitude_error_stats = computeStatistics(0.0)
        velocity_error_stats = computeStatistics(0.0)


    results_dict = {"NORESULT": (not has_ground_truth),
                    "position (m)": absolute_position_error_stats,
                    "attitude (d)": absolute_attitude_error_stats,
                    "velocity (m/s)": velocity_error_stats,
                    "scale": alignment.s().as_float(),
                    "NaN flag": nan_flag,
                    "Early Finish flag": early_finish_flag,
                    "Trajectory length": traj_length,
                    }

    if not make_plots:
        return results_dict, {}


    figs_dict = {}
    # -------------------------------
    # Plot the aligned trajectories
    # -------------------------------
    fig_traj, ax_traj = plot_trajectory(est_trajectory, 'r--', "Est.")
    if has_ground_truth:
        fig_traj, ax_traj = plot_trajectory(tru_trajectory, 'b-', "True", fig_traj, ax_traj)
    ax_traj[0, 0].set_title("Robot (IMU) Attitude")
    ax_traj[0, 1].set_title("Robot (IMU) Position")
    ax_traj[-1, 0].legend(loc="lower left", labelspacing=0.2)
    ax_traj[-1, 1].legend(loc="lower left", labelspacing=0.2)
    fig_traj.suptitle("Robot (IMU) Trajectory")
    fig_traj.tight_layout()

    fig_xy, ax_xy = plot_trajectory_xy(est_trajectory, 'r--', "Est.")
    if has_ground_truth:
        fig_xy, ax_xy = plot_trajectory_xy(tru_trajectory, 'b-', "True", fig_xy, ax_xy)
    fig_xy.suptitle("Robot (IMU) Trajectory 2D")
    ax_xy.legend(loc="lower left", labelspacing=0.2)

    figs_dict["trajectory"] = fig_traj
    figs_dict["trajectory_xy"] = fig_xy

    # ------------------------------------------
    # Plot the attitude and position errors over time
    # ------------------------------------------
    if has_ground_truth:
        fig_errs, ax_errs = plt.subplots(4, 2)
        plot_stack(ax_errs[:, 0], comp_times, np.vstack((err_attitude, np.linalg.norm(err_attitude, axis=0))), "{} (deg)", "m-", "Error")
        ax_errs[0, 0].set_title("Robot Attitude Error")
        ax_errs[3, 0].set_ylim([0.0, None])

        plot_stack(ax_errs[:, 1], comp_times, np.vstack((err_position, np.linalg.norm(err_position, axis=0))), "{} (m)", "m-", "Error")
        ax_errs[0, 1].set_title("Robot Position Error")
        ax_errs[3, 1].set_ylim([0.0, None])

        fig_errs.set_size_inches(11.69, 8.27)
        if papersize:
            fig_errs.set_size_inches(6.0, 6.0 * 8.27 / 11.69)
            fig_errs.tight_layout()
        
        figs_dict["trajectory_error"] = fig_errs

    # -------------------------------
    # Plot the velocity and gravity
    # -------------------------------
    fig_vel_grav, ax_vel_grav = plt.subplots(3, 2)
    if has_ground_truth:
        plot_stack(ax_vel_grav[:, 0], comp_times, tru_gravity, "gravity {} (m/s/s)", "b-", "True")
    plot_stack(ax_vel_grav[:, 0], comp_times, est_gravity, "gravity {} (m/s/s)", "r--", "Est.")
    ax_vel_grav[0, 0].set_title("Gravity Direction")
    ax_vel_grav[-1, 0].legend(loc="lower left", labelspacing=0.2)

    if has_ground_truth:
        plot_stack(ax_vel_grav[:, 1], comp_times, tru_velocities, "velocity {} (m/s)", "b-", "True")
    plot_stack(ax_vel_grav[:, 1], comp_times, est_velocities, "velocity {} (m/s)", "r--", "Est.")
    ax_vel_grav[0, 1].set_title("Body-Fixed Velocity")
    ax_vel_grav[-1, 1].legend(loc="lower left", labelspacing=0.2)

    fig_vel_grav.set_size_inches(11.69, 8.27)
    if papersize:
        fig_vel_grav.set_size_inches(6.0, 6.0 * 8.27 / 11.69)
        fig_vel_grav.tight_layout()

    figs_dict["gravity_and_velocity"] = fig_vel_grav

    return results_dict, figs_dict


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


def computeStatistics(values: list) -> dict:
    # Compute the statistics (mean, std, median, min, max) from a tuple of values
    stats = {}
    stats["rmse"] = float(np.sqrt(np.mean(values**2)))
    stats["mean"] = float(np.mean(values))
    stats["std"] = float(np.std(values))
    stats["med"] = float(np.median(values))
    stats["min"] = float(np.min(values))
    stats["max"] = float(np.max(values))

    return stats


def get_pos_att_stacks(trajectory: Trajectory):
    if len(trajectory) == 0:
        return np.zeros((3, 0)), np.zeros((3, 0))
    positions = np.stack([pose.x().as_vector() for pose in trajectory.get_elements()]).T
    attitudes = np.hstack([np.reshape(pose.R().as_euler(), (3, 1)) for pose in trajectory.get_elements()])
    return positions, attitudes


def constrain_axis_ylim(ax, llim, ulim):
    cllim, culim = ax.get_ylim()
    if cllim > llim:
        llim = cllim
    if culim < ulim:
        ulim = culim
    ax.set_ylim(llim, ulim)


def plot_trajectory(trajectory: Trajectory, ls: str = '-', label=None, fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots(3, 2)

    # Plot the attitude comparison
    positions, attitudes = get_pos_att_stacks(trajectory)
    times = trajectory.get_times()

    plot_stack(ax[:, 0], times, attitudes, "Euler {} (deg)", ls, label)
    plot_stack(ax[:, 1], times, positions, "Position {} (m)", ls, label)

    # Adjust axis limits for Euler angles
    constrain_axis_ylim(ax[0, 0], -180.0, 180.0)
    constrain_axis_ylim(ax[1, 0], -90.0, 90.0)
    constrain_axis_ylim(ax[2, 0], -180.0, 180.0)

    fig.set_size_inches(11.69, 8.27)
    if papersize:
        fig.set_size_inches(6.0, 6.0 * 8.27 / 11.69)

    return fig, ax


def plot_trajectory_xy(trajectory: Trajectory, ls: str = 'r-', label=None, fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1)

    positions = np.stack([pose.x().as_vector()
                           for pose in trajectory.get_elements()])
    ax.plot(positions[:,0], positions[:,1], ls, label=label)

    ax.set_xlabel("Position x (m)")
    ax.set_ylabel("Position y (m)")

    fig.set_size_inches(11.69, 8.27)
    if papersize:
        fig.set_size_inches(6.0, 6.0 * 8.27 / 11.69)

    return fig, ax


def plot_stack(ax, times, stack, ylabel_f, ls='b-', label=None):
    # colours = ['r', 'g', 'b', 'k']
    ax_names = ['x', 'y', 'z', 'norm']
    for i in range(stack.shape[0]):
        # ax[i].plot(times, stack[i, :], colours[i]+ls, label=label)
        ax[i].plot(times, stack[i, :], ls, label=label)
        ax[i].set_ylabel(ylabel_f.format(ax_names[i]))
        ax[i].set_xlim([times[0], times[-1]])
        ax[i].grid(visible=True)

        if i < stack.shape[0]-1:
            ax[i].set_xticklabels([])

    ax[-1].set_xlabel("Time (s)", labelpad=12.0)


def statString(stats: dict):
    result = ""
    for key, val in stats.items():
        result += "{:>6s}: {:<.4f}\n".format(key, val)
    return result


def analyse_dataset(ds_info: DatasetInfo, noplot: bool = False, save: bool = False):
    if not os.path.exists(ds_info.results_dir()):
        os.mkdir(ds_info.results_dir())

    try:
        results_dict, figures = analyse_imu_trajectory(ds_info, not noplot)
        results_dict["FPS"] = ds_info.count_frames() / ds_info.timing
        if not noplot:
            figures.update(plot_camera_offset(ds_info))
            figures.update(plot_biases(ds_info))
            figures.update(plot_feature_information(ds_info))

            timing_fname = ds_info.get_timing_fname()
            if os.path.isfile(timing_fname):
                time_results, time_figures = plot_timing_data(timing_fname)
                results_dict.update(time_results)
                if not noplot:
                    figures.update(time_figures)

    except AssertionError as e:
        print(e)
        # Delete the results yaml
        if os.path.exists(ds_info.results_yaml()):
            os.remove(ds_info.results_yaml())
        return

    # Save or show stats and figures
    print("Saving analysis results to {}".format(ds_info.results_dir()))
    if save:
        with open(ds_info.results_yaml(), 'w') as f:
            yaml.dump(results_dict, f)

        for name, fig in figures.items():
            fig.savefig(ds_info.results_dir()+name+".pdf")

    else:
        yaml.dump(results_dict, sys.stdout)
        if not noplot:
            plt.show()

    plt.close('all')
    for f in figures.values():
        plt.close(f)


if __name__ == '__main__':
    # Set argument parser
    parser = argparse.ArgumentParser(
        description="Analyse EQVIO performance.")
    parser.add_argument("--output", type=str, default=None,
                        help="The output file to analyse.")
    parser.add_argument("--save", action='store_true',
                        help="Save the results instead of plotting.")
    parser.add_argument("--noplot", action="store_true",
                        help="Disable plotting.")
    parser.add_argument("--papersize", action="store_true",
                        help="Make the figures a good size for single column.")
    parser = add_dataset_parsing(parser)
    args = parser.parse_args()

    datasets_info = read_dataset_list(args.directory, args.mode)

    papersize = args.papersize

    for ds_info in datasets_info:
        print("Analysing {}".format(ds_info.dataset_fname))
        if args.output is not None:
            import shutil
            shutil.copy(args.output, ds_info.eqvio_output_fname())

        analyse_dataset(ds_info, noplot=args.noplot, save=args.save)
