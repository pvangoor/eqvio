from dataclasses import dataclass
import os
from pylie import SE3, Trajectory, R3
import csv
import argparse
import yaml
import numpy as np


@dataclass
class DatasetInfo:
    dataset_fname: str
    mode: str
    eval_start: float = -1.0
    camera_file: str = None
    start = None
    stop = None
    timing: float = float('nan')

    def dir_name(self):
        return self.dataset_fname[:self.dataset_fname.rfind("/")] + "/"

    def short_name(self):
        if self.mode == 'ros' or self.mode == 'hilti':
            sname = self.rosbag_basename()
            sname = sname[sname.rfind("/")+1:]
        else:
            d = self.dir_name()
            sname = d[d[:-1].rfind('/')+1:-1]
        return sname

    def eqvio_command(self, config, simvis=False) -> str:
        eqvio_command = "build/eqvio_opt \"{}\" \"{}\" {} --mode {} --output {} --timing"

        if simvis:
            simvis_str = "--simvis"
        else:
            simvis_str = ""

        if self.camera_file is not None:
            eqvio_command += " --camera " + self.camera_file

        if self.start is not None:
            eqvio_command += " --start " + str(self.start)
        if self.stop is not None:
            eqvio_command += " --stop " + str(self.stop)

        eqvio_command = eqvio_command.format(
            self.dataset_fname, config, simvis_str, self.mode, self.eqvio_output_dir())
        return eqvio_command

    def eqvio_output_dir(self) -> str:
        if self.mode == 'ros' or self.mode == 'hilti':
            eqvio_dirname = self.rosbag_basename() + "_EQVIO_output/"
        else:
            eqvio_dirname = "{}EQVIO_output/".format(self.dir_name())
        return eqvio_dirname

    def eqvio_output_fname(self) -> str:
        return self.eqvio_output_dir() + "IMUState.csv"

    def eqvio_timing_fname(self) -> str:
        if self.mode == 'ros' or self.mode == 'hilti':
            eqvio_fname = self.rosbag_basename() + "_EQVIO_timing.csv"
        else:
            eqvio_fname = "{}EQVIO_timing.csv".format(self.dir_name())
        return eqvio_fname

    def ground_truth_fname(self) -> str:
        # Get the ground truth file
        if self.mode == 'ros':
            gt_fname = self.rosbag_basename() + "_ground_truth.csv"
        elif self.mode == 'asl':
            gt_fname = self.dir_name() + "mav0/state_groundtruth_estimate0/data.csv"
        elif self.mode == 'uzhfpv':
            gt_fname = self.dir_name() + "groundtruth.txt"
        elif self.mode == 'hilti':
            gt_fname = self.rosbag_basename() + "_imu.txt"
        elif self.mode == 'anu':
            gt_fname = self.dir_name() + "ground_truth.csv"
        else:
            gt_fname = "{}ground_truth.csv".format(
                self.dir_name())

        # Check ground truth exists
        if not os.path.isfile(gt_fname):
            print("The ground truth was not found at {}".format(gt_fname))
            gt_fname = None
        return gt_fname

    def has_groundtruth(self):
        gt_fname = self.ground_truth_fname()
        return (gt_fname is not None)

    def results_dir(self) -> str:
        if self.mode == 'ros' or self.mode == 'hilti':
            ret_dir = self.rosbag_basename() + "_results/"
        else:
            ret_dir = "{}results/".format(self.dir_name())
        return ret_dir

    def results_yaml(self) -> str:
        return self.results_dir() + "results.yaml"

    def rosbag_basename(self) -> str:
        assert(self.mode == 'ros' or self.mode == 'hilti')
        return self.dataset_fname[:self.dataset_fname.rfind('.')]

    def tscale(self) -> float:
        if self.mode == 'asl':
            return 1.0e-9
        else:
            return 1.0

    def get_groundtruth(self) -> Trajectory:
        gt_fname = self.ground_truth_fname()
        if gt_fname is None:
            return None
        traj, nan_flag = read_trajectory(gt_fname, self.tscale(), time_col=self.gt_pose_col()-1, pose_col=self.gt_pose_col(),
                                         start_offset=self.eval_start, delim=self.gt_delim(), fmt=self.gt_fmt())
        return traj, nan_flag

    def gt_delim(self) -> str:
        if self.mode == "uzhfpv" or self.mode == 'hilti':
            return " "
        else:
            return ","

    def gt_pose_col(self) -> int:
        return 1

    def gt_fmt(self) -> str:
        if self.mode == "uzhfpv" or self.mode == 'hilti':
            return "xq"
        else:
            return "xw"

    def get_est_imu(self) -> Trajectory:
        eqvio_fname = self.eqvio_output_fname()
        if eqvio_fname is None:
            return None
        traj, nan_flag = read_trajectory(eqvio_fname)
        return traj, nan_flag

    def get_est_vel(self) -> Trajectory:
        eqvio_fname = self.eqvio_output_fname()
        if eqvio_fname is None:
            return None
        traj = read_velocities(eqvio_fname, start_offset=self.eval_start)
        return traj

    def get_est_camera(self) -> Trajectory:
        camera_fname = self.eqvio_output_dir() + "camera.csv"
        traj = read_trajectory(camera_fname, pose_col=1,
                               start_offset=self.eval_start)
        return traj

    def get_est_bias(self):
        bias_fname = self.eqvio_output_dir() + "bias.csv"
        os.path.isfile(bias_fname)
        return read_biases(bias_fname)

    def get_features_info(self):
        features_fname = self.eqvio_output_dir() + "features.csv"
        os.path.isfile(features_fname)
        return read_features_info(features_fname)

    def count_frames(self) -> int:
        if self.mode == 'asl':
            frame_dir = self.dir_name() + "mav0/cam0/data"
            return len(os.listdir(frame_dir))
        elif self.mode == 'uzhfpv':
            images_file = self.dir_name() + "left_images.txt"
            with open(images_file, 'r') as f:
                num_frames = len(f.readlines()) - 1
            return num_frames
        elif self.mode == 'hilti':
            return len(self.get_estimated())
        else:
            return -1

    def get_timing_fname(self) -> str:
        return self.eqvio_output_dir() + "timing.csv"


def read_trajectory(fname: str, tscale: float = 1.0, time_col: int = 0, pose_col: int = 1, start_offset: float = -1.0, delim=',', fmt='xw') -> Trajectory:
    poses = []
    stamps = []
    nanflag = False
    with open(fname, 'r') as f:
        reader = csv.reader(f, delimiter=delim)
        next(reader)  # Skip the header

        stamp0 = None
        for row in reader:
            stamp = float(row[time_col]) * tscale
            if stamp < 0:
                continue

            if stamp0 is None:
                stamp0 = stamp

            if stamp < start_offset + stamp0:
                continue
            row = row[pose_col:pose_col+7]

            try:
                pose = SE3.from_list(row, fmt)
            except ValueError:
                print("NaNs detected in pose data. Ending data here.")
                nanflag = True
                break

            stamps.append(stamp)
            poses.append(pose)
    return Trajectory(poses, stamps), nanflag


def read_velocities(fname: str, start_offset: float = -1.0) -> Trajectory:
    velocities = []
    stamps = []
    with open(fname, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header

        for row in reader:
            stamp = float(row[0])
            if stamp < start_offset:
                continue
            row = row[8:11]

            try:
                velocity = R3.from_list(row)
            except ValueError:
                print("NaNs detected in pose data. Ending data here.")
                break

            stamps.append(stamp)
            velocities.append(velocity)
    return Trajectory(velocities, stamps)


def get_common_directory(datasets_info: list) -> str:
    assert(len(datasets_info) > 0), "The list of datasets is empty."
    if len(datasets_info) == 1:
        d = datasets_info[0].dir_name()
        dirname = d[:d[:-1].rfind("/")] + "/"
    else:
        abs_paths = [os.path.abspath(d.dir_name()) for d in datasets_info]
        dirname = os.path.commonpath(abs_paths)

    if dirname[-1] != '/':
        dirname = dirname + '/'

    return dirname


def add_dataset_parsing(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("directory", nargs="*",
                        help="The dataset directory/directories.")
    parser.add_argument("-m", "--mode", choices=['gift', 'ros', 'sim', 'asl', 'uzhfpv', 'hilti'], default=None,
                        help="Select the type of datasets to run on.")
    return parser


def read_datasets_yaml(fname: str) -> list:
    with open(fname, 'r') as f:
        ddict = yaml.safe_load(f)

    def info_from_yaml(dset: dict) -> DatasetInfo:
        info = DatasetInfo(dset["fname"], dset["mode"])
        if "start" in dset:
            info.start = float(dset["start"])
            info.eval_start = float(dset["start"])
        if "eval_start" in dset:
            info.eval_start = float(dset["eval_start"])
        if "stop" in dset:
            info.stop = float(dset["stop"])
        if "camera" in dset:
            info.camera_file = dset["camera"]
        return info

    datasets_info = [info_from_yaml(dset) for dset in ddict.values()]
    return datasets_info


def read_dataset_list(directory: list, mode: str | None) -> list:
    if not directory:
        import sys
        directory = sys.stdin.readlines()
        directory = [d.strip(' \n') for d in directory]

    if mode is None:
        # The directory argument contains a yaml file with dataset information.
        # This is required if you want to add options like start offset or camera file location.
        datasets_info = [i for d in directory for i in read_datasets_yaml(d)]
    else:
        # Each element (possibly only 1) of the directory argument is a dataset
        datasets_info = [DatasetInfo(name, mode) for name in directory]

    return datasets_info


def read_biases(fname):
    full_stack = np.loadtxt(fname, skiprows=1, delimiter=',')
    times = full_stack[:, 0]

    good_rows = (times > 0)
    times = times[good_rows]
    bias_gyr = full_stack[good_rows, 1:4]
    bias_acc = full_stack[good_rows, 4:7]
    return times.T, bias_gyr.T, bias_acc.T


def read_features_info(fname):
    with open(fname, 'r') as features_file:
        reader = csv.reader(features_file)
        next(reader)

        times = []
        features = []

        for row in reader:
            stamp = float(row[0])
            if stamp < 0:
                continue
            times.append(stamp)
            del row[0]

            n = len(row) // 3
            current_features = {}
            for i in range(n):
                num = int(row[3*i])
                coordinates = np.reshape(row[3*i+1:3*i+3], (2, 1))
                current_features[num] = coordinates

            features.append(current_features)

        return times, features
