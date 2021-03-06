/*
    This file is part of EqVIO.

    EqVIO is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    EqVIO is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with EqVIO.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include "VIOState.h"
#include "common.h"

/** @brief A struct with a time stamp and SE(3) pose.
 */
struct StampedPose {
    double t;         ///< The time stamp of the pose data
    liepp::SE3d pose; ///< The pose data as an SE(3) element
};
/** @brief A simulator for VIO measurements from a given groundtruth trajectory.
 *
 * This class is able to simulate IMU and feature measurements from a given groundtruth VIO trajectory. The simulation
 * world has a large number of landmark points, and these are tracked between requests for vision measurements,
 * similarly to a real-world feature tracking output. This code is mostly used for testing purposes, and can be used
 * alongside real data to provide just IMU or just feature measurements.
 */
class VIOSimulator {
  protected:
    uint randomSeed; ///< The value used to seed the random number generation

    mutable std::map<int, int> pointId2TrackingId; ///< A map from world point indexes to tracking id numbers
    mutable int currentTrackingId = 0;             ///< The latest tracking id assigned to a point

    std::vector<StampedPose> poses;       ///< All the ground truth poses with time stamps.
    std::vector<Landmark> inertialPoints; ///< All the points generated for the simulation world.

    liepp::SE3d cameraOffset = liepp::SE3d::Identity(); ///< The chosen offset between IMU and camera.
    double fieldOfView = 60 * 3.14 / 180.;              ///< The camera's desired field of view, in radians.
    size_t maxFeatures = 30;                            ///< The maximum number of features tracked at any given time.

    /** @brief Generate a number of points for the simulator world.
     *
     * @param num The number of points to generate in the world.
     * @param distance The distance of the world's walls from the trajectory.
     * @return The generated world points.
     *
     * The world points are generated by determining simulation world walls, and then placing an even number of points
     * on each of the six walls. The walls are determined as the extremities of the provided trajectory plus the given
     * distance.
     */
    std::vector<Landmark> generateWorldPoints(const int num = 1000, const double distance = 1.0) const;

  public:
    /** @brief Get a vision measurement at the given time.
     *
     * @param time The time of the vision measurement.
     * @return The tracked features at the given time.
     *
     * The currently tracked features are tracked to the given time, if possible. Then, if this is not the maximum
     * number of features, new features are added from the world points.
     */
    VisionMeasurement getVision(const double& time) const;

    /** @brief Get an IMU measurement at the given time.
     *
     * @param time The time of the IMU measurement.
     * @return The IMU velocity at the given time.
     *
     * The IMU measurement at the given time is obtained by numerical differentiation using four groundtruth poses.
     */
    IMUVelocity getIMU(const double& time) const;

    /** @brief Get a const reference to the vector of ground truth poses.
     *
     * @return The ground truth stamped poses.
     */
    const std::vector<StampedPose>& viewPoses() const;

    VIOSimulator() = default;

    /** @brief Create a simulator from a groundtruth file and some settings
     *
     * @param posesFileName The name of the groundtruth file
     * @param settings The simulator settings YAML node
     * @param timeScale The representation of time in the groundtruth file. e.g. 1.0e-9 for ns.
     * @param column The first column with relevant data in the groundtruth file.
     * @param delim The file delimiter
     *
     * @todo Reading groundtruth in this way should be the responsibility of the relevant datasetReader. The simulator
     * constructor should instead just accept the list of stamped poses.
     */
    VIOSimulator(
        const std::string& posesFileName, const YAML::Node& settings = YAML::Node(), const double timeScale = 1.0,
        const int column = 0, const char delim = ',');
};
