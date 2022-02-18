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

#include "liepp/SE3.h"

#include "eigen3/Eigen/Eigen"
#include "eqvio/Geometry.h"
#include "eqvio/IMUVelocity.h"
#include "eqvio/VisionMeasurement.h"
#include "eqvio/csv/CSVReader.h"

#include <functional>
#include <memory>
#include <ostream>
#include <vector>

/** @file */

//-----------------------------------
// VIO State Space Structures
//-----------------------------------

/** @brief A structure for VIO landmarks.
 */
struct Landmark {
    Eigen::Vector3d p; ///< The landmark's position
    int id = -1;       ///< The landmark's id number

    constexpr static int CompDim = 3; ///< The dimension of the space of landmarks.
};

/** @brief The states of the sensors of the VIO system.
 *
 * In this way of thinking, the VIO states consist of a fixed number of sensor states and a dynamic number of landmark
 * states.
 */
struct VIOSensorState {
    Eigen::Matrix<double, 6, 1> inputBias; ///< The IMU biases as a vector
    liepp::SE3d pose;                      ///< The IMU/robot pose
    Eigen::Vector3d velocity;              ///< The IMU/robot velocity, in the IMU/robot frame.
    liepp::SE3d cameraOffset;              ///< The camera pose w.r.t. the IMU.

    /** @brief provide the direction of gravity w.r.t. the IMU frame.
     */
    Eigen::Vector3d gravityDir() const;

    constexpr static int CompDim = 6 + 6 + 3 + 6; ///< The dimesion of the space of sensor states.
};

/** @brief The states of the VIO system.
 *
 * The sensor states include the IMU position, velocity, and biases, and the camera offset. The camera landmarks include
 * all the landmark positions in the camera-fixed frame.
 */
struct VIOState {
    VIOSensorState sensor;                 ///< The sensor states of the VIO system
    std::vector<Landmark> cameraLandmarks; ///< The landmark positions in the camera frame.

    /** @brief get the id numbers of all of the camera landmarks.
     */
    std::vector<int> getIds() const;

    constexpr static int CompDim = Eigen::Dynamic; ///< The dimesion of the VIO state space.
    /** @brief get the dimension of the VIO state space element in question.
     */
    int Dim() const;
};

/** @brief Integrate the VIO dynamics
 *
 * @param state The initial state of the VIO system
 * @param velocity The IMU velocities to use for integration
 * @param dt The period of time for which to integrate
 * @return The VIO system state after integration
 *
 * This function discretises the VIO dynamics by assuming the IMU velocity is constant over the time period dt.
 */
[[nodiscard]] VIOState integrateSystemFunction(const VIOState& state, const IMUVelocity& velocity, const double& dt);

/** @brief Make a measurement of features from a VIO state
 *
 * @param state The state used to compute measurements.
 * @param cameraPtr A pointer to the camera model used to produce the measurements.
 * @return The feature measurements obtained.
 */
[[nodiscard]] VisionMeasurement measureSystemState(const VIOState& state, const GIFT::GICameraPtr& cameraPtr);

/** @brief Write VIO sensor states to a CSV line
 *
 * @param line A reference to the CSV line where the data should be written.
 * @param sensor The VIO sensor states to write to the CSV line.
 * @return A reference to the CSV line with the data written.
 */
CSVLine& operator<<(CSVLine& line, const VIOSensorState& sensor);

/** @brief Write a VIO state to a CSV line
 *
 * @param line A reference to the CSV line where the data should be written.
 * @param state The VIO state to write to the CSV line.
 * @return A reference to the CSV line with the data written.
 */
CSVLine& operator<<(CSVLine& line, const VIOState& state);

//-----------------------------------
// Coordinate Chart Declarations
//-----------------------------------

extern const CoordinateChart<VIOSensorState> sensorChart_std; ///< Standard local coordinates for the VIO sensors states
extern const CoordinateChart<VIOSensorState>
    sensorChart_normal;                                     ///< Normal local coordinates for the VIO sensors states
extern const CoordinateChart<Landmark> pointChart_euclid;   ///< Euclidean local coordinates for landmark points
extern const CoordinateChart<Landmark> pointChart_invdepth; ///< Inverse-depth local coordinates for landmark points
extern const CoordinateChart<Landmark> pointChart_normal;   ///< Normal local coordinates for landmark points

/** @brief Construct a VIO coordinate chart from charts for the sensor states and landmarks
 *
 * @param sensorBundleChart The coordinate chart to use for the sensor states.
 * @param pointChart The coordinate chart to use for each of the landmark points.
 */
const CoordinateChart<VIOState> constructVIOChart(
    const CoordinateChart<VIOSensorState>& sensorBundleChart, const CoordinateChart<Landmark>& pointChart);

extern const CoordinateChart<VIOState> VIOChart_euclid;   ///< Euclidean local coordinates for the VIO states
extern const CoordinateChart<VIOState> VIOChart_invdepth; ///< Inverse-depth local coordinates for the VIO states
extern const CoordinateChart<VIOState> VIOChart_normal;   ///< Normal local coordinates for the VIO states

//-----------------------------------
// Coordinate chart differentials
//-----------------------------------

/** @brief Compute the differential of the coordinate transition from Inverse-depth to Euclidean.
 *
 * @param xi0 The origin at which to compute the derivative
 * @return The differential matrix.
 */
const Eigen::MatrixXd coordinateDifferential_invdepth_euclid(const VIOState& xi0);

/** @brief Compute the differential of the coordinate transition from Normal to Euclidean.
 *
 * @param xi0 The origin at which to compute the derivative
 * @return The differential matrix.
 */
const Eigen::MatrixXd coordinateDifferential_normal_euclid(const VIOState& xi0);

//-----------------------------------
// Sphere projection functions and charts
//-----------------------------------

/** @brief Project a point on the sphere stereographically about e3
 *
 * @param eta The point on the sphere to project.
 * @return The stereographic projection of eta about e3.
 */
Eigen::Vector2d e3ProjectSphere(const Eigen::Vector3d& eta);

/** @brief Recover a point on the sphere from its stereographic projection about e3
 *
 * @param y The stereographic projection from which a point should be recovered.
 * @return The recovered point on the sphere.
 */
Eigen::Vector3d e3ProjectSphereInv(const Eigen::Vector2d& y);

/** @brief Compute the differential of the stereographic projection about e3.
 *
 * @param eta The point on the sphere at which the differential is to be computed.
 * @return The differential of the stereographic projection w.r.t. eta.
 */
Eigen::Matrix<double, 2, 3> e3ProjectSphereDiff(const Eigen::Vector3d& eta);

/** @brief Compute the differential of the inverse of stereographic projection about e3.
 *
 * @param y The projected point w.r.t. which the differential is to be computed.
 * @return The differential of the inverse stereographic projection w.r.t. y.
 */
Eigen::Matrix<double, 3, 2> e3ProjectSphereInvDiff(const Eigen::Vector2d& y);

extern const EmbeddedCoordinateChart<3, 2> sphereChart_stereo; ///< Stereographic local coordinates for the sphere
extern const EmbeddedCoordinateChart<3, 2> sphereChart_normal; ///< Normal local coordinates for the sphere