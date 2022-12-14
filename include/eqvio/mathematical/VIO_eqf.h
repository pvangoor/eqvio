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

#include <memory>
#include <ostream>

#include "eqvio/csv/CSVReader.h"
#include "eqvio/mathematical/EqFMatrices.h"
#include "eqvio/mathematical/IMUVelocity.h"
#include "eqvio/mathematical/VIOGroup.h"
#include "eqvio/mathematical/VIOState.h"
#include "eqvio/mathematical/VisionMeasurement.h"

/** @brief The implementation of the EqF for VIO.
 *
 * This struct implements the basic equivariant filter for visual inertial odometry.
 */
struct VIO_eqf {

    EqFCoordinateSuite const* coordinateSuite = &EqFCoordinateSuite_euclid; ///< The suite of EqF matrix functions.
    VIOState xi0;                                                           ///< The fixed origin configuration.
    VIOGroup X = VIOGroup::Identity();                                      ///< The EqF's observer state
    Eigen::MatrixXd Sigma =
        Eigen::MatrixXd::Identity(VIOSensorState::CompDim, VIOSensorState::CompDim); ///< The EqF's Riccati matrix

    double currentTime = -1; ///< The time of the filter states.

    /** @brief Add new landmarks to the EqF state from the provided vector.
     *
     * @param newLandmarks The landmarks to be added to the state.
     * @param newLandmarkCov The covariance of the new landmarks to be added.
     *
     * This method adds new landmarks to the state exactly as they are provided, and augments the Riccati matrix as
     * appropriate.
     */
    void addNewLandmarks(std::vector<Landmark>& newLandmarks, const Eigen::MatrixXd& newLandmarkCov);

    /** @brief Remove a landmark from the EqF states based on its index in the EqF state vector.
     *
     * @param idx The index of the landmark to remove.
     */
    void removeLandmarkByIndex(const int& idx);

    /** @brief Remove a landmark from the EqF states based on its id number.
     *
     * @param id The id number of the landmark to remove.
     */
    void removeLandmarkById(const int& id);

    /** @brief Remove all landmarks with depth values that are too small or too large.
     */
    void removeInvalidLandmarks();

    /** @brief Get the marginalised covariance associated with a landmark by its id number
     *
     * @param id The id number of the landmark with the desired covariance.
     */
    Eigen::Matrix3d getLandmarkCovById(const int& id) const;

    /** @brief Get the covariance associated with the measurement of a particular value.
     *
     * @param id The id number to which the measurement covariance is associated.
     * @param y The real measurement of pixel coordinates associated with this id number.
     * @param camPtr The camera model for the camera used to make the measurement y.
     *
     * The availability of the true measurement y allows us to use an equivariant output approximation here.
     */
    Eigen::Matrix2d getOutputCovById(const int& id, const Eigen::Vector2d& y, const GIFT::GICameraPtr& camPtr) const;

    void integrateObserverState(const IMUVelocity& imuVelocity, const double& dt, const bool& discreteLift = true);
    void integrateRiccatiStateFast(
        const IMUVelocity& imuVelocity, const double& dt,
        const Eigen::Matrix<double, IMUVelocity::CompDim, IMUVelocity::CompDim>& inputGainMatrix,
        const Eigen::MatrixXd& stateGainMatrix);
    void integrateRiccatiStateAccurate(
        const IMUVelocity& imuVelocity, const double& dt,
        const Eigen::Matrix<double, IMUVelocity::CompDim, IMUVelocity::CompDim>& inputGainMatrix,
        const Eigen::MatrixXd& stateGainMatrix);
    void integrateRiccatiStateDiscrete(
        const IMUVelocity& imuVelocity, const double& dt,
        const Eigen::Matrix<double, IMUVelocity::CompDim, IMUVelocity::CompDim>& inputGainMatrix,
        const Eigen::MatrixXd& stateGainMatrix);

    /** @brief Process a sequence of IMU measurements
     *
     * @param measurement The measurements to use for the update.
     * @param outputGainMatrix The covariance associated with uncertainty in the measurements.
     * @param useEquivariantOutput Set true to use the equivariant output approximation (recommended true).
     * @param discreteCorrection Set true to use a discrete-time correction (recommended false).
     */
    void performVisionUpdate(
        const VisionMeasurement& measurement, const Eigen::MatrixXd& outputGainMatrix,
        const bool& useEquivariantOutput = true, const bool& discreteCorrection = false);

    /** @brief Provide the current state estimate.
     *
     * This is computed by applying the group action with the observer state X to the fixed origin configuration xi0.
     */
    VIOState stateEstimate() const;

    /** @brief Provide a prediction of the state at the given time stamp.
     *
     * This is calculated by computing the current state estimate and then integrating the dynamics.
     */
    VIOState predictState(const double& stamp, const std::vector<IMUVelocity> imuVelocities) const;

    double computeNEES(const VIOState& trueState) const;

    /** @brief Write an the filter states to a file.
     *
     * @param line A reference to the CSV line where the data should be written.
     * @param filter The filter states to write to the CSV line.
     * @return A reference to the CSV line with the data written.
     *
     * The data is formatted in the CSV line as [xi0, X, Sigma]
     */
    friend CSVLine& operator<<(CSVLine& line, const VIO_eqf& filter);
};

CSVLine& operator<<(CSVLine& line, const VIO_eqf& filter);