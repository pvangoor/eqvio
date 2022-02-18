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

#include "eqvio/EqFMatrices.h"
#include "eqvio/IMUVelocity.h"
#include "eqvio/VIOGroup.h"
#include "eqvio/VIOState.h"
#include "eqvio/VisionMeasurement.h"
#include "eqvio/csv/CSVReader.h"

/** @brief The implementation of the EqF for VIO.
 *
 * This class implements the equivariant filter for visual inertial odometry. Aside from the basic EqF dynamics, this
 * implementation also features methods for outlier rejection, state augmentation, and buffering IMU measurements.
 */
class VIOFilter {
  protected:
    EqFCoordinateSuite const* coordinateSuite = &EqFCoordinateSuite_euclid; ///< The suite of EqF matrix functions.

    VIOState xi0;                      ///< The fixed origin configuration.
    VIOGroup X = VIOGroup::Identity(); ///< The EqF's observer state
    Eigen::MatrixXd Sigma =
        Eigen::MatrixXd::Identity(VIOSensorState::CompDim, VIOSensorState::CompDim); ///< The EqF's Riccati matrix

    bool initialisedFlag = false; ///< True once the EqF has been initialised from an IMU velocity measurement.
    double currentTime = -1;      ///< The time of the last measurement processed.
    IMUVelocity currentVelocity = IMUVelocity::Zero(); ///< The latest IMU velocity measurement

    IMUVelocity accumulatedVelocity =
        IMUVelocity::Zero();      ///< The sum of all IMU velocities since the last image measurement.
    double accumulatedTime = 0.0; ///< The time since the last image measurement.

    /** @brief Integrate the EqF dynamics to the given time without correction terms.
     *
     * @param newTime The time to which the EqF states should be integrated.
     * @param doRiccati Propagate the Riccati matrix iff this is true.
     * @return True iff the integration could be performed.
     *
     * If doRiccati is set to true, then the Riccati matrix is propagated without the correction terms. This is an
     * expensive operation, however, so it is generally better just to propagate the Riccati upon receiving a new image
     * measurement by using the average IMU velocity. If newTime is less than the current filter time or the filter is
     * not yet initialised, then the integration cannot go ahead and this method returns false.
     */
    bool integrateUpToTime(const double& newTime, const bool doRiccati = true);

    /** @brief Add new landmarks to the EqF state from the provided features.
     *
     * @param measurement The features to use when adding new landmarks.
     *
     * If there are any features with id numbers that are not already part of the state, then new landmark states are
     * added to the state from those features. Depending on the chosen settings, these landmarks are initialised with
     * either the median scene depth or a fixed depth value.
     */
    void addNewLandmarks(const VisionMeasurement& measurement);

    /** @brief Add new landmarks to the EqF state from the provided features.
     *
     * @param newLandmarks The landmarks to be added to the state.
     *
     * This method adds new landmarks to the state exactly as they are provided, and augments the Riccati matrix as
     * appropriate.
     */
    void addNewLandmarks(std::vector<Landmark>& newLandmarks);

    /** @brief Removes all landmarks with id numbers that do not appear in measurementIds
     *
     * @param measurementIds The ids of landmarks that will be kept.
     */
    void removeOldLandmarks(const std::vector<int>& measurementIds);

    /** @brief Identify landmarks with outlier measurements and remove them from the EqF states and the measurement,
     * respectively.
     *
     * @param measurement The current measurement.
     *
     * This method uses both an absolute check and a covariance-based check. If a measurement differs from the expected
     * measurement by more than a set value, then it is considered an outlier. If weighted measurement distance is
     * larger than a set value, it is also considered an outlier.
     */
    void removeOutliers(VisionMeasurement& measurement);

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

    /** @brief Compute the median of the depth values of all the landmarks.
     */
    double getMedianSceneDepth() const;

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

  public:
    struct Settings;
    std::unique_ptr<VIOFilter::Settings>
        settings; ///< The filter settings, including noise parameters and discretisation settings.

    VIOFilter() = default;

    /** @brief Create the EqF using the given settings.
     *
     * @param settings The initial settings to use.
     */
    VIOFilter(const VIOFilter::Settings& settings);

    /** @brief Initialise the EqF states using an IMU velocity measurement.
     *
     * @param imuVelocity The IMU velocity to use in initialisation.
     *
     * It is assumed that the IMU acceleration is almost entirely due to gravity, allowing us to determine the pitch and
     * roll components of the robot's orientation. It is important to do this at the start.
     */
    void initialiseFromIMUData(const IMUVelocity& imuVelocity);

    /** @brief Reset the EqF using the given state xi.
     *
     * @param xi The state to which the EqF should be set.
     *
     * Given a state xi, the origin xi0 is set to xi, the state X is set to identity, and the Riccati matrix Sigma is
     * set to its default values.
     */
    void setState(const VIOState& xi);

    //-------------------------
    // Input
    //-------------------------

    /** @brief Process an IMU velocity measurement
     *
     * @param imuVelocity The IMU velocity to process.
     *
     * The EqF is integrated using the given IMU velocity measurement. Depending on the settings, the Riccati matrix may
     * or may not be propagated at this time. The given IMU velocity is recorded as the current velocity.
     */
    void processIMUData(const IMUVelocity& imuVelocity);

    /** @brief Process a vision measurement
     *
     * @param measurement The measurement of feature points obtained from the front-end.
     *
     * A number of steps are undertaken when processing vision data. First, the EqF is integrated using the latest
     * velocity, and the Riccati matrix is propagated to the current time. Second, any landmarks that were not tracked
     * successfully, or those that are associated with outlier measurements, are removed from the EqF state. Third and
     * final, the EqF correction terms are computed and applied.
     */
    void processVisionData(const VisionMeasurement& measurement);

    //-------------------------
    // Output
    //-------------------------
    /** @brief Get the current filter time.
     */
    double getTime() const;

    /** @brief Return true once the filter is initialised from and IMU velocity.
     */
    bool isInitialised() const { return initialisedFlag; };

    /** @brief Provide estimated feature measurements.
     *
     * @param camPtr The camera model used to compute the predictions
     * @param stamp The requested time stamp of the predictions. Defaults to the current filter time.
     *
     * Compute the predicted feature measurements based on the provided camera model. If a time stamp is provided, then
     * the filter is first integrated up to this time. This method can be helpful in aiding feature tracking under rapid
     * motions.
     */
    VisionMeasurement getFeaturePredictions(const GIFT::GICameraPtr& camPtr, const double& stamp = -1);

    /** @brief Provide the current system state estimate.
     *
     * This is computed by applying the group action with the observer state X to the fixed origin configuration xi0.
     */
    VIOState stateEstimate() const;

    /** @brief Write an the filter states to a file.
     *
     * @param line A reference to the CSV line where the data should be written.
     * @param filter The filter states to write to the CSV line.
     * @return A reference to the CSV line with the data written.
     *
     * The data is formatted in the CSV line as [xi0, X, Sigma]
     */
    friend CSVLine& operator<<(CSVLine& line, const VIOFilter& filter);
};

CSVLine& operator<<(CSVLine& line, const VIOFilter& filter);