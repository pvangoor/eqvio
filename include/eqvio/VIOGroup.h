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

#include "eqvio/VIOState.h"
#include "eqvio/csv/CSVReader.h"
#include "liepp/SE3.h"
#include "liepp/SOT3.h"

/** @file */

/** @brief The Lie group used for VIO.
 *
 * This is the Lie group used for VIO in the EqF. Note that the dimension is variable since there may be an arbitrary
 * number of SOT(3) elements.
 */
struct VIOGroup {
    Eigen::Matrix<double, 6, 1> beta; ///< The symmetry component related to IMU biases
    liepp::SE3d A;                    ///< The symmetry component related to robot pose
    Eigen::Vector3d w;                ///< The symmetry component related to robot velocity
    liepp::SE3d B;                    ///< The symmetry component related to camera offset
    std::vector<liepp::SOT3d> Q;      ///< The symmetry components related to landmark positions
    std::vector<int> id;              ///< The id numbers associated with the SOT(3) components

    /** @brief The group multiplication operator
     *
     * @param other The right factor in the group product.
     * @return The result of multiplication
     *
     * The order of multiplication is very important here. It is given by:
     * result = (*this) * (other)
     */
    [[nodiscard]] VIOGroup operator*(const VIOGroup& other) const;

    /** @brief Create the group identity from the ids
     *
     * @param id The id numbers required in the group identity.
     * @return The group identity with the provided id numbers
     */
    [[nodiscard]] static VIOGroup Identity(const std::vector<int>& id = {});

    /** @brief The group inverse operator
     *
     * @return The inverse of the group element used to call this method.
     */
    [[nodiscard]] VIOGroup inverse() const;

    /** @brief Check if any components of the group are NaN.
     *
     * @return true if any components have NaN values.
     *
     * This is primarily useful for debugging.
     */
    [[nodiscard]] bool hasNaN() const;
};

/** @brief Write a group element to a CSV line
 *
 * @param line A reference to the CSV line where the data should be written.
 * @param X The group element to write to the CSV line.
 * @return A reference to the CSV line with the data written.
 */
CSVLine& operator<<(CSVLine& line, const VIOGroup& X);

/** @brief The Lie algebra of the VIOGroup
 *
 * This is the Lie algebra associated with the Lie group used for VIO in the EqF. Note that the dimension is variable
 * since there may be an arbitrary number of sot(3) elements.
 */
struct VIOAlgebra {
    Eigen::Matrix<double, 6, 1> u_beta; ///< The algebra component related to IMU biases
    liepp::se3d U_A;                    ///< The algebra component related to robot pose
    liepp::se3d U_B;                    ///< The algebra component related to robot velocity
    Eigen::Vector3d u_w;                ///< The algebra component related to camera offset
    std::vector<Eigen::Vector4d> W;     ///< The algebra components related to landmark positions
    std::vector<int> id;                ///< The id numbers associated with the sot(3) components

    /** @brief Scale an algebra element by a constant
     *
     * @param c The scaling factor
     * @return The result of scaling
     */
    [[nodiscard]] VIOAlgebra operator*(const double& c) const;

    /** @brief Get the negative version of an algebra element.
     *
     * @return The negative version of *this with the same id numbers
     */
    [[nodiscard]] VIOAlgebra operator-() const;

    /** @brief Add another Lie algebra element
     *
     * @param other the Lie algebra element to add.
     * @return The sum of this with other as vectors.
     */
    [[nodiscard]] VIOAlgebra operator+(const VIOAlgebra& other) const;

    /** @brief Subtract another Lie algebra element
     *
     * @param other the Lie algebra element to subtract.
     * @return The difference of this with other as vectors.
     */
    [[nodiscard]] VIOAlgebra operator-(const VIOAlgebra& other) const;
};

/** @brief Scale an algebra element by a constant
 *
 * @param c The scaling factor
 * @param lambda The Lie algebra element to be scaled.
 * @return The result of scaling
 */
[[nodiscard]] VIOAlgebra operator*(const double& c, const VIOAlgebra& lambda);

/** @brief The group action of the VIO group on the sensor states.
 *
 * @param X The VIO group element
 * @param sensor The sensor states
 * @return The result of applying the right group action of X on sensor.
 */
[[nodiscard]] VIOSensorState sensorStateGroupAction(const VIOGroup& X, const VIOSensorState& sensor);

/** @brief The group action of the VIO group on the a VIO system state.
 *
 * @param X The VIO group element
 * @param state The VIO system state
 * @return The transformed VIO system state.
 */
[[nodiscard]] VIOState stateGroupAction(const VIOGroup& X, const VIOState& state);

/** @brief The group action of the VIO group on the a visual output measurements.
 *
 * @param X The VIO group element.
 * @param measurement The feature measurements.
 * @return The transformed measurements.
 *
 * Note that this action differs from the one presented in the paper as it takes into account an arbitrary camera model.
 * Importantly, this group action is compatible with the VIO state group action, such that the measurement function is
 * equivariant.
 */
[[nodiscard]] VisionMeasurement outputGroupAction(const VIOGroup& X, const VisionMeasurement& measurement);

/** @brief Lift the VIO system velocity from the state space to the VIO Lie algebra
 *
 * @param state The VIO state at which to lift.
 * @param velocity The IMU velocity to be lifted.
 * @return The lifted velocity on the VIO Lie algebra.
 */
[[nodiscard]] VIOAlgebra liftVelocity(const VIOState& state, const IMUVelocity& velocity);

/** @brief A discrete version of the VIO system lift.
 *
 * @param state The VIO state at which to lift.
 * @param velocity The IMU velocity to be lifted.
 * @param dt The time period over which to compute the discrete lift.
 * @return The lifted velocity on the VIO Lie algebra.
 */
[[nodiscard]] VIOGroup liftVelocityDiscrete(const VIOState& state, const IMUVelocity& velocity, const double& dt);

/** @brief The Lie theoretic exponential map from the VIO Lie algebra to the VIO group.
 *
 * @param lambda The algebra element to which the exponential should be applied.
 * @return The resulting group element.
 */
[[nodiscard]] VIOGroup VIOExp(const VIOAlgebra& lambda);