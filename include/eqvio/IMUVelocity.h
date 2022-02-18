#pragma once

#include "eigen3/Eigen/Eigen"
#include "eqvio/csv/CSVReader.h"

/** @file */

/// The approximate value of acceleration due to gravity.
constexpr double GRAVITY_CONSTANT = 9.80665;

/** @brief An Inertial Measurement Unit (IMU) reading.
 *
 * IMUVelocity objects contain a gyroscope and accelerometer measurement, along with a timestamp. Additionally, they can
 * also carry bias velocities, although these are typically left as zero.
 */
struct IMUVelocity {
    /// The timestamp of the measurements
    double stamp;

    Eigen::Vector3d gyr;                                  ///< The angular velocity measured by the gyroscope
    Eigen::Vector3d acc;                                  ///< The linear acceleration measured by the accelerometer
    Eigen::Vector3d gyrBiasVel = Eigen::Vector3d::Zero(); ///< The velocity of the gyroscope bias. Usually zero.
    Eigen::Vector3d accBiasVel = Eigen::Vector3d::Zero(); ///< The velocity of the accelerometer bias. Usually zero.

    /// @brief An IMUVelocity with zero acc and gyr.
    static IMUVelocity Zero();

    IMUVelocity() = default;
    /// @brief Construct the IMU velocity from a vector containing (gyr, acc)
    IMUVelocity(const Eigen::Matrix<double, 6, 1>& vec);
    /// @brief Construct the IMU velocity from a vector containing (gyr, acc, gBiasVel, aBiasVel)
    IMUVelocity(const Eigen::Matrix<double, 12, 1>& vec);

    /** @brief add an IMU velocity to this one.
     *
     * @param other The other velocity to add to this one.
     * @return the sum of IMU velocities.
     *
     * @note The stamp is taken from whichever summand has a positive stamp. The left summand is preferred.
     */
    IMUVelocity operator+(const IMUVelocity& other) const;
    /// @brief add anything by converting it to an IMU velocity
    IMUVelocity operator+(const auto& other) const { return this->operator+(IMUVelocity(other)); }

    /** @brief subtract a vector from the velocity components
     *
     * @param vec The vector to subtract, assumed to be of the form (gyr, acc, gBiasVel, aBiasVel)
     * @return the resulting IMU velocity.
     */
    IMUVelocity operator-(const Eigen::Matrix<double, 12, 1>& vec) const;
    /** @brief subtract a vector from the velocity components
     *
     * @param vec The vector to subtract, assumed to be of the form (gyr, acc)
     * @return the resulting IMU velocity. The gBiasVel and aBiasVel are left unchanged.
     */
    IMUVelocity operator-(const Eigen::Matrix<double, 6, 1>& vec) const;

    /** @brief scale the IMU velocity by a constant.
     *
     * @param c The parameter by which to scale.
     * @return An IMU velocity with each velocity component multiplied by c.
     */
    IMUVelocity operator*(const double& c) const;

    /// The dimension of the IMU velocity vector space.
    constexpr static int CompDim = 12;
};

/** @brief Write an IMU velocity to a CSV line
 *
 * @param line A reference to the CSV line where the IMU velocity should be written.
 * @param imu The IMU data to write to the CSV line.
 * @return A reference to the CSV line with the IMU velocity written.
 *
 * The data is formatted in the CSV line as stamp, gyr_x, gyr_y, gyr_z, acc_x, acc_y, acc_z
 */
CSVLine& operator<<(CSVLine& line, const IMUVelocity& imu);

/** @brief Read an IMU velocity from a CSV line
 *
 * @param line A reference to the CSV line from which the IMU velocity should be read.
 * @param imu A reference to an IMU velocity where data is to be stored.
 * @return A reference to the CSV line with the IMU velocity removed during reading.
 *
 * The data is formatted in the CSV line as stamp, gyr_x, gyr_y, gyr_z, acc_x, acc_y, acc_z
 */
CSVLine& operator>>(CSVLine& line, IMUVelocity& imu);