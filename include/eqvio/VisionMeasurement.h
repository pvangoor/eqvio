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

#include "Geometry.h"
#include "eqvio/csv/CSVReader.h"

#include "GIFT/camera/camera.h"
#include "eigen3/Eigen/Eigen"

#include <map>
#include <memory>

/** @brief A measurement of features from an image.
 *
 * This is the VIO system measurement, along with a time stamp and a pointer to a camera model used to generate the
 * measurement. The camera model is important, as it provides all the necessary information to replicate the measurement
 * function in the VIO system.
 */
struct VisionMeasurement {
    double stamp;                                  ///< The time stamp of the image features.
    std::map<int, Eigen::Vector2d> camCoordinates; ///< The pixel coordinates of the features, by id number.
    GIFT::GICameraPtr cameraPtr; ///< A pointer to the camera model associated with the image features.

    constexpr static int CompDim = Eigen::Dynamic; ///< The dimension of the space of feature measurements.

    /** @brief get the id numbers associated with this measurement.
     *
     * @return All the id numbers present in the camCoordinates map.
     */
    std::vector<int> getIds() const;

    /** @brief cast the measurement to an Eigen Vector
     *
     * @return The cam coordinates as a vector
     *
     * The order in which feature coordinates are listed in the vector depends on their ordering in the camCoordinates
     * map. In principle, this means they will be in ascending order of id numbers.
     */
    operator Eigen::VectorXd() const;

    /** @brief cast the feature coordinates to OpenCV Point2f
     *
     * @return The map of camera coordinates as Point2f
     */
    std::map<int, cv::Point2f> ocvCoordinates() const;
};

/** @brief Compute the difference between two vision measurements
 *
 * @param y1 The left-hand term in the subtraction
 * @param y2 The right-hand term in the subtraction
 * @return A vision measurement with the difference between each feature in both y1 and y2
 *
 * It is expected that y1 and y2 share the same camera pointer. This method is primarily useful for subtracting the
 * expected measurement from the true measurement in the EqF framework.
 */
VisionMeasurement operator-(const VisionMeasurement& y1, const VisionMeasurement& y2);

/** @brief Write a vision measurement to a CSV line
 *
 * @param line A reference to the CSV line where the data should be written.
 * @param vision The vision measurement data to write to the CSV line.
 * @return A reference to the CSV line with the data written.
 */
CSVLine& operator<<(CSVLine& line, const VisionMeasurement& vision);

/** @brief Read an IMU velocity from a CSV line
 *
 * @param line A reference to the CSV line from which the data should be read.
 * @param vision A reference to a vision measurement where data is to be stored.
 * @return A reference to the CSV line with the data removed during reading.
 */
CSVLine& operator>>(CSVLine& line, VisionMeasurement& vision);
