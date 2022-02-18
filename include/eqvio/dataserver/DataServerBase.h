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

#include "eqvio/dataserver/DatasetReaderBase.h"

/** @brief The common interface of data servers.
 *
 * This abstraction helps to create a common interface between different data server implementations. A data server
 * provides the visual and inertial measurements in order, and can be queried to find out what the next measurement type
 * is. It is able to do this using data from any dataset reader.
 */
class DataServerBase {
  protected:
    std::unique_ptr<DatasetReaderBase> datasetReaderPtr; ///< The dataset reader providing measurements.

  public:
    /** @brief Read a camera by using the dataset reader
     *
     * @param cameraFileName The name of the file with camera information.
     */
    void readCamera(const std::string& cameraFileName);

    /** @brief Get a pointer to the dataset camera intrinsics*/
    virtual GIFT::GICameraPtr camera() const;
    /** @brief Get a pointer to the dataset camera extrinsics*/
    virtual std::shared_ptr<liepp::SE3d> cameraExtrinsics() const;

    /** @brief Get the type of the next measurement in the dataset.*/
    virtual MeasurementType nextMeasurementType() const = 0;
    /** @brief Get the next image data. */
    virtual StampedImage getImage() = 0;
    /** @brief Get the next IMU data. */
    virtual IMUVelocity getIMU() = 0;
    /** @brief Get the timestamp of the next data, IMU or image.*/
    virtual double nextTime() const = 0;

    /** @brief get a simulated vision measurement at the time of the next real image data.*/
    virtual VisionMeasurement getSimVision();
    /** @brief get a simulated IMU measurement at the time of the next real IMU data.*/
    virtual IMUVelocity getSimIMU();

    /** @brief Construct a data server with a new dataset reader
     *
     * @param datasetReader An rvalue reference to a dataset reader unique_ptr.
     */
    DataServerBase(std::unique_ptr<DatasetReaderBase>&& datasetReader);
    DataServerBase() = default;
};
