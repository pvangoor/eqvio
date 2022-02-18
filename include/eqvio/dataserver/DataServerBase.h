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
