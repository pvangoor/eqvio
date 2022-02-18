#pragma once

#include "eqvio/dataserver/DataServerBase.h"

/** @brief The simplest implementation of a data server
 *
 * This data server reads the next image and IMU data, and then passes them on as requested. When the image data is
 * requested, the currently stored data is provided and the next data is read in anticipation of the next request.
 */
class SimpleDataServer : public DataServerBase {
  protected:
    std::unique_ptr<StampedImage> nextImageData; ///< The next image measurement
    std::unique_ptr<IMUVelocity> nextIMUData;    ///< The next IMU measurement

  public:
    virtual MeasurementType nextMeasurementType() const override;
    virtual StampedImage getImage() override;
    virtual IMUVelocity getIMU() override;
    virtual double nextTime() const override;

    /** @brief construct the simple data server from a dataset reader.
     *
     * @param datasetReader An rvalue reference to a dataset reader unique_ptr.
     */
    SimpleDataServer(std::unique_ptr<DatasetReaderBase>&& datasetReader);
};
