#pragma once

#include "eqvio/dataserver/DatasetReaderBase.h"
#include "rosbag/bag.h"
#include "rosbag/view.h"

/** @brief The ROSBag dataset reader.
 *
 * This class can be used to read datasets in the ROSBag format.
 */
class RosbagDatasetReader : public DatasetReaderBase {
  protected:
    rosbag::Bag bag;                    ///< The rosbag containing the visual and inertial data.
    rosbag::View imuBagView;            ///< A rosbag view of IMU data
    rosbag::View::iterator imuViewIt;   ///< The IMU data view iterator
    rosbag::View imageBagView;          ///< A rosbag view of image data
    rosbag::View::iterator imageViewIt; ///< The image data view iterator

    RosbagDatasetReader() = default;

  public:
    /** @brief construct a rosbag dataset reader with the given IMU and Image topic. */
    RosbagDatasetReader(
        const std::string& rosbagFileName, const std::string& imuTopic = "/imu0",
        const std::string& imageTopic = "/cam0/image_raw");

    virtual std::unique_ptr<StampedImage> nextImage() override;
    virtual std::unique_ptr<IMUVelocity> nextIMU() override;
    virtual void readCamera(const std::string& cameraFileName) override;
};