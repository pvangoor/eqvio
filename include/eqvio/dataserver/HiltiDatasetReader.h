#pragma once

#include "eqvio/dataserver/RosbagDatasetReader.h"

/** @brief The HILTI dataset reader.
 *
 * This class can be used to read datasets in the ROSBag format used by the HILTI SLAM challenge. For details, please
 * see <a href="https://www.hilti-challenge.com/">HILTI Challenge</a>
 *
 */
class HiltiDatasetReader : public RosbagDatasetReader {
  protected:
    HiltiDatasetReader() = default;

  public:
    /** @brief Construct a HILTI dataset reader from the given rosbag filename.
     *
     * The only real difference with the rosbag dataserver is that the IMU and image data topics are set to those found
     * in the HILTI dataset.
     */
    HiltiDatasetReader(const std::string& rosbagFileName)
        : RosbagDatasetReader(rosbagFileName, "/alphasense/imu", "/alphasense/cam0/image_raw"){};

    virtual void readCamera(const std::string& cameraFileName) override;
};