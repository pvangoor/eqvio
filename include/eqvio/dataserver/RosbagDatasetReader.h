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