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

#include "eqvio/dataserver/RosbagDatasetReader.h"
#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/Imu.h"

#include <filesystem>

static IMUVelocity msgToIMU(const sensor_msgs::ImuPtr& imuMsg) {
    assert(imuMsg != nullptr);
    IMUVelocity data;
    data.stamp = imuMsg->header.stamp.toSec();
    data.gyr << imuMsg->angular_velocity.x, imuMsg->angular_velocity.y, imuMsg->angular_velocity.z;
    data.acc << imuMsg->linear_acceleration.x, imuMsg->linear_acceleration.y, imuMsg->linear_acceleration.z;
    return data;
}

static StampedImage msgToImage(const sensor_msgs::ImageConstPtr& imageMsg) {
    assert(imageMsg != nullptr);
    cv_bridge::CvImagePtr cvImageMsg = cv_bridge::toCvCopy(imageMsg);
    StampedImage data;
    data.stamp = cvImageMsg->header.stamp.toSec();
    data.image = cvImageMsg->image;
    return data;
}

RosbagDatasetReader::RosbagDatasetReader(
    const std::string& rosbagFileName, const std::string& imuTopic, const std::string& imageTopic) {

    // Read the rosbag
    bag.open(rosbagFileName);

    imuBagView.addQuery(bag, rosbag::TopicQuery(imuTopic));
    imageBagView.addQuery(bag, rosbag::TopicQuery(imageTopic));

    imuViewIt = imuBagView.begin();
    imageViewIt = imageBagView.begin();

    // Try to load the camera
    const std::string datasetDir = rosbagFileName.substr(0, rosbagFileName.rfind("/")) + "/";
    const std::string cameraFileName = datasetDir + "intrinsics.yaml";
    if (!std::filesystem::exists(cameraFileName)) {
        std::cout << "No camera intrinsics were not found at\n" << cameraFileName << std::endl;
    } else {
        readCamera(cameraFileName);
    }
}

void RosbagDatasetReader::readCamera(const std::string& cameraFileName) {
    YAML::Node cameraFileNode = YAML::LoadFile(cameraFileName);

    // Read the intrinsics
    cv::Size imageSize;
    imageSize.width = cameraFileNode["resolution"][0].as<int>();
    imageSize.height = cameraFileNode["resolution"][1].as<int>();

    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = cameraFileNode["intrinsics"][0].as<double>();
    K.at<double>(1, 1) = cameraFileNode["intrinsics"][1].as<double>();
    K.at<double>(0, 2) = cameraFileNode["intrinsics"][2].as<double>();
    K.at<double>(1, 2) = cameraFileNode["intrinsics"][3].as<double>();

    std::vector<double> distortion;
    distortion = cameraFileNode["distortion_coefficients"].as<std::vector<double>>();

    GIFT::StandardCamera stdCamera = GIFT::StandardCamera(imageSize, K, distortion);
    this->camera = std::make_shared<GIFT::StandardCamera>(stdCamera);
    // camera = std::make_shared<GIFT::StandardCamera>(GIFT::StandardCamera(cv::String(cameraFileName)));

    // Read the extrinsics
    const std::vector<double> extrinsics_entries = cameraFileNode["T_BS"]["data"].as<std::vector<double>>();
    Eigen::Matrix4d extrinsics_matrix(extrinsics_entries.data());
    // The data is in row-major form, but eigen uses column-major by default.
    extrinsics_matrix.transposeInPlace();
    cameraExtrinsics = std::make_unique<liepp::SE3d>(extrinsics_matrix);

}

std::unique_ptr<StampedImage> RosbagDatasetReader::nextImage() {
    if (imageViewIt == imageBagView.end()) {
        return nullptr;
    }
    StampedImage currentImage = msgToImage(imageViewIt->instantiate<sensor_msgs::Image>());
    ++imageViewIt;
    return std::make_unique<StampedImage>(currentImage);
}

std::unique_ptr<IMUVelocity> RosbagDatasetReader::nextIMU() {
    if (imuViewIt == imuBagView.end()) {
        return nullptr;
    }
    IMUVelocity currentIMU = msgToIMU(imuViewIt->instantiate<sensor_msgs::Imu>());
    ++imuViewIt;
    return std::make_unique<IMUVelocity>(currentIMU);
}
