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

#include "eqvio/dataserver/ASLDatasetReader.h"
#include "yaml-cpp/yaml.h"
#include <filesystem>

ASLDatasetReader::ASLDatasetReader(const std::string& datasetMainDir) {
    // Set up required structures
    // --------------------------

    // Set up the data iterator for IMU
    IMUCSVFile = CSVFile(datasetMainDir + "mav0/imu0/" + "data.csv");
    IMUCSVFile.nextLine(); // skip the header

    // Read the image data file
    cam_dir = datasetMainDir + "mav0/cam0/";
    ImageCSVFile = CSVFile(cam_dir + "data.csv");
    ImageCSVFile.nextLine(); // skip the header

    // Get the ground truth
    groundtruthFileName = datasetMainDir + "mav0/state_groundtruth_estimate0/data.csv";

    // Read the camera file
    std::string cameraFileName = cam_dir + "sensor.yaml";
    readCamera(cameraFileName);
}

std::unique_ptr<IMUVelocity> ASLDatasetReader::nextIMU() {
    if (!IMUCSVFile) {
        return nullptr;
    }

    CSVLine imuLine = IMUCSVFile.nextLine();
    IMUVelocity temp;
    imuLine >> temp;
    temp.stamp *= 1e-9;
    return std::make_unique<IMUVelocity>(temp);
}

std::unique_ptr<StampedImage> ASLDatasetReader::nextImage() {
    if (!ImageCSVFile) {
        return nullptr;
    }

    CSVLine imageLine = ImageCSVFile.nextLine();
    std::string nextImageFname;
    double rawStamp;
    imageLine >> rawStamp >> nextImageFname;
    nextImageFname = cam_dir + "data/" + nextImageFname;

    if (*nextImageFname.rbegin() == '\r') {
        nextImageFname.erase(nextImageFname.end() - 1);
    }

    StampedImage temp;
    temp.stamp = 1e-9 * rawStamp - cameraLag;
    temp.image = cv::imread(nextImageFname);
    return std::make_unique<StampedImage>(temp);
}

void ASLDatasetReader::readCamera(const std::string& cameraFileName) {
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

    // Read the extrinsics
    const std::vector<double> extrinsics_entries = cameraFileNode["T_BS"]["data"].as<std::vector<double>>();
    Eigen::Matrix4d extrinsics_matrix(extrinsics_entries.data());
    // The data is in row-major form, but eigen uses column-major by default.
    extrinsics_matrix.transposeInPlace();
    cameraExtrinsics = std::make_unique<liepp::SE3d>(extrinsics_matrix);
}

std::vector<StampedPose> ASLDatasetReader::groundtruth() {
    // Find the poses file
    assert(std::filesystem::exists(groundtruthFileName));
    std::ifstream poseFile = std::ifstream(groundtruthFileName);
    CSVReader poseFileIter(poseFile, ',');
    ++poseFileIter; // skip header

    std::vector<StampedPose> poses;

    double prevPoseTime = -1e8;
    for (CSVLine row : poseFileIter) {
        StampedPose pose;
        row >> pose.t >> pose.pose;
        pose.t *= 1.0e-9;
        if (pose.t > prevPoseTime + 1e-8) {
            // Avoid poses with the same timestamp.
            poses.emplace_back(pose);
            prevPoseTime = pose.t;
        }
    }

    return poses;
}
