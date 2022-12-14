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

#include "eqvio/dataserver/UZHFPVDatasetReader.h"
#include "yaml-cpp/yaml.h"
#include <filesystem>
#include <iostream>

UZHFPVDatasetReader::UZHFPVDatasetReader(const std::string& datasetMainDir) {
    // Set up required structures
    // --------------------------

    baseDir = datasetMainDir;

    // Set up the data iterator for IMU
    IMUCSVFile = CSVFile(datasetMainDir + "imu.txt", ' ');
    IMUCSVFile.nextLine(); // skip the header

    // Read the image data file
    ImageCSVFile = CSVFile(datasetMainDir + "left_images.txt", ' ');
    ImageCSVFile.nextLine(); // skip the header

    // Read the camera file
    std::string cameraFileName =
        baseDir + "../indoor_forward_calib_snapdragon/camchain-imucam-..indoor_forward_calib_snapdragon_imu.yaml";
    if (std::filesystem::exists(cameraFileName)) {
        readCamera(cameraFileName);
    } else {
        std::cout << "Camera file was not found at:\n" << cameraFileName << std::endl;
        throw std::runtime_error("The camera file could not be found.");
    }
}

std::unique_ptr<IMUVelocity> UZHFPVDatasetReader::nextIMU() {
    if (!IMUCSVFile) {
        return nullptr;
    }

    CSVLine imuLine = IMUCSVFile.nextLine();
    int num;
    IMUVelocity temp;
    imuLine >> num >> temp;
    return std::make_unique<IMUVelocity>(temp);
}

std::unique_ptr<StampedImage> UZHFPVDatasetReader::nextImage() {
    if (!ImageCSVFile) {
        return nullptr;
    }
    StampedImage temp;

    CSVLine imageLine = ImageCSVFile.nextLine();
    std::string nextImageFname;
    double rawStamp;
    int num;
    imageLine >> num >> rawStamp >> nextImageFname;

    nextImageFname = baseDir + nextImageFname;
    if (*nextImageFname.rbegin() == '\r') {
        nextImageFname.erase(nextImageFname.end() - 1);
    }

    temp.stamp = rawStamp - cameraLag;
    temp.image = cv::imread(nextImageFname);

    return std::make_unique<StampedImage>(temp);
}

void UZHFPVDatasetReader::readCamera(const std::string& cameraFileName) {
    YAML::Node cameraFileNode = YAML::LoadFile(cameraFileName);

    // Read the camera intrinsics
    cameraFileNode = cameraFileNode["cam0"];

    cv::Size imageSize;
    imageSize.width = cameraFileNode["resolution"][0].as<int>();
    imageSize.height = cameraFileNode["resolution"][1].as<int>();

    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = cameraFileNode["intrinsics"][0].as<double>();
    K.at<double>(1, 1) = cameraFileNode["intrinsics"][1].as<double>();
    K.at<double>(0, 2) = cameraFileNode["intrinsics"][2].as<double>();
    K.at<double>(1, 2) = cameraFileNode["intrinsics"][3].as<double>();

    std::array<double, 4> distortion;
    distortion = cameraFileNode["distortion_coeffs"].as<std::array<double, 4>>();

    camera = std::make_shared<GIFT::EquidistantCamera>(imageSize, K, distortion);

    // Read the camera extrinsics
    const std::vector<std::vector<double>> extrinsics_rows =
        cameraFileNode["T_cam_imu"].as<std::vector<std::vector<double>>>();
    Eigen::Matrix4d extrinsics_matrix;
    assert(extrinsics_rows.size() == 4);
    for (int i = 0; i < 4; ++i) {
        assert(extrinsics_rows[i].size() == 4);
        extrinsics_matrix.row(i) = Eigen::Vector4d(extrinsics_rows[i].data()).transpose();
    }
    // The UZH FPV dataset reports the pose of the IMU w.r.t. the camera, so it needs to be inverted.
    cameraExtrinsics = std::make_unique<liepp::SE3d>(extrinsics_matrix.inverse());
}

std::vector<StampedPose> UZHFPVDatasetReader::groundtruth() {
    const std::string groundtruthFileName = baseDir + "groundtruth.txt";

    assert(std::filesystem::exists(groundtruthFileName));
    std::ifstream poseFile = std::ifstream(groundtruthFileName);
    CSVReader poseFileIter(poseFile, ' ');
    ++poseFileIter; // skip header

    std::vector<StampedPose> poses;

    double prevPoseTime = -1e8;
    for (CSVLine row : poseFileIter) {

        StampedPose pose;
        row >> pose.t >> pose.pose;
        if (pose.t > prevPoseTime + 1e-8) {
            // Avoid poses with the same timestamp.
            poses.emplace_back(pose);
            prevPoseTime = pose.t;
        }
    }

    return poses;
}