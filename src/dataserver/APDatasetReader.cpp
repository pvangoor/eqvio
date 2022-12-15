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

#include "eqvio/dataserver/APDatasetReader.h"

#include <filesystem>

APDatasetReader::APDatasetReader(const std::string& datasetFileName) {

    const std::string datasetDir = datasetFileName.substr(0, datasetFileName.rfind("/")) + "/";

    // Set up required structures
    // --------------------------

    // Set up the data iterator for IMU
    IMUCSVFile = CSVFile(datasetDir + "mav_imu.csv");
    IMUCSVFile.nextLine(); // skip the header

    // Read the image data
    cam_dir = datasetDir + "frames/";
    ImageCSVFile = CSVFile(datasetDir + "cam.csv");
    ImageCSVFile.nextLine(); // skip the header

    const std::string groundtruthFileName = datasetDir + "ground_truth.csv";

    // Read the camera file
    readCamera(datasetDir + "undistort.yaml");
}

void APDatasetReader::readCamera(const std::string& cameraFileName) {
    assert(std::filesystem::exists(cameraFileName));
    cv::FileStorage fs(cameraFileName, cv::FileStorage::READ);
    cv::Mat K, dist;

    fs["camera_matrix"] >> K;
    fs["dist_coeffs"] >> dist;

    std::array<double, 4> distVec;
    for (int i = 0; i < dist.cols; ++i) {
        distVec[i] = dist.at<double>(i);
    }

    camera = std::make_shared<GIFT::EquidistantCamera>(GIFT::EquidistantCamera(cv::Size(0, 0), K, distVec));
}

std::unique_ptr<IMUVelocity> APDatasetReader::nextIMU() {
    if (!IMUCSVFile) {
        return nullptr;
    }

    CSVLine imuLine = IMUCSVFile.nextLine();
    IMUVelocity temp;
    imuLine >> temp;
    temp.gyrBiasVel.setZero();
    temp.accBiasVel.setZero();
    return std::make_unique<IMUVelocity>(temp);
}

std::unique_ptr<StampedImage> APDatasetReader::nextImage() {
    if (!ImageCSVFile) {
        return nullptr;
    }

    CSVLine imageLine = ImageCSVFile.nextLine();
    std::string nextImageFname;
    double rawStamp;
    imageLine >> rawStamp >> nextImageFname;
    if (*nextImageFname.rbegin() == '\r') {
        nextImageFname.erase(nextImageFname.end() - 1);
    }

    nextImageFname = cam_dir + "frame_" + nextImageFname + ".jpg";

    StampedImage temp;
    temp.stamp = rawStamp - cameraLag;
    temp.image = cv::imread(nextImageFname);
    return std::make_unique<StampedImage>(temp);
}

std::vector<StampedPose> APDatasetReader::groundtruth() {
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
        if (pose.t > prevPoseTime + 1e-8) {
            // Avoid poses with the same timestamp.
            poses.emplace_back(pose);
            prevPoseTime = pose.t;
        }
    }

    return poses;
}
