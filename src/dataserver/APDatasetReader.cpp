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

APDatasetReader::APDatasetReader(const std::string& datasetFileName, const YAML::Node& simSettings) {

    const std::string datasetDir = datasetFileName.substr(0, datasetFileName.rfind("/")) + "/";

    // Set up required structures
    // --------------------------

    // Set up the data iterator for IMU
    IMUCSVFile = CSVFile(datasetDir + "mav_imu.csv");
    IMUCSVFile.skipLine(); // skip the header

    // Read the image data
    cam_dir = datasetDir + "frames/";
    ImageCSVFile = CSVFile(datasetDir + "cam.csv");
    ImageCSVFile.skipLine(); // skip the header

    // Set up simulator
    const std::string groundtruthFileName = datasetDir + "ground_truth.csv";
    simulator = std::make_unique<VIOSimulator>(groundtruthFileName, simSettings);

    // Read the camera file
    readCamera(datasetDir + "undistort.yaml");
}

void APDatasetReader::readCamera(const std::string& cameraFileName) {
    if (std::filesystem::exists(cameraFileName)) {
        cv::FileStorage fs(cameraFileName, cv::FileStorage::READ);
        cv::Mat K, dist;

        fs["camera_matrix"] >> K;
        fs["dist_coeffs"] >> dist;

        std::array<double, 4> distVec;
        for (int i = 0; i < dist.cols; ++i) {
            distVec[i] = dist.at<double>(i);
        }

        camera = std::make_shared<GIFT::EquidistantCamera>(GIFT::EquidistantCamera(cv::Size(0, 0), K, distVec));
    } else {
        std::cout << "No camera file found at: " << cameraFileName << std::endl;
    }
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
