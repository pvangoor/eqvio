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

#include "eqvio/dataserver/HiltiDatasetReader.h"
#include "yaml-cpp/yaml.h"

void HiltiDatasetReader::readCamera(const std::string& cameraFileName) {
    YAML::Node calibrationNode = YAML::LoadFile(cameraFileName);
    const auto& cameraNode = calibrationNode["sensors"]["cam0"];

    // Get the intrinsics
    const auto& intrinsicParams = cameraNode["intrinsics"]["parameters"];

    cv::Size imageSize(intrinsicParams["image_size"][0].as<int>(), intrinsicParams["image_size"][1].as<int>());

    const cv::Mat K =
        (cv::Mat_<double>(3, 3) << intrinsicParams["fx"].as<double>(), 0, intrinsicParams["cx"].as<double>(), 0,
         intrinsicParams["fy"].as<double>(), intrinsicParams["cy"].as<double>(), 0, 0, 1);

    std::array<double, 4> distortion = {
        intrinsicParams["k1"].as<double>(), intrinsicParams["k2"].as<double>(), intrinsicParams["k3"].as<double>(),
        intrinsicParams["k4"].as<double>()};

    camera = std::make_shared<GIFT::EquidistantCamera>(imageSize, K, distortion);

    // Get the extrinsics
    const auto& extrinsicParams = cameraNode["extrinsics"];
    // Note the quaternion is in Hamilton form (w is last instead of first) but we use standard form.
    Eigen::Quaterniond attitude = Eigen::Quaterniond(
        extrinsicParams["quaternion"][3].as<double>(), extrinsicParams["quaternion"][0].as<double>(),
        extrinsicParams["quaternion"][1].as<double>(), extrinsicParams["quaternion"][2].as<double>());
    Eigen::Vector3d position = Eigen::Vector3d(
        extrinsicParams["translation"][0].as<double>(), extrinsicParams["translation"][1].as<double>(),
        extrinsicParams["translation"][2].as<double>());
    liepp::SE3d extrinsics = liepp::SE3d(liepp::SO3d(attitude), position);
    cameraExtrinsics = std::make_unique<liepp::SE3d>(extrinsics);
}
