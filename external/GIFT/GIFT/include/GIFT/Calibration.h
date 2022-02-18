/*
    This file is part of GIFT.

    GIFT is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    GIFT is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with GIFT.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include "GIFT/camera/camera.h"

namespace GIFT {

Eigen::Matrix3T initialisePinholeIntrinsics(const std::vector<cv::Mat>& homographies);
Eigen::Matrix3T initialisePinholeIntrinsics(const std::vector<Eigen::Matrix3T>& homographies);

std::vector<Eigen::Matrix4T> initialisePoses(
    const std::vector<cv::Mat>& homographies, const Eigen::Matrix3T& cameraMatrix);
Eigen::Matrix4T initialisePose(const cv::Mat& homography, const Eigen::Matrix3T& cameraMatrix);
std::vector<Eigen::Matrix4T> initialisePoses(
    const std::vector<Eigen::Matrix3T>& homographies, const Eigen::Matrix3T& cameraMatrix);
Eigen::Matrix4T initialisePose(const Eigen::Matrix3T& homography, const Eigen::Matrix3T& cameraMatrix);

}; // namespace GIFT