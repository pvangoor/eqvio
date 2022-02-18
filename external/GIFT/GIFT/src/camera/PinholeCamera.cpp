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

#include "GIFT/camera/PinholeCamera.h"

using namespace GIFT;

PinholeCamera::PinholeCamera(cv::Size sze, cv::Mat K) {
    this->imageSize = sze;
    assert(K.rows == 3 && K.cols == 3);
    this->fx = K.at<double>(0, 0);
    this->fy = K.at<double>(1, 1);
    this->cx = K.at<double>(0, 2);
    this->cy = K.at<double>(1, 2);
}

PinholeCamera::PinholeCamera(const cv::String& cameraConfigFile) {

    cv::FileStorage fs(cameraConfigFile, cv::FileStorage::READ);

    std::vector<int> tempSize = {0, 0};
    if (!fs["image_size"].empty()) {
        fs["image_size"] >> tempSize;
    } else if (!fs["size"].empty()) {
        fs["size"] >> tempSize;
    }
    this->imageSize = cv::Size(tempSize[1], tempSize[0]);

    cv::Mat K;
    if (!fs["camera_matrix"].empty()) {
        fs["camera_matrix"] >> K;
    } else if (!fs["camera"].empty()) {
        fs["camera"] >> K;
    } else if (!fs["K"].empty()) {
        fs["K"] >> K;
    }
    this->fx = K.at<double>(0, 0);
    this->fy = K.at<double>(1, 1);
    this->cx = K.at<double>(0, 2);
    this->cy = K.at<double>(1, 2);
}

Eigen::Vector3T PinholeCamera::undistortPointEigen(const Eigen::Vector2T& point) const {
    Eigen::Vector3T result;
    result << (point.x() - cx) / fx, (point.y() - cy) / fy, 1.0;
    return result.normalized();
}

Eigen::Matrix<double, 2, 3> PinholeCamera::projectionJacobian(const Eigen::Vector3T& point) const {
    Eigen::Matrix<double, 2, 3> J;
    J << fx / point.z(), 0, -fx * point.x() / (point.z() * point.z()), 0, fy / point.z(),
        -fy * point.y() / (point.z() * point.z());
    return J;
}

Eigen::Vector2T PinholeCamera::projectPointEigen(const Eigen::Vector3T& point) const {
    Eigen::Vector2T result;
    result << fx * point.x() / point.z() + cx, fy * point.y() / point.z() + cy;
    return result;
}

cv::Mat PinholeCamera::K() const {
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = fx;
    K.at<double>(1, 1) = fy;
    K.at<double>(0, 2) = cx;
    K.at<double>(1, 2) = cy;
    return K;
}
