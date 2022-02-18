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

#include "GIFT/camera/DoubleSphereCamera.h"
using namespace GIFT;

DoubleSphereCamera::DoubleSphereCamera(const std::array<ftype, 6>& doubleSphereParameters, cv::Size sze) {
    imageSize = sze;
    fx = doubleSphereParameters[0];
    fy = doubleSphereParameters[1];
    cx = doubleSphereParameters[2];
    cy = doubleSphereParameters[3];
    xi = doubleSphereParameters[4];
    alpha = doubleSphereParameters[5];
}

DoubleSphereCamera::DoubleSphereCamera(const cv::String& cameraConfigFile) {

    cv::FileStorage fs(cameraConfigFile, cv::FileStorage::READ);

    std::vector<int> tempSize = {0, 0};
    if (!fs["image_size"].empty()) {
        fs["image_size"] >> tempSize;
    } else if (!fs["size"].empty()) {
        fs["size"] >> tempSize;
    }
    this->imageSize = cv::Size(tempSize[1], tempSize[0]);

    cv::Mat K;
    bool KFlag = true;
    if (!fs["camera_matrix"].empty()) {
        fs["camera_matrix"] >> K;
    } else if (!fs["camera"].empty()) {
        fs["camera"] >> K;
    } else if (!fs["K"].empty()) {
        fs["K"] >> K;
    } else {
        KFlag = false;
    }
    if (KFlag) {
        this->fx = K.at<double>(0, 0);
        this->fy = K.at<double>(1, 1);
        this->cx = K.at<double>(0, 2);
        this->cy = K.at<double>(1, 2);
    } else {
        if (!fs["fx"].empty())
            fs["fx"] >> this->fx;
        if (!fs["fy"].empty())
            fs["fy"] >> this->fy;
        if (!fs["cx"].empty())
            fs["cx"] >> this->cx;
        if (!fs["cy"].empty())
            fs["cy"] >> this->cy;
    }

    if (!fs["xi"].empty()) {
        fs["xi"] >> this->xi;
    }
    if (!fs["alpha"].empty()) {
        fs["alpha"] >> this->alpha;
    }
}

std::array<ftype, 6> DoubleSphereCamera::parameters() const { return std::array<ftype, 6>{fx, fy, cx, cy, xi, alpha}; }

Eigen::Vector3T DoubleSphereCamera::undistortPointEigen(const Eigen::Vector2T& point) const {
    Eigen::Vector3T mVec;

    mVec.x() = (point.x() - cx) / fx;
    mVec.y() = (point.y() - cy) / fy;
    const ftype r2 = mVec.x() * mVec.x() + mVec.y() * mVec.y();
    mVec.z() = (1 - alpha * alpha * r2) / (alpha * sqrt(1 - (2 * alpha - 1) * r2) + 1. - alpha);

    const ftype factor = (mVec.z() * xi + sqrt(mVec.z() * mVec.z() + (1 - xi * xi) * r2)) / (mVec.z() * mVec.z() + r2);
    mVec = factor * mVec - Eigen::Vector3T(0, 0, xi);

    return mVec;
}
Eigen::Vector2T DoubleSphereCamera::projectPointEigen(const Eigen::Vector3T& point) const {
    Eigen::Vector2T projPoint;
    const ftype d1 = point.norm();
    const ftype d2 = (point + Eigen::Vector3T(0, 0, xi * d1)).norm();
    const ftype denom = 1.0 / (alpha * d2 + (1 - alpha) * (xi * d1 + point.z()));
    projPoint.x() = fx * denom * point.x() + cx;
    projPoint.y() = fy * denom * point.y() + cy;

    return projPoint;
}
