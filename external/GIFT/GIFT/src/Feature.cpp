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

#include "GIFT/Feature.h"
#include "GIFT/camera/camera.h"

using namespace GIFT;
using namespace cv;
using namespace Eigen;

Feature::Feature(const Point2f& newCamCoords, const std::shared_ptr<const GICamera>& cameraPtr, int idNumber) {
    this->camCoordinates = newCamCoords;
    this->cameraPtr = cameraPtr;

    this->opticalFlowNorm.setZero();

    this->idNumber = idNumber;

    lifetime = 1;
}

void Feature::update(const cv::Point2f& newCamCoords) {
    cv::Point2f newCamCoordsNorm = cameraPtr->undistortPoint(newCamCoords);
    this->opticalFlowNorm << newCamCoordsNorm.x - this->camCoordinatesNorm().x,
        newCamCoordsNorm.y - this->camCoordinatesNorm().y;

    this->camCoordinates = newCamCoords;

    Vector3T bearing = Vector3T(newCamCoordsNorm.x, newCamCoordsNorm.y, 1).normalized();

    ++lifetime;
}

Eigen::Vector3T Feature::sphereCoordinates() const {
    Eigen::Vector3T result;
    result << camCoordinatesNorm().x, camCoordinatesNorm().y, 1.0;
    return result.normalized();
}

cv::Point2f Feature::camCoordinatesNorm() const { return cameraPtr->undistortPoint(camCoordinates); }

Eigen::Vector2T Feature::camCoordinatesEigen() const {
    Eigen::Vector2T ptEigen;
    ptEigen << camCoordinates.x, camCoordinates.y;
    return ptEigen;
}

Eigen::Vector3T Feature::opticalFlowSphere() const {
    const Vector3T bearing = sphereCoordinates();
    const Vector3T sphereFlow = bearing.z() * (Matrix3T::Identity() - bearing * bearing.transpose()) *
                                Vector3T(opticalFlowNorm.x(), opticalFlowNorm.y(), 0);
    return sphereFlow;
}