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

#include "GIFT/ftype.h"
#include "eigen3/Eigen/Dense"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <array>
#include <memory>
#include <vector>

namespace GIFT {

class GICamera;
struct Feature {
    cv::Point2f camCoordinates;

    std::shared_ptr<const GICamera> cameraPtr;

    Eigen::Vector2T opticalFlowNorm;

    int idNumber;
    int lifetime = 0;

    Feature(){};
    Feature(const cv::Point2f& newCamCoords, const std::shared_ptr<const GICamera>& cameraPtr, int idNumber);
    void update(const cv::Point2f& newCamCoords);

    Eigen::Vector2T camCoordinatesEigen() const;
    cv::Point2f camCoordinatesNorm() const;
    Eigen::Vector3T sphereCoordinates() const;
    Eigen::Vector3T opticalFlowSphere() const;
};

} // namespace GIFT
