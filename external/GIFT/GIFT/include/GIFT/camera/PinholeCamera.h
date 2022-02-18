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

#include "GIFT/camera/GICamera.h"

namespace GIFT {

class PinholeCamera : public GICamera {
  protected:
    ftype fx, fy, cx, cy; // intrinsic parameters

    virtual Eigen::Vector3T undistortPointEigen(const Eigen::Vector2T& point) const override;
    virtual Eigen::Vector2T projectPointEigen(const Eigen::Vector3T& point) const override;

  public:
    cv::Mat K() const; // intrinsic matrix (3x3)
    PinholeCamera(cv::Size sze = cv::Size(0, 0), cv::Mat K = cv::Mat::eye(3, 3, CV_64F));
    PinholeCamera(const cv::String& cameraConfigFile);
    virtual Eigen::Matrix<ftype, 2, 3> projectionJacobian(const Eigen::Vector3T& point) const override;
};

} // namespace GIFT
