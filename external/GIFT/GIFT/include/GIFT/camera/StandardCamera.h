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

#include "GIFT/camera/PinholeCamera.h"

namespace GIFT {

class StandardCamera : public PinholeCamera {
  protected:
    std::vector<ftype> dist;
    std::vector<ftype> invDist;
    std::vector<ftype> computeInverseDistortion() const;

    virtual Eigen::Vector3T undistortPointEigen(const Eigen::Vector2T& point) const override;
    virtual Eigen::Vector2T projectPointEigen(const Eigen::Vector3T& point) const override;

  public:
    StandardCamera(
        cv::Size sze = cv::Size(0, 0), cv::Mat K = cv::Mat::eye(3, 3, CV_64F), std::vector<ftype> dist = {0, 0, 0, 0});
    StandardCamera(const cv::String& cameraConfigFile);

    // Geometry functions
    const std::vector<ftype>& distortion() const;

    static Eigen::Vector2T distortHomogeneousPoint(const Eigen::Vector2T& point, const std::vector<ftype>& dist);
    static Eigen::Vector2T distortPoint(const Eigen::Vector3T& point, const std::vector<ftype>& dist) {
        return distortHomogeneousPoint(
            (Eigen::Vector2T() << point.x() / point.z(), point.y() / point.z()).finished(), dist);
    }

    virtual Eigen::Matrix<ftype, 2, 3> projectionJacobian(const Eigen::Vector3T& point) const override;
};

} // namespace GIFT
