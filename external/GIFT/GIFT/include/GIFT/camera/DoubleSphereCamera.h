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

class DoubleSphereCamera : public PinholeCamera {
    // Implements the double sphere camera model found here:
    // https://arxiv.org/pdf/1807.08957.pdf
  protected:
    ftype xi, alpha;

    virtual Eigen::Vector3T undistortPointEigen(const Eigen::Vector2T& point) const override;
    virtual Eigen::Vector2T projectPointEigen(const Eigen::Vector3T& point) const override;

  public:
    DoubleSphereCamera() = default;
    DoubleSphereCamera(const std::array<ftype, 6>& doubleSphereParameters, cv::Size sze = cv::Size(0, 0));
    DoubleSphereCamera(const cv::String& cameraConfigFile);

    // Geometry functions
    std::array<ftype, 6> parameters() const;
};

} // namespace GIFT
