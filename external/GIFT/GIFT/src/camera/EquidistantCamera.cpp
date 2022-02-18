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

#include "GIFT/camera/EquidistantCamera.h"
using namespace GIFT;

EquidistantCamera::EquidistantCamera(cv::Size sze, cv::Mat K, std::array<ftype, 4> dist)
    : PinholeCamera(sze, K), dist(dist) {}

EquidistantCamera::EquidistantCamera(const cv::String& cameraConfigFile) : PinholeCamera(cameraConfigFile) {
    cv::FileStorage fs(cameraConfigFile, cv::FileStorage::READ);
    std::vector<ftype> temp;
    if (!fs["distortion_coefficients"].empty()) {
        fs["distortion_coefficients"] >> temp;
    } else if (!fs["distortion"].empty()) {
        fs["distortion"] >> temp;
    } else if (!fs["dist"].empty()) {
        fs["dist"] >> temp;
    }
    std::copy(temp.begin(), temp.begin() + 4, this->dist.begin());
}

Eigen::Vector2T EquidistantCamera::projectPointEigen(const Eigen::Vector3T& point) const {
    const Eigen::Vector2T homogeneousPoint =
        (Eigen::Vector2T() << point.x() / point.z(), point.y() / point.z()).finished();
    Eigen::Vector3T distortedPoint;
    distortedPoint << distortHomogeneousPoint(homogeneousPoint, this->dist), 1.0;
    const Eigen::Vector2T projectedPoint = PinholeCamera::projectPointEigen(distortedPoint);
    return projectedPoint;
}

Eigen::Vector3T EquidistantCamera::undistortPointEigen(const Eigen::Vector2T& point) const {
    constexpr ftype stepEps = 0.005;
    constexpr ftype resEps = 0.1;
    constexpr int max_iter = 30;
    constexpr ftype trust = 1000.0;

    // Use Gauss-Newton to estimate
    Eigen::Vector3T result = PinholeCamera::undistortPointEigen(point); // Initial guess
    for (int iter = 0; iter < max_iter; ++iter) {
        const Eigen::Vector2T estPoint = projectPoint(result);
        const auto res = point - estPoint;
        if (res.norm() < resEps) {
            break;
        }
        const auto Jac = projectionJacobian(result);
        const auto Hes = Jac.transpose() * Jac + Eigen::Matrix3T::Identity() * trust;
        Eigen::Vector3T step = Hes.ldlt().solve(Jac.transpose() * res);
        result = (result + step).normalized();
        if (step.norm() < stepEps) {
            break;
        }
    }
    return result;
}

Eigen::Vector2T EquidistantCamera::distortHomogeneousPoint(
    const Eigen::Vector2T& point, const std::array<ftype, 4>& dist) {

    const ftype r = sqrt(point.x() * point.x() + point.y() * point.y());
    const ftype theta = atan(r);
    const ftype temp = theta * (1.0 + dist[0] * pow(theta, 2) + dist[1] * pow(theta, 4) + dist[2] * pow(theta, 6) +
                                   dist[3] * pow(theta, 8));
    const ftype scale = (r > 1e-6) ? temp / r : 1.0;

    return scale * point;
}

Eigen::Matrix<ftype, 2, 3> EquidistantCamera::projectionJacobian(const Eigen::Vector3T& point) const {
    // Projection from sphere to homogeneous R2 point
    Eigen::Matrix<ftype, 2, 3> homogeneousProjJac;
    homogeneousProjJac << 1.0 / point.z(), 0, -1.0 * point.x() / (point.z() * point.z()), 0, 1.0 / point.z(),
        -1.0 * point.y() / (point.z() * point.z());

    // Distortion of homogeneous R2 point
    Eigen::Vector2T hPoint = point.segment<2>(0) / point.z();
    Eigen::Matrix2T distortionJac = Eigen::Matrix2T::Identity();

    const ftype r = hPoint.norm();
    if (r > 1e-6) {
        const ftype theta = atan(r);
        const ftype temp = (1.0 + dist[0] * pow(theta, 2) + dist[1] * pow(theta, 4) + dist[2] * pow(theta, 6) +
                            dist[3] * pow(theta, 8));
        distortionJac = temp * theta / r * Eigen::Matrix2T::Identity();

        // Differentiate wrt theta
        const Eigen::Matrix<ftype, 1, 2> Dr = hPoint.transpose() / r;
        const Eigen::Matrix<ftype, 1, 2> Dth = Dr / (1.0 + r * r);

        ftype DTemp = temp / r;
        for (int i = 1; i < 5; ++i) {
            DTemp += theta / r * dist[i - 1] * (2 * i) * pow(theta, 2 * i - 1);
        }
        distortionJac += hPoint * DTemp * Dth;

        // Differentiate wrt r
        distortionJac += -theta / (r * r) * temp * hPoint * Dr;
    };

    Eigen::Matrix2T pixelProjJac;
    pixelProjJac << fx, 0.0, 0.0, fy;

    const Eigen::Matrix<ftype, 2, 3> fullProjJac = pixelProjJac * distortionJac * homogeneousProjJac;
    return fullProjJac;
}

const std::array<ftype, 4>& EquidistantCamera::distortion() const { return dist; }
