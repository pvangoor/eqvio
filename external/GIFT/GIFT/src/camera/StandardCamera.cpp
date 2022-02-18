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

#include "GIFT/camera/StandardCamera.h"
using namespace GIFT;

StandardCamera::StandardCamera(cv::Size sze, cv::Mat K, std::vector<ftype> dist) : PinholeCamera(sze, K) {
    this->dist = dist;
    this->invDist = computeInverseDistortion();
}

StandardCamera::StandardCamera(const cv::String& cameraConfigFile) : PinholeCamera(cameraConfigFile) {

    cv::FileStorage fs(cameraConfigFile, cv::FileStorage::READ);

    if (!fs["distortion_coefficients"].empty()) {
        fs["distortion_coefficients"] >> this->dist;
    } else if (!fs["distortion"].empty()) {
        fs["distortion"] >> this->dist;
    } else if (!fs["dist"].empty()) {
        fs["dist"] >> this->dist;
    }

    this->invDist = computeInverseDistortion();
}

Eigen::Vector2T StandardCamera::projectPointEigen(const Eigen::Vector3T& point) const {
    const Eigen::Vector2T homogeneousPoint =
        (Eigen::Vector2T() << point.x() / point.z(), point.y() / point.z()).finished();
    Eigen::Vector3T distortedPoint;
    distortedPoint << distortHomogeneousPoint(homogeneousPoint, this->dist), 1.0;
    const Eigen::Vector2T projectedPoint = PinholeCamera::projectPointEigen(distortedPoint);
    return projectedPoint;
}

Eigen::Vector3T StandardCamera::undistortPointEigen(const Eigen::Vector2T& point) const {
    Eigen::Vector3T unprojectedPoint = PinholeCamera::undistortPointEigen(point);
    Eigen::Vector2T undistortedPoint = distortPoint(unprojectedPoint, invDist);
    Eigen::Vector3T result;
    result << undistortedPoint.x(), undistortedPoint.y(), 1.0;
    return result.normalized();
}

Eigen::Vector2T StandardCamera::distortHomogeneousPoint(const Eigen::Vector2T& point, const std::vector<ftype>& dist) {
    Eigen::Vector2T distortedPoint = point;
    const ftype r2 = point.x() * point.x() + point.y() * point.y();
    if (dist.size() >= 2) {
        distortedPoint.x() += point.x() * (dist[0] * r2 + dist[1] * r2 * r2);
        distortedPoint.y() += point.y() * (dist[0] * r2 + dist[1] * r2 * r2);
    }
    if (dist.size() >= 4) {
        distortedPoint.x() += 2 * dist[2] * point.x() * point.y() + dist[3] * (r2 + 2 * point.x() * point.x());
        distortedPoint.y() += 2 * dist[3] * point.x() * point.y() + dist[2] * (r2 + 2 * point.y() * point.y());
    }
    if (dist.size() >= 5) {
        distortedPoint.x() += point.x() * dist[4] * r2 * r2 * r2;
        distortedPoint.y() += point.y() * dist[4] * r2 * r2 * r2;
    }
    return distortedPoint;
}

Eigen::Matrix<ftype, 2, 3> StandardCamera::projectionJacobian(const Eigen::Vector3T& point) const {
    // Projection from sphere to homogeneous R2 point
    Eigen::Matrix<ftype, 2, 3> homogeneousProjJac;
    homogeneousProjJac << 1.0 / point.z(), 0, -1.0 * point.x() / (point.z() * point.z()), 0, 1.0 / point.z(),
        -1.0 * point.y() / (point.z() * point.z());

    // Distortion of homogeneous R2 point
    Eigen::Vector2T hPoint = point.segment<2>(0) / point.z();
    Eigen::Matrix2T distortionJac = Eigen::Matrix2d::Identity();

    const ftype r2 = hPoint.x() * hPoint.x() + hPoint.y() * hPoint.y();
    const auto Dr2 = 2 * hPoint.transpose();
    if (dist.size() >= 2) {
        distortionJac += Eigen::Matrix2T::Identity() * (dist[0] * r2 + dist[1] * r2 * r2);
        distortionJac += hPoint * (dist[0] + 2 * r2 * dist[1]) * Dr2;
    }
    if (dist.size() >= 4) {
        const ftype& px = hPoint.x();
        const ftype& py = hPoint.y();
        distortionJac(0, 0) += 2.0 * dist[2] * py + 6.0 * dist[3] * px;
        distortionJac(0, 1) += 2.0 * dist[2] * px + 2.0 * dist[3] * py;

        distortionJac(1, 0) += 2.0 * dist[2] * px + 2.0 * dist[3] * py;
        distortionJac(1, 1) += 6.0 * dist[2] * py + 2.0 * dist[3] * px;
    }
    if (dist.size() >= 5) {
        distortionJac += Eigen::Matrix2T::Identity() * dist[4] * r2 * r2 * r2;
        distortionJac += hPoint * (dist[4] * 3 * r2 * r2) * Dr2;
    }

    Eigen::Matrix2T pixelProjJac;
    pixelProjJac << fx, 0.0, 0.0, fy;

    const Eigen::Matrix<ftype, 2, 3> fullProjJac = pixelProjJac * distortionJac * homogeneousProjJac;
    return fullProjJac;
}

std::vector<ftype> StandardCamera::computeInverseDistortion() const {
    const cv::Size compSize = imageSize.area() == 0 ? cv::Size(int(round(cx * 2)), int(round(cy * 2))) : imageSize;

    // Construct a vector of normalised points
    constexpr int maxPoints = 30;
    std::vector<Eigen::Vector2T> distortedPoints;
    std::vector<Eigen::Vector2T> normalPoints;
    for (int x = 0; x < compSize.width; x += compSize.width / maxPoints) {
        for (int y = 0; y < compSize.height; y += compSize.height / maxPoints) {
            // Normalise and distort the point
            const Eigen::Vector2T normalPoint((x - cx) / fx, (y - cy) / fy);
            const Eigen::Vector2T distPoint = distortHomogeneousPoint(normalPoint, dist);

            normalPoints.emplace_back(normalPoint);
            distortedPoints.emplace_back(distPoint);
        }
    }

    // Set up the least squares problem
    Eigen::Matrix<ftype, Eigen::Dynamic, 5> lmat(distortedPoints.size() * 2, 5);
    Eigen::Matrix<ftype, Eigen::Dynamic, 1> rvec(distortedPoints.size() * 2, 1);
    for (int i = 0; i < distortedPoints.size(); ++i) {
        const Eigen::Vector2T& p = distortedPoints[i];
        const ftype r2 = p.x() * p.x() + p.y() * p.y();
        lmat.block<2, 5>(2 * i, 0) << p.x() * r2, p.x() * r2 * r2, 2 * p.x() * p.y(), r2 + 2 * p.x() * p.x(),
            p.x() * r2 * r2 * r2, p.y() * r2, p.y() * r2 * r2, r2 + 2 * p.y() * p.y(), 2 * p.x() * p.y(),
            p.y() * r2 * r2 * r2;
        rvec.block<2, 1>(2 * i, 0) << normalPoints[i].x() - p.x(), normalPoints[i].y() - p.y();
    }
    const Eigen::Matrix<ftype, 5, 1> invDist = lmat.colPivHouseholderQr().solve(rvec);
    std::vector<ftype> invDistVec(invDist.data(), invDist.data() + invDist.rows());
    return invDistVec;
}

const std::vector<ftype>& StandardCamera::distortion() const { return dist; }
