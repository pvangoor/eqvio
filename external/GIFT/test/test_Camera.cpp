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

#include "GIFT/camera/camera.h"
#include "gtest/gtest.h"

using namespace Eigen;
using namespace std;
using namespace cv;
using namespace GIFT;

static double norm(const Point2f& p) { return pow(p.x * p.x + p.y * p.y, 0.5); }

Eigen::MatrixXd numericalDifferential(
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> f, const Eigen::VectorXd& x, ftype h = -1.0) {
    if (h < 0) {
        h = std::cbrt(std::numeric_limits<ftype>::epsilon());
    }
    Eigen::MatrixXd Df(f(x).rows(), x.rows());
    for (int j = 0; j < Df.cols(); ++j) {
        const Eigen::VectorXd ej = Eigen::VectorXd::Unit(Df.cols(), j);
        Df.col(j) = (f(x + h * ej) - f(x - h * ej)) / (2 * h);
    }
    return Df;
}

template <typename F>
void testDifferential(const F& f, const Eigen::VectorXd& x, const Eigen::MatrixXd& Df, double h = -1.0) {
    // Check that each partial derivative is correct
    if (h < 0) {
        h = std::cbrt(std::numeric_limits<double>::epsilon());
    }
    const int& n = Df.rows();
    const int& m = Df.cols();
    Eigen::MatrixXd numericalDf = numericalDifferential(f, x, h);

    EXPECT_FALSE(Df.hasNaN());
    EXPECT_FALSE(numericalDf.hasNaN());
    // Check each entry
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            EXPECT_NEAR(Df(i, j), numericalDf(i, j), std::max(h, 5e3 * h * abs(Df(i, j))))
                << "Entry (" << i << ", " << j << ")";
        }
    }
}

TEST(CameraTest, PinholeProject) {
    const Size imageSize = Size(752, 480);
    const double fx = 458.654;
    const double fy = 457.296;
    const double cx = 367.215;
    const double cy = 248.375;
    const Mat K = (Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

    PinholeCamera cam = PinholeCamera(imageSize, K);

    // Test on a grid of points
    constexpr int skip = 30;
    for (int x = 0; x < imageSize.width; x += skip) {
        for (int y = 0; y < imageSize.height; y += skip) {
            const Point2f imagePoint(x, y);
            const Point2f normalPoint((x - cx) / fx, (y - cy) / fy);
            const Point2f estNormalPoint = cam.undistortPoint(imagePoint);

            const double error = norm(estNormalPoint - normalPoint);
            EXPECT_LE(error, 1e-4);
        }
    }
}

TEST(CameraTest, StandardProjectionJacobian) {
    const Size imageSize = Size(752, 480);
    const double fx = 458.654;
    const double fy = 457.296;
    const double cx = 367.215;
    const double cy = 248.375;
    const Mat K = (Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

    StandardCamera cam = StandardCamera(imageSize, K, {-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 1e-4});

    // Test on a grid of points
    constexpr int skip = 30;
    const auto projFun = [&cam](const Eigen::Vector3T& sp) { return cam.projectPoint(sp); };
    for (int x = 0; x < imageSize.width; x += skip) {
        for (int y = 0; y < imageSize.height; y += skip) {
            const Vector2T imagePoint(x, y);
            const Vector3T spherePoint = cam.undistortPoint(imagePoint);

            const Matrix<ftype, 2, 3> Jacobian = cam.projectionJacobian(spherePoint);
            testDifferential(projFun, spherePoint, Jacobian);
        }
    }
}

TEST(CameraTest, EquidistantProjectionJacobian) {
    const Size imageSize = Size(640, 480);
    const double fx = 278.66723066149086;
    const double fy = 278.48991409740296;
    const double cx = 319.75221200593535;
    const double cy = 241.96858910358173;
    const Mat K = (Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

    EquidistantCamera cam = EquidistantCamera(
        imageSize, K, {-0.013721808247486035, 0.020727425669427896, -0.012786476702685545, 0.0025242267320687625});

    // Test on a grid of points
    constexpr int skip = 30;
    const auto projFun = [&cam](const Eigen::Vector3T& sp) { return cam.projectPoint(sp); };
    for (int x = 0; x < imageSize.width; x += skip) {
        for (int y = 0; y < imageSize.height; y += skip) {
            const Vector2T imagePoint(x, y);
            const Vector3T spherePoint = cam.undistortPoint(imagePoint);

            const Matrix<ftype, 2, 3> Jacobian = cam.projectionJacobian(spherePoint);
            testDifferential(projFun, spherePoint, Jacobian);
        }
    }
}

TEST(CameraTest, EquidistantReprojection) {
    const Size imageSize = Size(640, 480);
    const double fx = 278.66723066149086;
    const double fy = 278.48991409740296;
    const double cx = 319.75221200593535;
    const double cy = 241.96858910358173;
    const Mat K = (Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

    EquidistantCamera cam = EquidistantCamera(
        imageSize, K, {-0.013721808247486035, 0.020727425669427896, -0.012786476702685545, 0.0025242267320687625});

    // Test on a grid of points
    constexpr int skip = 30;
    for (int x = 0; x < imageSize.width; x += skip) {
        for (int y = 0; y < imageSize.height; y += skip) {
            const Eigen::Vector2T imagePoint(x, y);

            const Eigen::Vector3T spherePoint = cam.undistortPoint(imagePoint);
            const Eigen::Vector2T estImagePoint = cam.projectPoint(spherePoint);

            const double error = (estImagePoint - imagePoint).norm();
            EXPECT_LE(error, 1e-1);
        }
    }
}

TEST(CameraTest, DoubleSphereReprojection) {
    const Size imageSize = Size(752, 480);
    const double fx = 458.654;
    const double fy = 457.296;
    const double cx = 367.215;
    const double cy = 248.375;
    const double xi = -0.2;
    const double alpha = 0.5;

    DoubleSphereCamera cam{std::array<ftype, 6>{fx, fy, cx, cy, xi, alpha}};

    // Test on a grid of points
    constexpr int skip = 30;
    for (int x = 0; x < imageSize.width; x += skip) {
        for (int y = 0; y < imageSize.height; y += skip) {
            const Eigen::Vector2T imagePoint(x, y);

            const Eigen::Vector3T spherePoint = cam.undistortPoint(imagePoint);
            const Eigen::Vector2T estImagePoint = cam.projectPoint(spherePoint);

            const double error = (estImagePoint - imagePoint).norm();
            EXPECT_LE(error, 1e-3);
        }
    }
}