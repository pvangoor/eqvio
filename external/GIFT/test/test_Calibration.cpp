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

#include "GIFT/Calibration.h"
#include "gtest/gtest.h"

#include "eigen3/unsupported/Eigen/MatrixFunctions"

#include "opencv2/core/eigen.hpp"

using namespace Eigen;
using namespace std;
using namespace cv;

static double norm(const Point2f& p) { return pow(p.x * p.x + p.y * p.y, 0.5); }
static Matrix3T skew(const Vector3T& v) {
    Matrix3T m;
    m << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
    return m;
};

TEST(CalibrationTest, IntrinsicsRecovery) {
    const Size imageSize = Size(752, 480);
    const double fx = 458.654;
    const double fy = 457.296;
    const double cx = 367.215;
    const double cy = 248.375;

    Eigen::Matrix3T trueK;
    trueK << fx, 0, cx, 0, fy, cy, 0, 0, 1;

    // Create a target
    std::vector<Eigen::Vector4T> targetPoints(64);
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            targetPoints[8 * i + j] << 0.05 * i, 0.05 * j, 0.0, 1.0;
        }
    }

    // Generate poses and homographies
    std::vector<Eigen::Matrix4T> truePoses(20);
    std::vector<Eigen::Matrix3T> homographies(20);
    for (int i = 0; i < 20; ++i) {
        Eigen::Matrix4T& P = truePoses[i];
        P.setZero();
        P.block<3, 1>(0, 3) = Vector3T::Random();
        P.block<3, 3>(0, 0) = skew(Vector3T::Random() * 0.2);
        P = P.exp();

        Eigen::Matrix3T& H = homographies[i];
        H = truePoses[i].block<3, 3>(0, 0);
        H.block<3, 1>(0, 2) = truePoses[i].block<3, 1>(0, 3);
        H = trueK * H;
    }

    Eigen::Matrix3T estK = GIFT::initialisePinholeIntrinsics(homographies);

    EXPECT_LE((estK - trueK).norm(), 1e-6);

    // Estimate the poses again
    std::vector<Eigen::Matrix4T> estPoses = GIFT::initialisePoses(homographies, estK);
    ASSERT_EQ(estPoses.size(), truePoses.size());
    for (int i = 0; i < truePoses.size(); ++i) {
        EXPECT_LE((estPoses[i] - truePoses[i]).norm(), 1e-6);
    }
}
