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

#include "gtest/gtest.h"

#include "GIFT/RANSAC.h"
#include "GIFT/camera/camera.h"

using namespace cv;
using namespace std;
using namespace GIFT;

TEST(RANSACTest, NoOutliers) {
    srand(0);
    std::mt19937 generator(0);

    const Size imageSize = Size(752, 480);
    const double fx = 458.654;
    const double fy = 457.296;
    const double cx = 367.215;
    const double cy = 248.375;
    const Mat K = (Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

    GICameraPtr cam = make_shared<PinholeCamera>(imageSize, K);

    Eigen::Quaternion<ftype> q;
    q.vec() = 0.1 * Eigen::Vector3T::Random();
    q.w() = 1.0;
    q.normalize();

    Eigen::Matrix3T R = q.toRotationMatrix();
    Eigen::Vector3T x = 0.1 * Eigen::Vector3T::Random();

    const int n_points = 50;
    vector<Eigen::Vector3T> points1(n_points);
    vector<Eigen::Vector3T> points2(n_points);
    vector<GIFT::Feature> features(n_points);

    for (int i = 0; i < n_points; ++i) {
        points1[i] = Eigen::Vector3d::Random() + Eigen::Vector3d(0, 0, 3);
        points2[i] = R * points1[i] + x;

        const cv::Point2f cc1i = cam->projectPointCV(points1[i]);
        features[i] = GIFT::Feature(cc1i, cam, i);
        const cv::Point2f cc2i = cam->projectPointCV(points2[i]);
        features[i].update(cc2i);
    }

    vector<GIFT::Feature> inlierFeatures = determineStaticWorldInliers(features, RansacParameters(), generator);
    EXPECT_EQ(inlierFeatures.size(), features.size());
}