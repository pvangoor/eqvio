/*
    This file is part of EqVIO.

    EqVIO is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    EqVIO is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with EqVIO.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "eigen3/Eigen/Dense"
#include "eqvio/VIOFilter.h"
#include "testing_utilities.h"
#include "gtest/gtest.h"

using namespace Eigen;
using namespace std;

TEST(CoordinateChartTest, SphereChartE3) {
    // Test the sphere coordinate charts
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        const Vector3d eta = Vector3d::Random().normalized();
        const Vector2d y1 = e3ProjectSphere(eta);
        const Vector3d eta1 = e3ProjectSphereInv(y1);
        const double dist_eta1 = (eta - eta1).norm();
        EXPECT_LE(dist_eta1, NEAR_ZERO);

        const Vector2d y = Vector2d::Random();
        const Vector3d eta2 = e3ProjectSphereInv(y);
        const Vector2d y2 = e3ProjectSphere(eta2);
        const double dist_y2 = (y2 - y).norm();
        EXPECT_LE(dist_y2, NEAR_ZERO);
    }
}

TEST(CoordinateChartTest, SphereChartPole) {
    // Test the sphere coordinate charts with poles
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        const Vector3d pole = Vector3d::Random().normalized();

        const Vector2d poleCoords = sphereChart_stereo(pole, pole);
        EXPECT_LE(poleCoords.norm(), NEAR_ZERO);

        const Vector3d eta = Vector3d::Random().normalized();
        const Vector2d y1 = sphereChart_stereo(eta, pole);
        const Vector3d eta1 = sphereChart_stereo.inv(y1, pole);
        const double dist_eta1 = (eta - eta1).norm();
        EXPECT_LE(dist_eta1, NEAR_ZERO);

        const Vector2d y = Vector2d::Random();
        const Vector3d eta2 = sphereChart_stereo.inv(y, pole);
        const Vector2d y2 = sphereChart_stereo(eta2, pole);
        const double dist_y2 = (y2 - y).norm();
        EXPECT_LE(dist_y2, NEAR_ZERO);
    }
}

TEST(CoordinateChartTest, SphereChartPoleNormal) {
    // Test the sphere coordinate charts with poles
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        const Vector3d pole = Vector3d::Random().normalized();

        const Vector2d poleCoords = sphereChart_normal(pole, pole);
        EXPECT_LE(poleCoords.norm(), NEAR_ZERO);

        const Vector3d eta = Vector3d::Random().normalized();
        const Vector2d y1 = sphereChart_normal(eta, pole);
        const Vector3d eta1 = sphereChart_normal.inv(y1, pole);
        const double dist_eta1 = (eta - eta1).norm();
        EXPECT_LE(dist_eta1, NEAR_ZERO);

        const Vector2d y = Vector2d::Random();
        const Vector3d eta2 = sphereChart_normal.inv(y, pole);
        const Vector2d y2 = sphereChart_normal(eta2, pole);
        const double dist_y2 = (y2 - y).norm();
        EXPECT_LE(dist_y2, NEAR_ZERO);
    }
}

TEST(CoordinateChartTest, SphereChartE3Differential) {
    // Test the sphere chart differential
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        const Vector3d eta = Vector3d::Random().normalized();
        testDifferential(e3ProjectSphere, eta, e3ProjectSphereDiff(eta));

        // Test the sphere chart inverse differential
        const Vector2d y = Vector2d::Random();
        testDifferential(e3ProjectSphereInv, y, e3ProjectSphereInvDiff(y));
    }
}

TEST(CoordinateChartTest, SphereChartPoleDifferential) {
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        // Test the sphere chart differential with poles
        const Vector3d pole = Vector3d::Random().normalized();
        testDifferential(
            [&](const Vector3d& eta) { return sphereChart_stereo.chart(eta, pole); }, pole,
            sphereChart_stereo.chartDiff0(pole));

        // Test the sphere chart inverse differential
        testDifferential(
            [&](const Vector2d& y) { return sphereChart_stereo.chartInv(y, pole); }, Vector2d::Zero(),
            sphereChart_stereo.chartInvDiff0(pole));
    }
}

TEST(CoordinateChartTest, SphereChartPoleDifferentialNormal) {
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        // Test the sphere chart differential with poles
        const Vector3d pole = Vector3d::Random().normalized();
        testDifferential(
            [&](const Vector3d& eta) { return sphereChart_normal.chart(eta, pole); }, pole,
            sphereChart_normal.chartDiff0(pole));

        // Test the sphere chart inverse differential
        testDifferential(
            [&](const Vector2d& y) { return sphereChart_normal.chartInv(y, pole); }, Vector2d::Zero(),
            sphereChart_normal.chartInvDiff0(pole));
    }
}

void VIOChart_test(const CoordinateChart<VIOState>& chart) {
    vector<int> ids = {0, 1, 2, 3, 4};
    // Test the VIO manifold coordinate chart
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        const VIOState xi0 = randomStateElement(ids);
        const VIOState xi1 = randomStateElement(ids);

        const VectorXd eps = chart(xi1, xi0);
        const VIOState xi2 = chart.inv(eps, xi0);

        double dist1Total = stateDistance(xi1, xi2);
        EXPECT_LE(dist1Total, 1e-8);
    }
}

TEST(CoordinateChartTest, VIOChart_euclid) { VIOChart_test(VIOChart_euclid); }
TEST(CoordinateChartTest, VIOChart_invdepth) { VIOChart_test(VIOChart_invdepth); }
TEST(CoordinateChartTest, VIOChart_normal) { VIOChart_test(VIOChart_normal); }

TEST(CoordinateChartTest, VIOChart_euclid_invdepth_diff) {
    const vector<int> ids = {0, 1, 2, 3, 4};
    // Test the VIO manifold coordinate chart
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        const VIOState xi0 = randomStateElement(ids);

        auto coordChange = [&](const VectorXd& eps) { return VIOChart_invdepth(VIOChart_euclid.inv(eps, xi0), xi0); };
        const MatrixXd changeMatrix = coordinateDifferential_invdepth_euclid(xi0);
        testDifferential(coordChange, VIOChart_euclid(xi0, xi0), changeMatrix);
    }
}

TEST(CoordinateChartTest, VIOChart_euclid_normal_diff) {
    const vector<int> ids = {0, 1, 2, 3, 4};
    // Test the VIO manifold coordinate chart
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        const VIOState xi0 = randomStateElement(ids);

        auto coordChange = [&](const VectorXd& eps) { return VIOChart_normal(VIOChart_euclid.inv(eps, xi0), xi0); };
        const MatrixXd changeMatrix = coordinateDifferential_normal_euclid(xi0);
        testDifferential(coordChange, VIOChart_euclid(xi0, xi0), changeMatrix);
    }
}
