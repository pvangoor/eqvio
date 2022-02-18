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

#include "GIFT/EgoMotion.h"
#include "gtest/gtest.h"

using namespace Eigen;
using namespace std;

class EgoMotionTest : public ::testing::Test {
  protected:
    EgoMotionTest(int pointCount = 100) {
        this->pointCount = pointCount;
        for (int i = 0; i < pointCount; ++i) {
            Vector3T point = Vector3T::Random() * 100; // Points uniform in 100 size cube
            while (point.norm() < 1e-1)
                point = Vector3T::Random() * 100;
            ftype rho = 1 / point.norm();
            Vector3T eta = point * rho;
            this->bearingsAndInvDepths.emplace_back(make_pair(eta, rho));
        }
    }

    vector<pair<Vector3T, ftype>> bearingsAndInvDepths;
    int pointCount;
};

TEST_F(EgoMotionTest, ConvergesLinVel) {
    int testCount = 20;
    for (int i = 0; i < testCount; ++i) {
        Vector3T trueLinVel = (Vector3T::Random()).normalized();
        Vector3T trueAngVel = Vector3T::Random();

        vector<pair<Vector3T, Vector3T>> sphereFlows;
        for (const auto& etaRho : bearingsAndInvDepths) {
            Vector3T phi =
                etaRho.second * (Matrix3T::Identity() - etaRho.first * etaRho.first.transpose()) * trueLinVel -
                trueAngVel.cross(etaRho.first);
            sphereFlows.emplace_back(make_pair(etaRho.first, phi));
        }

        Vector3T initialLinVel = (trueLinVel + i * trueLinVel.norm() * Vector3T::Random() / testCount).normalized();

        GIFT::EgoMotion egoMotion(sphereFlows, initialLinVel, trueAngVel);
        const Vector3T& estLinVel = egoMotion.linearVelocity.normalized();
        const Vector3T& estAngVel = egoMotion.angularVelocity;

        EXPECT_LE((estAngVel - trueAngVel).norm(), 1e-2);
        EXPECT_LE(pow(estLinVel.dot(trueLinVel), 2) - 1, 1e-2);
    }
}

TEST_F(EgoMotionTest, ConvergesAngVel) {
    int testCount = 20;
    for (int i = 0; i < testCount; ++i) {
        Vector3T trueLinVel = (Vector3T::Random()).normalized();
        Vector3T trueAngVel = Vector3T::Random() * 4;

        vector<pair<Vector3T, Vector3T>> sphereFlows;
        for (const auto& etaRho : bearingsAndInvDepths) {
            Vector3T phi =
                etaRho.second * (Matrix3T::Identity() - etaRho.first * etaRho.first.transpose()) * trueLinVel -
                trueAngVel.cross(etaRho.first);
            sphereFlows.emplace_back(make_pair(etaRho.first, phi));
        }

        Vector3T initialAngVel = trueAngVel + trueAngVel.norm() * i * Vector3T::Random() / testCount;

        GIFT::EgoMotion egoMotion(sphereFlows, trueLinVel, initialAngVel);
        const Vector3T& estLinVel = egoMotion.linearVelocity.normalized();
        const Vector3T& estAngVel = egoMotion.angularVelocity;

        EXPECT_LE((estAngVel - trueAngVel).norm(), 1e-2);
        EXPECT_LE(pow(estLinVel.dot(trueLinVel), 2) - 1, 1e-2);
    }
}

TEST_F(EgoMotionTest, ConvergesFullVel) {
    int testCount = 20;
    for (int i = 0; i < testCount; ++i) {
        Vector3T trueLinVel = (Vector3T::Random()).normalized();
        Vector3T trueAngVel = Vector3T::Random() * 4;

        vector<pair<Vector3T, Vector3T>> sphereFlows;
        for (const auto& etaRho : bearingsAndInvDepths) {
            Vector3T phi =
                etaRho.second * (Matrix3T::Identity() - etaRho.first * etaRho.first.transpose()) * trueLinVel -
                trueAngVel.cross(etaRho.first);
            sphereFlows.emplace_back(make_pair(etaRho.first, phi));
        }

        Vector3T initialLinVel = (trueLinVel + i * trueLinVel.norm() * Vector3T::Random() / testCount).normalized();
        Vector3T initialAngVel = trueAngVel + trueAngVel.norm() * i * Vector3T::Random() / testCount;

        GIFT::EgoMotion egoMotion(sphereFlows, initialLinVel, initialAngVel);
        const Vector3T& estLinVel = egoMotion.linearVelocity.normalized();
        const Vector3T& estAngVel = egoMotion.angularVelocity;

        EXPECT_LE((estAngVel - trueAngVel).norm(), 1e-2);
        EXPECT_LE(pow(estLinVel.dot(trueLinVel), 2) - 1, 1e-2);
    }
}

TEST_F(EgoMotionTest, ConvergesAngVelFromLinVel) {
    int testCount = 20;
    for (int i = 0; i < testCount; ++i) {
        Vector3T trueLinVel = (Vector3T::Random()).normalized();
        Vector3T trueAngVel = Vector3T::Random() * 4;

        vector<pair<Vector3T, Vector3T>> sphereFlows;
        for (const auto& etaRho : bearingsAndInvDepths) {
            Vector3T phi =
                etaRho.second * (Matrix3T::Identity() - etaRho.first * etaRho.first.transpose()) * trueLinVel -
                trueAngVel.cross(etaRho.first);
            sphereFlows.emplace_back(make_pair(etaRho.first, phi));
        }

        Vector3T initialLinVel =
            (trueLinVel + i * 0.25 * trueLinVel.norm() * Vector3T::Random() / testCount).normalized();

        GIFT::EgoMotion egoMotion(sphereFlows, initialLinVel);
        const Vector3T& estLinVel = egoMotion.linearVelocity.normalized();
        const Vector3T& estAngVel = egoMotion.angularVelocity;

        EXPECT_LE((estAngVel - trueAngVel).norm(), 1e-2);
        EXPECT_LE(pow(estLinVel.dot(trueLinVel), 2) - 1, 1e-2);
    }
}