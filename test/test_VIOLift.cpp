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
#include "eqvio/EqFMatrices.h"
#include "eqvio/VIOGroup.h"
#include "eqvio/VIOState.h"
#include "testing_utilities.h"
#include "gtest/gtest.h"

using namespace Eigen;
using namespace std;

TEST(VIOLiftTest, Lift) {
    vector<int> ids = {0, 1, 2, 3, 4};
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        const VIOState xi0 = randomStateElement(ids);
        const IMUVelocity velocity = randomVelocityElement();

        // Check the convergence of the derivative
        double previousDist = 1e8;
        for (int i = 0; i < 8; ++i) {
            // Integrate the system normally
            const double dt = pow(10.0, -i);
            const VIOState xi1 = integrateSystemFunction(xi0, velocity, dt);

            // Lift the velocity and apply as a group action
            const VIOAlgebra lambda = liftVelocity(xi0, velocity);
            const VIOGroup lambdaExp = VIOExp(dt * lambda);
            const VIOState xi2 = stateGroupAction(lambdaExp, xi0);

            // Check the error has decreased
            const double diffDist = stateDistance(xi1, xi2) / dt;
            EXPECT_LE(diffDist, previousDist);
            previousDist = diffDist;
        }
    }
}

TEST(VIOLiftTest, DiscreteLift) {
    vector<int> ids = {0, 1, 2, 3, 4};
    const double dt = 0.1;
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        const VIOState xi0 = randomStateElement(ids);
        const IMUVelocity velocity = randomVelocityElement();

        // Check the discrete result
        const VIOState xi1 = integrateSystemFunction(xi0, velocity, dt);

        const VIOGroup X = liftVelocityDiscrete(xi0, velocity, dt);
        const VIOState xi2 = stateGroupAction(X, xi0);

        const double dist12 = stateDistance(xi1, xi2);
        EXPECT_LE(dist12, NEAR_ZERO);
    }
}

void innovationLift_test(const EqFCoordinateSuite& coordinateSuite) {
    vector<int> ids = {0, 1, 2, 3, 4};
    const int N = ids.size();
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        const VIOState xi0 = randomStateElement(ids);
        const VIOGroup X = randomGroupElement(ids);
        MatrixXd Sigma = MatrixXd::Random(xi0.Dim(), xi0.Dim());
        Sigma = Sigma * Sigma.transpose(); // Makes Sigma positive (semi) definite

        auto reprojectionFunc = [&](const VectorXd& eps) {
            const VIOAlgebra liftedInnovation = coordinateSuite.liftInnovation(eps, xi0);
            const VIOGroup Delta = VIOExp(liftedInnovation);
            const VIOState xi1 = stateGroupAction(Delta, xi0);
            const VectorXd reprojectedInnovation = coordinateSuite.stateChart(xi1, xi0);
            return reprojectedInnovation;
        };

        testDifferential(reprojectionFunc, VectorXd::Zero(xi0.Dim()), MatrixXd::Identity(xi0.Dim(), xi0.Dim()));
    }
}

void discreteInnovationLift_test(const EqFCoordinateSuite& coordinateSuite) {
    vector<int> ids = {0, 1, 2, 3, 4};
    const int N = ids.size();
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        const VIOState xi0 = randomStateElement(ids);
        const VIOGroup X = randomGroupElement(ids);
        MatrixXd Sigma = MatrixXd::Random(xi0.Dim(), xi0.Dim());
        Sigma = Sigma * Sigma.transpose(); // Makes Sigma positive (semi) definite

        auto reprojectionFunc = [&](const VectorXd& eps) {
            const VIOGroup discreteInn = coordinateSuite.liftInnovationDiscrete(eps, xi0);
            const VIOState xi1 = stateGroupAction(discreteInn, xi0);
            const VectorXd reprojectedInnovation = coordinateSuite.stateChart(xi1, xi0);
            return reprojectedInnovation;
        };

        testDifferential(reprojectionFunc, VectorXd::Zero(xi0.Dim()), MatrixXd::Identity(xi0.Dim(), xi0.Dim()));
    }
}

TEST(VIOLiftTest, InnovationLifts_euclid) {
    innovationLift_test(EqFCoordinateSuite_euclid);
    discreteInnovationLift_test(EqFCoordinateSuite_euclid);
}

TEST(VIOLiftTest, InnovationLifts_invdepth) {
    innovationLift_test(EqFCoordinateSuite_invdepth);
    discreteInnovationLift_test(EqFCoordinateSuite_invdepth);
}