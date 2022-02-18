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
#include "testing_utilities.h"
#include "gtest/gtest.h"

using namespace Eigen;
using namespace std;

TEST(EqFMatricesTest, euclid_invdepth_compatibility) {
    vector<int> ids = {0, 1, 2, 3, 4};
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {

        // Check that the charts are related by a change of basis
        const VIOState xi0 = randomStateElement(ids);
        const VIOGroup X = randomGroupElement(ids);
        const IMUVelocity vel = randomVelocityElement();

        const MatrixXd M = coordinateDifferential_invdepth_euclid(xi0);

        const MatrixXd A_euclid = EqFCoordinateSuite_euclid.stateMatrixA(X, xi0, vel);
        const MatrixXd A_invdepth = EqFCoordinateSuite_invdepth.stateMatrixA(X, xi0, vel);
        const MatrixXd A_diff = A_invdepth - M * A_euclid * M.inverse();
        EXPECT_LE(A_diff.norm(), 1e-6);

        const MatrixXd B_euclid = EqFCoordinateSuite_euclid.inputMatrixB(X, xi0);
        const MatrixXd B_invdepth = EqFCoordinateSuite_invdepth.inputMatrixB(X, xi0);
        const MatrixXd B_diff = B_invdepth - M * B_euclid;
        EXPECT_LE(B_diff.norm(), 1e-6);

        const GIFT::GICameraPtr camPtr = createDefaultCamera();
        const VIOState xiHat = stateGroupAction(X, xi0);
        const VisionMeasurement yHat = measureSystemState(xiHat, camPtr);

        const MatrixXd C_euclid = EqFCoordinateSuite_euclid.outputMatrixC(xi0, X, yHat);
        const MatrixXd C_invdepth = EqFCoordinateSuite_invdepth.outputMatrixC(xi0, X, yHat);
        const MatrixXd C_diff = C_invdepth - C_euclid * M.inverse();
        EXPECT_LE(C_diff.norm(), 1e-4);
    }
}

class EqFSuiteTest : public testing::TestWithParam<EqFCoordinateSuite> {};

TEST_P(EqFSuiteTest, stateMatrixA) {
    srand(0);
    const EqFCoordinateSuite& coordinateSuite = GetParam();
    vector<int> ids = {0, 1, 2, 3, 4};
    const int N = ids.size();
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {

        // Set some random conditions
        const VIOState xi0 = reasonableStateElement(ids);
        const VIOGroup XHat = reasonableGroupElement(ids);
        const IMUVelocity vel = randomVelocityElement();
        const MatrixXd A0t = coordinateSuite.stateMatrixA(XHat, xi0, vel);

        // Compare the A matrix to the lemma from which it is derived
        // The lemma says
        // A0 = Deps . Dphi_{X^-1} . Dphi_{xi} . DLambda_v . Dphi_X . Deps^{-1} (0)
        // Let a0(delta) = eps o phi_{X^-1} o phi_xi o exp o LambdaTilde_v o phi_X o eps^{-1} (delta)
        // Then we expect that
        //  (a0(dt * epsilon) - a0(0)) / dt --> A0 epsilon
        // as dt --> 0, for any vector epsilon

        auto a0 = [&](const VectorXd& epsilon) {
            const auto xi_hat = stateGroupAction(XHat, xi0);
            const auto xi_e = coordinateSuite.stateChart.inv(epsilon, xi0);
            const auto xi = stateGroupAction(XHat, xi_e);
            const auto LambdaTilde = liftVelocity(xi, vel) - liftVelocity(xi_hat, vel);
            const auto xi_hat1 = stateGroupAction(VIOExp(LambdaTilde), xi_hat);
            const auto xi_e1 = stateGroupAction(XHat.inverse(), xi_hat1);
            const VectorXd epsilon1 = coordinateSuite.stateChart(xi_e1, xi0);
            return epsilon1;
        };

        // Check the function at zero
        const VectorXd a0AtZero = a0(VectorXd::Zero(xi0.Dim()));
        EXPECT_LE(a0AtZero.norm(), NEAR_ZERO);

        testDifferential(a0, VectorXd::Zero(xi0.Dim()), A0t);
    }
}

TEST_P(EqFSuiteTest, inputMatrixB) {
    srand(0);
    const EqFCoordinateSuite& coordinateSuite = GetParam();
    vector<int> ids = {0, 1, 2, 3, 4};
    const int N = ids.size();
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {

        // Set some random conditions
        const VIOState xi0 = reasonableStateElement(ids);
        const VIOGroup XHat = reasonableGroupElement(ids);
        const MatrixXd Bt = coordinateSuite.inputMatrixB(XHat, xi0);
        const IMUVelocity vel = randomVelocityElement();

        // Compare the A matrix to the lemma from which it is derived
        // The lemma says
        // A0 = Deps . Dphi_{X^-1} . Dphi_{xi} . DLambda_v . Dphi_X . Deps^{-1} (0)
        // Let a0(delta) = eps o phi_{X^-1} o phi_xi o exp o LambdaTilde_v o phi_X o eps^{-1} (delta)
        // Then we expect that
        //  (a0(dt * epsilon) - a0(0)) / dt --> A0 epsilon
        // as dt --> 0, for any vector epsilon

        auto b0 = [&](const Matrix<double, 12, 1>& vel_err_vec) {
            const auto xi_hat = stateGroupAction(XHat, xi0);
            const auto vel_err = IMUVelocity(vel_err_vec);
            const auto LambdaTilde = liftVelocity(xi_hat, vel + vel_err) - liftVelocity(xi_hat, vel);
            const auto xi_hat1 = stateGroupAction(VIOExp(LambdaTilde), xi_hat);
            const auto xi_e1 = stateGroupAction(XHat.inverse(), xi_hat1);
            const VectorXd epsilon = coordinateSuite.stateChart(xi_e1, xi0);
            return epsilon;
        };

        // Check the function at zero
        const VectorXd b0AtZero = b0(Matrix<double, 12, 1>::Zero());
        EXPECT_LE(b0AtZero.norm(), NEAR_ZERO);

        testDifferential(b0, Matrix<double, 12, 1>::Zero(), Bt);
    }
}

TEST_P(EqFSuiteTest, outputMatrixC) {
    srand(0);
    const EqFCoordinateSuite& coordinateSuite = GetParam();
    vector<int> ids = {5, 0, 1, 2, 3, 4};
    const int N = ids.size();
    const GIFT::GICameraPtr camPtr = createDefaultCamera();

    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {

        // Set some random conditions
        const VIOState xi0 = reasonableStateElement(ids);
        const VIOGroup XHat = reasonableGroupElement(ids);
        const VIOState xiHat = stateGroupAction(XHat, xi0);
        const VisionMeasurement yHat = measureSystemState(xiHat, camPtr);
        const MatrixXd Ct = coordinateSuite.outputMatrixC(xi0, XHat, yHat);
        const MatrixXd Ct2 = coordinateSuite.outputMatrixC(xi0, XHat, yHat, false);
        assertMatrixEquality(Ct, Ct2);

        // Compare the C matrix to the expression from which it is derived
        // Ct = Dh(xiHat) . Dphi_XHat(xiHat) . Deps^{-1} (0)
        // Let ct(epsilon) = h o phi_XHat o eps^{-1} (epsilon)
        // Then we expect that
        // (ct(dt * epsilon) - ct(0)) / dt --> Ct epsilon
        // as dt --> 0, for any vector epsilon

        auto ct = [&](const VectorXd& epsilon) {
            const auto xi_e = coordinateSuite.stateChart.inv(epsilon, xi0);
            const auto xi = stateGroupAction(XHat, xi_e);
            const auto y = measureSystemState(xi, camPtr);
            const VectorXd yTilde = y - yHat;
            return yTilde;
        };

        // Check the function at zero
        const VectorXd c0AtZero = ct(VectorXd::Zero(xi0.Dim()));
        EXPECT_LE(c0AtZero.norm(), NEAR_ZERO);

        const double floatStep = std::cbrt(std::numeric_limits<float>::epsilon());
        testDifferential(ct, VectorXd::Zero(xi0.Dim()), Ct, floatStep);
    }
}

TEST(EqFSuiteTest, outputMatrixCStar) {
    srand(0);
    const EqFCoordinateSuite& coordinateSuite = EqFCoordinateSuite_euclid;
    const GIFT::GICameraPtr camPtr = createDefaultCamera();

    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {

        // Set some random conditions
        const Vector3d q0 = Vector3d::Random() * 10 + Vector3d(0, 0, 20.0);
        liepp::SOT3d QHat;
        QHat.R = liepp::SO3d::exp(Vector3d::Random() * 0.02);
        QHat.a = 2.0 * (double)rand() / RAND_MAX + 1.0;

        const Vector3d qHat = QHat.inverse() * q0;
        const Vector2d yHat = camPtr->projectPoint(qHat);

        // CtStar depends on the measurement y
        const auto CtStar = [&q0, &QHat, &camPtr, &coordinateSuite](const Vector2d& y) {
            return coordinateSuite.outputMatrixCiStar(q0, QHat, camPtr, y);
        };
        const MatrixXd Ct = coordinateSuite.outputMatrixCi(q0, QHat, camPtr);

        // Compare the C matrix to the true output residual
        // We expect CtStar is a better approximation to hTilde than Ct

        auto hFunc = [&](const Vector3d& epsilon) {
            // Normal coordinates from euclidean coordinates
            liepp::SOT3d::VectorDS eps_normal;
            eps_normal.segment<3>(0) = -liepp::SO3d::skew(q0) * epsilon;
            eps_normal(3) = -q0.transpose() * epsilon;
            eps_normal = eps_normal / q0.squaredNorm();

            const auto q_e = liepp::SOT3d::exp(-eps_normal) * q0;
            const auto q = QHat.inverse() * q_e;
            const auto y = camPtr->projectPoint(q);
            return y;
        };

        // TODO: this test is a mess
        const double floatStep = 100.0 * std ::cbrt(std::numeric_limits<float>::epsilon());
        for (int j = 0; j < Ct.cols(); ++j) {
            const Eigen::Vector3d ej = Eigen::VectorXd::Unit(Ct.cols(), j);
            const Eigen::Vector3d eps = floatStep * ej;
            const Eigen::Vector2d yTrue = hFunc(eps);
            const Eigen::Vector2d yTilde = yTrue - yHat;

            const Eigen::Matrix<double, 2, 3> CtS = CtStar(yTrue);
            const Eigen::Vector2d yTildeStar = CtS * eps;
            const Eigen::Vector2d yTildeEst0 = Ct * eps;

            const Eigen::Matrix<double, 2, 3> CtS2 = 0.5 * (Ct + numericalDifferential(hFunc, eps, floatStep));

            const double linErrorEst0 = (yTildeEst0 - yTilde).norm();
            const double linErrorStar = (yTildeStar - yTilde).norm();

            EXPECT_LE(linErrorStar, linErrorEst0) << "Direction " << j << std::endl;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    EqFSuites, EqFSuiteTest,
    testing::Values(EqFCoordinateSuite_euclid, EqFCoordinateSuite_invdepth, EqFCoordinateSuite_normal));
