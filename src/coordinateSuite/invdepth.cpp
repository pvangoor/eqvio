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

#include "eqvio/EqFMatrices.h"

using namespace Eigen;
using namespace std;
using namespace liepp;

Eigen::MatrixXd EqFStateMatrixA_invdepth(const VIOGroup& X, const VIOState& xi0, const IMUVelocity& imuVel);
Eigen::MatrixXd EqFInputMatrixB_invdepth(const VIOGroup& X, const VIOState& xi0);
Eigen::Matrix<double, 2, 3> EqFoutputMatrixCiStar_invdepth(
    const Vector3d& q0, const SOT3d& QHat, const GIFT::GICameraPtr& camPtr, const Eigen::Vector2d& y);

VIOAlgebra liftInnovation_invdepth(const Eigen::VectorXd& totalInnovation, const VIOState& xi0);
VIOGroup liftInnovationDiscrete_invdepth(const Eigen::VectorXd& totalInnovation, const VIOState& xi0);

const EqFCoordinateSuite EqFCoordinateSuite_invdepth{VIOChart_invdepth,        EqFStateMatrixA_invdepth,
                                                     EqFInputMatrixB_invdepth, EqFoutputMatrixCiStar_invdepth,
                                                     liftInnovation_invdepth,  liftInnovationDiscrete_invdepth};

Eigen::MatrixXd EqFStateMatrixA_invdepth(const VIOGroup& X, const VIOState& xi0, const IMUVelocity& imuVel) {
    const int N = xi0.cameraLandmarks.size();
    MatrixXd A0t = MatrixXd::Zero(xi0.Dim(), xi0.Dim());

    // Rows / Cols and their corresponding state components
    // I am using zero indexing and half open ranges
    // [0,6): Input bias (gyr/acc)
    // [6,12): Pose attitude and position
    // [12,15): Body-fixed velocity
    // [15,21): Camera Offset from IMU
    // [21+3i,21+3(i+1)): Body-fixed landmark i

    // Effect of bias
    A0t.block(0, 0, xi0.Dim(), 6) = -EqFInputMatrixB_invdepth(X, xi0).block(0, 0, xi0.Dim(), 6);

    // Effect of velocity on translation
    A0t.block<3, 3>(9, 12) = Matrix3d::Identity();

    // Effect of gravity cov on velocity cov
    A0t.block<3, 3>(12, 6) = -GRAVITY_CONSTANT * SO3d::skew(xi0.sensor.gravityDir());

    const VIOState xi_hat = stateGroupAction(X, xi0);
    const auto v_est = imuVel - xi_hat.sensor.inputBias;
    const se3d U_I = (se3d() << v_est.gyr, xi_hat.sensor.velocity).finished();

    // Effect of camera offset cov on self
    // Formula is \ad(\Ad_{\mr{T}}^{-1} \Ad_{\hat{A}} U_I)
    A0t.block<6, 6>(15, 15) = SE3d::adjoint(xi0.sensor.cameraOffset.inverse().Adjoint() * X.A.Adjoint() * U_I);

    const auto conv_euc2ind = [](const Vector3d& q0) {
        const double& rho0i = 1. / q0.norm();
        const Vector3d& y0i = q0 * rho0i;
        Matrix3d conv_M;
        conv_M.block<2, 3>(0, 0) =
            rho0i * sphereChart_stereo.chartDiff0(y0i) * (Matrix3d::Identity() - y0i * y0i.transpose());
        conv_M.block<1, 3>(2, 0) = -rho0i * rho0i * y0i.transpose();
        return conv_M;
    };
    const auto conv_ind2euc = [](const Vector3d& q0) {
        const double& rho0i = 1. / q0.norm();
        const Vector3d& y0i = q0 * rho0i;
        Matrix3d conv_M;
        conv_M.block<3, 2>(0, 0) = sphereChart_stereo.chartInvDiff0(y0i) / rho0i;
        conv_M.block<3, 1>(0, 2) = -y0i / (rho0i * rho0i);
        return conv_M;
    };

    // Effect of velocity cov on landmarks cov
    const Matrix3d R_IC = xi_hat.sensor.cameraOffset.R.asMatrix();
    const Matrix3d R_Ahat = X.A.R.asMatrix();
    for (int i = 0; i < N; ++i) {
        const Vector3d& q0 = xi0.cameraLandmarks[i].p;
        const Matrix3d Qhat_i = X.Q[i].R.asMatrix() * X.Q[i].a;
        A0t.block<3, 3>(VIOSensorState::CompDim + 3 * i, 12) =
            -conv_euc2ind(q0) * Qhat_i * R_IC.transpose() * R_Ahat.transpose();
    }

    // Effect of camera offset cov on landmarks cov
    const Matrix<double, 6, 6> commonTerm =
        X.B.inverse().Adjoint() * SE3d::adjoint(xi0.sensor.cameraOffset.inverse().Adjoint() * X.A.Adjoint() * U_I);
    for (int i = 0; i < N; ++i) {
        const Vector3d& q0 = xi0.cameraLandmarks[i].p;
        Matrix<double, 3, 6> temp;
        temp << SO3d::skew(q0) * X.Q[i].R.asMatrix(), -X.Q[i].a * X.Q[i].R.asMatrix();

        A0t.block<3, 6>(VIOSensorState::CompDim + 3 * i, 15) = conv_euc2ind(q0) * temp * commonTerm;
    }

    // Effect of landmark cov on landmark cov
    const se3d U_C = xi_hat.sensor.cameraOffset.inverse().Adjoint() * U_I;
    const Vector3d v_C = U_C.block<3, 1>(3, 0);
    for (int i = 0; i < N; ++i) {
        const Vector3d& q0 = xi0.cameraLandmarks[i].p;
        const Matrix3d Qhat_i = X.Q[i].R.asMatrix() * X.Q[i].a;
        const Vector3d& qhat_i = xi_hat.cameraLandmarks[i].p;
        const Matrix3d A_qi =
            -Qhat_i * (SO3d::skew(qhat_i) * SO3d::skew(v_C) - 2 * v_C * qhat_i.transpose() + qhat_i * v_C.transpose()) *
            Qhat_i.inverse() * (1 / qhat_i.squaredNorm());
        A0t.block<3, 3>(VIOSensorState::CompDim + 3 * i, VIOSensorState::CompDim + 3 * i) =
            conv_euc2ind(q0) * A_qi * conv_ind2euc(q0);
    }

    assert(!A0t.hasNaN());

    return A0t;
}

Eigen::MatrixXd EqFInputMatrixB_invdepth(const VIOGroup& X, const VIOState& xi0) {

    const auto conv_euc2ind = [](const Vector3d& q0) {
        const double& rho0i = 1. / q0.norm();
        const Vector3d& y0i = q0 * rho0i;
        Matrix3d conv_M;
        conv_M.block<2, 3>(0, 0) =
            rho0i * sphereChart_stereo.chartDiff0(y0i) * (Matrix3d::Identity() - y0i * y0i.transpose());
        conv_M.block<1, 3>(2, 0) = -rho0i * rho0i * y0i.transpose();
        return conv_M;
    };

    const int N = xi0.cameraLandmarks.size();
    MatrixXd Bt = MatrixXd::Zero(xi0.Dim(), IMUVelocity::CompDim);

    // Rows and their corresponding state components
    // I am using zero indexing and half open ranges
    // [0,6): Input bias (gyr/acc)
    // [6,12): Pose attitude and position
    // [12,15): Body-fixed velocity
    // [15,21): Camera Offset from IMU
    // [21+3i,21+3(i+1)): Body-fixed landmark i

    // Cols and their corresponding output components
    // [0, 3): Angular velocity gyr
    // [3, 6): Linear acceleration acc
    // [6, 9): gyr bias velocity
    // [9, 12): acc bias velocity

    const VIOState xi_hat = stateGroupAction(X, xi0);

    // Biases
    Bt.block<6, 6>(0, 6) = Matrix<double, 6, 6>::Identity();

    // Attitude
    const Matrix3d R_A = X.A.R.asMatrix();
    Bt.block<3, 3>(6, 0) = R_A;

    // Position
    Bt.block<3, 3>(9, 0) = SO3d::skew(X.A.x) * R_A;

    // Body fixed velocity
    Bt.block<3, 3>(12, 0) = R_A * SO3d::skew(xi_hat.sensor.velocity);
    Bt.block<3, 3>(12, 3) = R_A;

    // Landmarks
    const Matrix3d RT_IC = xi_hat.sensor.cameraOffset.R.inverse().asMatrix();
    const Vector3d x_IC = xi_hat.sensor.cameraOffset.x;
    for (int i = 0; i < N; ++i) {
        const Vector3d& q0 = xi0.cameraLandmarks[i].p;
        const Matrix3d Qhat_i = X.Q[i].R.asMatrix() * X.Q[i].a;
        const Vector3d& qhat_i = xi_hat.cameraLandmarks[i].p;
        Bt.block<3, 3>(VIOSensorState::CompDim + 3 * i, 0) =
            conv_euc2ind(q0) * Qhat_i * (SO3d::skew(qhat_i) * RT_IC + RT_IC * SO3d::skew(x_IC));
    }

    assert(!Bt.hasNaN());
    return Bt;
}

VIOAlgebra liftInnovation_invdepth(const Eigen::VectorXd& totalInnovation, const VIOState& xi0) {
    assert(totalInnovation.size() == xi0.Dim());
    VIOAlgebra Delta;

    Delta.u_beta = totalInnovation.segment<6>(0);

    // Delta_A
    Delta.U_A = totalInnovation.segment<6>(6);

    // Delta w
    const Vector3d& gamma_v = totalInnovation.segment<3>(12);
    Delta.u_w = -gamma_v - SO3d::skew(Delta.U_A.segment<3>(0)) * xi0.sensor.velocity;

    // Delta_B
    Delta.U_B = totalInnovation.segment<6>(15) + xi0.sensor.cameraOffset.inverse().Adjoint() * Delta.U_A;

    // Delta q_i
    const int N = xi0.cameraLandmarks.size();
    Delta.id.resize(N);
    Delta.W.resize(N);
    for (int i = 0; i < N; ++i) {
        const Vector3d& qi0 = xi0.cameraLandmarks[i].p;
        const double& r0 = qi0.norm();
        const Vector3d& y0 = qi0 / r0;

        Eigen::Matrix3d ind2euc;
        ind2euc.block<3, 2>(0, 0) = r0 * sphereChart_stereo.chartInvDiff0(y0);
        ind2euc.block<3, 1>(0, 2) = -r0 * qi0;

        const Vector3d& gamma_qi0 = ind2euc * totalInnovation.segment<3>(VIOSensorState::CompDim + 3 * i);

        // Rotation part
        Delta.W[i].block<3, 1>(0, 0) = -qi0.cross(gamma_qi0) / qi0.squaredNorm();
        // scale part
        Delta.W[i](3) = -qi0.dot(gamma_qi0) / qi0.squaredNorm();
        // id number
        Delta.id[i] = xi0.cameraLandmarks[i].id;
    }

    return Delta;
}

VIOGroup liftInnovationDiscrete_invdepth(const Eigen::VectorXd& totalInnovation, const VIOState& xi0) {
    // Lift the innovation discretely
    VIOGroup lift;
    lift.beta = totalInnovation.segment<6>(0);
    lift.A = SE3d::exp(totalInnovation.segment<6>(6));
    lift.w = xi0.sensor.velocity - lift.A.R * (xi0.sensor.velocity + totalInnovation.segment<3>(12));
    lift.B = xi0.sensor.cameraOffset.inverse() * lift.A * xi0.sensor.cameraOffset *
             SE3d::exp(totalInnovation.segment<6>(15));

    // Lift for each of the points
    const int N = xi0.cameraLandmarks.size();
    lift.id.resize(N);
    lift.Q.resize(N);
    for (int i = 0; i < N; ++i) {
        const Landmark& q0i = xi0.cameraLandmarks[i];

        const Vector3d& Gamma_q0i = totalInnovation.segment<3>(VIOSensorState::CompDim + 3 * i);
        const Landmark& q1i = pointChart_invdepth.inv(Gamma_q0i, q0i);

        lift.Q[i].R = SO3d::SO3FromVectors(q1i.p.normalized(), q0i.p.normalized());
        lift.Q[i].a = q0i.p.norm() / q1i.p.norm();

        lift.id[i] = q0i.id;
    }

    assert(!lift.hasNaN());

    return lift;
}

Eigen::Matrix<double, 2, 3> EqFoutputMatrixCiStar_invdepth(
    const Vector3d& q0, const SOT3d& QHat, const GIFT::GICameraPtr& camPtr, const Eigen::Vector2d& y) {
    const double& r0 = q0.norm();
    const Vector3d& y0 = q0 / r0;

    Eigen::Matrix3d ind2euc;
    ind2euc.block<3, 2>(0, 0) = r0 * sphereChart_stereo.chartInvDiff0(y0);
    ind2euc.block<3, 1>(0, 2) = -r0 * q0;

    Eigen::Matrix<double, 2, 3> C0i = EqFCoordinateSuite_euclid.outputMatrixCiStar(q0, QHat, camPtr, y) * ind2euc;
    return C0i;
}