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

#include "eqvio/VIOState.h"
#include "liepp/SEn3.h"
#include <cmath>

using namespace std;
using namespace Eigen;
using namespace liepp;

Matrix3d skew(const Vector3d& vec);

VIOState integrateSystemFunction(const VIOState& state, const IMUVelocity& velocity, const double& dt) {
    VIOState newState;

    IMUVelocity v_est = velocity - state.sensor.inputBias;

    newState.sensor.inputBias.segment<3>(0) = state.sensor.inputBias.segment<3>(0) + dt * velocity.gyrBiasVel;
    newState.sensor.inputBias.segment<3>(3) = state.sensor.inputBias.segment<3>(3) + dt * velocity.accBiasVel;

    // Integrate the pose
    const VIOSensorState& sensor = state.sensor;
    VIOSensorState& newSensor = newState.sensor;
    SE3d::VectorDS poseVel;
    poseVel << v_est.gyr,
        sensor.velocity + 0.5 * dt * (v_est.acc + sensor.pose.R.inverse() * Vector3d(0, 0, -GRAVITY_CONSTANT));

    newSensor.pose = sensor.pose * SE3d::exp(dt * poseVel);

    // Integrate the velocity
    Vector3d inertialVelocityDiff = sensor.pose.R.asMatrix() * v_est.acc + Vector3d(0, 0, -GRAVITY_CONSTANT);
    newSensor.velocity = newSensor.pose.R.inverse() * (sensor.pose.R * sensor.velocity + dt * inertialVelocityDiff);

    // Landmarks are transformed in the body fixed frame
    const SE3d::VectorDS cameraPoseVel = sensor.cameraOffset.inverse().Adjoint() * poseVel;
    const SE3d cameraPoseChangeInv = SE3d::exp(-dt * cameraPoseVel);
    newState.cameraLandmarks.resize(state.cameraLandmarks.size());
    transform(
        state.cameraLandmarks.begin(), state.cameraLandmarks.end(), newState.cameraLandmarks.begin(),
        [&cameraPoseChangeInv](const Landmark& blm) {
            Landmark result;
            result.p = cameraPoseChangeInv * blm.p;
            result.id = blm.id;
            return result;
        });

    // Camera offset is constant
    newSensor.cameraOffset = sensor.cameraOffset;

    return newState;
}

VisionMeasurement measureSystemState(const VIOState& state, const GIFT::GICameraPtr& cameraPtr) {
    VisionMeasurement result;
    transform(
        state.cameraLandmarks.begin(), state.cameraLandmarks.end(),
        std::inserter(result.camCoordinates, result.camCoordinates.begin()),
        [&cameraPtr](const Landmark& lm) { return std::make_pair(lm.id, cameraPtr->projectPoint(lm.p)); });
    result.cameraPtr = cameraPtr;
    return result;
}

CSVLine& operator<<(CSVLine& line, const VIOState& state) {
    line << state.sensor;
    line << state.cameraLandmarks.size();
    for (const Landmark& blm : state.cameraLandmarks) {
        line << blm.id << blm.p;
    }
    return line;
}

CSVLine& operator<<(CSVLine& line, const VIOSensorState& sensor) {
    line << sensor.pose << sensor.velocity << sensor.cameraOffset << sensor.inputBias;
    return line;
}

Eigen::Vector3d VIOSensorState::gravityDir() const { return pose.R.inverse() * Vector3d::UnitZ(); }

vector<int> VIOState::getIds() const {
    vector<int> ids(cameraLandmarks.size());
    transform(cameraLandmarks.begin(), cameraLandmarks.end(), ids.begin(), [](const Landmark& pt) { return pt.id; });
    return ids;
}

int VIOState::Dim() const { return VIOSensorState::CompDim + Landmark::CompDim * cameraLandmarks.size(); }

const CoordinateChart<VIOSensorState> sensorChart_std{
    [](const VIOSensorState& Xi, const VIOSensorState& Xi0) {
        Vector<double, VIOSensorState::CompDim> eps;
        eps.segment<6>(0) = Xi.inputBias - Xi0.inputBias;
        eps.segment<6>(6) = SE3d::log(Xi0.pose.inverse() * Xi.pose);
        eps.segment<3>(12) = Xi.velocity - Xi0.velocity;
        eps.segment<6>(15) = SE3d::log(Xi0.cameraOffset.inverse() * Xi.cameraOffset);
        assert(!eps.hasNaN());
        return eps;
    },
    [](const Vector<double, VIOSensorState::CompDim>& eps, const VIOSensorState& Xi0) {
        VIOSensorState Xi;
        Xi.inputBias = Xi0.inputBias + eps.segment<6>(0);
        Xi.pose = Xi0.pose * SE3d::exp(eps.segment<6>(6));
        Xi.velocity = Xi0.velocity + eps.segment<3>(12);
        Xi.cameraOffset = Xi0.cameraOffset * SE3d::exp(eps.segment<6>(15));
        return Xi;
    }};

const CoordinateChart<VIOSensorState> sensorChart_normal{
    [](const VIOSensorState& Xi, const VIOSensorState& Xi0) {
        SE3d A = Xi0.pose.inverse() * Xi.pose;
        Vector3d w = Xi0.velocity - A.R * Xi.velocity;
        SE3d B = Xi0.cameraOffset.inverse() * A * Xi.cameraOffset;

        Vector<double, VIOSensorState::CompDim> eps;
        eps.segment<6>(0) = Xi.inputBias - Xi0.inputBias;
        eps.segment<9>(6) = SE23d::log(SE23d(A.R, {A.x, w}));
        eps.segment<6>(15) = SE3d::log(B);

        return eps;
    },
    [](const Vector<double, VIOSensorState::CompDim>& eps, const VIOSensorState& Xi0) {
        SE23d X = SE23d::exp(eps.segment<9>(6));
        SE3d B = SE3d::exp(eps.segment<6>(15));
        SE3d A = SE3d(X.R, X.x[0]);
        Vector3d w = X.x[1];

        VIOSensorState Xi;
        Xi.inputBias = Xi0.inputBias + eps.segment<6>(0);
        Xi.pose = Xi0.pose * A;
        Xi.velocity = A.R.inverse() * (Xi0.velocity - w);
        Xi.cameraOffset = A.inverse() * Xi0.cameraOffset * B;
        return Xi;
    }};

const CoordinateChart<Landmark> pointChart_euclid{
    [](const Landmark& q, const Landmark& q0) { return q.p - q0.p; },
    [](const Vector3d& eps, const Landmark& q0) {
        return Landmark{q0.p + eps, q0.id};
    }};

const CoordinateChart<Landmark> pointChart_invdepth{
    [](const Landmark& q, const Landmark& q0) {
        assert(q0.id == q.id);
        const double rho_i = 1.0 / q.p.norm();
        const double rho0_i = 1.0 / q0.p.norm();
        const Vector3d y_i = q.p * rho_i;
        const Vector3d y0_i = q0.p * rho0_i;

        // Store the bearing and then the inverse depth
        Vector3d eps;
        eps.segment<2>(0) = sphereChart_stereo(y_i, y0_i);
        eps(2) = rho_i - rho0_i;
        return eps;
    },
    [](const Vector3d& eps, const Landmark& q0) {
        const double rho0_i = 1.0 / q0.p.norm();
        const Vector3d y0_i = q0.p * rho0_i;

        // Retrieve bearing and inverse depth
        const Vector3d y_i = sphereChart_stereo.inv(eps.segment<2>(0), y0_i);
        double rho_i = eps(2) + rho0_i;
        if (rho_i <= 0.0) {
            rho_i = 1e-6;
            // throw(domain_error("The inverse depth cannot be negative."));
        }

        return Landmark{y_i / rho_i, q0.id};
    }};

const CoordinateChart<Landmark> pointChart_normal{
    [](const Landmark& q, const Landmark& q0) {
        assert(q0.id == q.id);
        const double rho_i = 1.0 / q.p.norm();
        const double rho0_i = 1.0 / q0.p.norm();
        const Vector3d y_i = q.p * rho_i;
        const Vector3d y0_i = q0.p * rho0_i;

        // Store the bearing and then the inverse depth
        Vector3d eps;
        eps.segment<2>(0) = sphereChart_normal(y_i, y0_i);
        eps(2) = log(rho_i / rho0_i);
        return eps;
    },
    [](const Vector3d& eps, const Landmark& q0) {
        const double rho0_i = 1.0 / q0.p.norm();
        const Vector3d y0_i = q0.p * rho0_i;

        // Retrieve bearing and inverse depth
        const Vector3d y_i = sphereChart_normal.inv(eps.segment<2>(0), y0_i);
        double rho_i = rho0_i * exp(eps(2));

        return Landmark{y_i / rho_i, q0.id};
    }};

const CoordinateChart<VIOState>
constructVIOChart(const CoordinateChart<VIOSensorState>& sensorChart, const CoordinateChart<Landmark>& pointChart) {

    const CoordinateChart<VIOState> chart{
        [&sensorChart, &pointChart](const VIOState& Xi, const VIOState& Xi0) {
            const size_t& N = Xi.cameraLandmarks.size();
            assert(N == Xi0.cameraLandmarks.size());
            VectorXd eps = VectorXd(VIOSensorState::CompDim + 3 * N);
            eps.segment<VIOSensorState::CompDim>(0) = sensorChart(Xi.sensor, Xi0.sensor);
            for (size_t i = 0; i < N; ++i) {
                eps.segment<3>(VIOSensorState::CompDim + 3 * i) =
                    pointChart(Xi.cameraLandmarks[i], Xi0.cameraLandmarks[i]);
            }
            return eps;
        },
        [&sensorChart, &pointChart](const VectorXd& eps, const VIOState& Xi0) {
            const size_t& N = Xi0.cameraLandmarks.size();
            VIOState Xi;
            Xi.sensor = sensorChart.inv(eps.segment<VIOSensorState::CompDim>(0), Xi0.sensor);
            Xi.cameraLandmarks.resize(N);
            for (size_t i = 0; i < N; ++i) {
                Xi.cameraLandmarks[i] =
                    pointChart.inv(eps.segment<3>(VIOSensorState::CompDim + 3 * i), Xi0.cameraLandmarks[i]);
            }
            return Xi;
        }};
    return chart;
}

const CoordinateChart<VIOState> VIOChart_invdepth = constructVIOChart(sensorChart_std, pointChart_invdepth);
const CoordinateChart<VIOState> VIOChart_euclid = constructVIOChart(sensorChart_std, pointChart_euclid);
const CoordinateChart<VIOState> VIOChart_normal = constructVIOChart(sensorChart_normal, pointChart_normal);

Eigen::Vector2d e3ProjectSphere(const Eigen::Vector3d& eta) {
    static const Matrix<double, 2, 3> I23 = Matrix<double, 2, 3>::Identity();
    static const Vector3d e3 = Vector3d(0, 0, 1);
    const Vector2d y = I23 * (eta - e3) / (1 - e3.dot(eta));
    return y;
}

Eigen::Vector3d e3ProjectSphereInv(const Eigen::Vector2d& y) {
    static const Vector3d e3 = Vector3d(0, 0, 1);
    const Vector3d yBar = (Vector3d() << y, 0).finished();
    const Vector3d eta = e3 + 2.0 / (yBar.squaredNorm() + 1) * (yBar - e3);
    return eta;
}

Eigen::Matrix<double, 2, 3> e3ProjectSphereDiff(const Eigen::Vector3d& eta) {
    static const Matrix<double, 2, 3> I23 = Matrix<double, 2, 3>::Identity();
    static const Vector3d e3 = Vector3d(0, 0, 1);
    Eigen::Matrix<double, 2, 3> Diff;
    Diff = I23 * (Matrix3d::Identity() * (1 - eta.z()) + (eta - e3) * e3.transpose());
    Diff = pow((1 - e3.dot(eta)), -2.0) * Diff;
    return Diff;
}

Eigen::Matrix<double, 3, 2> e3ProjectSphereInvDiff(const Eigen::Vector2d& y) {
    Eigen::Matrix<double, 3, 2> Diff;
    Diff.block<2, 2>(0, 0) = Matrix2d::Identity() * (y.squaredNorm() + 1.0) - 2 * y * y.transpose();
    Diff.block<1, 2>(2, 0) = 2 * y.transpose();
    Diff = 2.0 * pow((y.squaredNorm() + 1.0), -2.0) * Diff;
    return Diff;
}

Eigen::Vector2d sphereChart_stereo_impl(const Eigen::Vector3d& eta, const Eigen::Vector3d& pole);
Eigen::Vector3d sphereChart_stereo_inv_impl(const Eigen::Vector2d& y, const Eigen::Vector3d& pole);
Eigen::Matrix<double, 2, 3> sphereChart_stereo_diff0_impl(const Eigen::Vector3d& pole);
Eigen::Matrix<double, 3, 2> sphereChart_stereo_inv_diff0_impl(const Eigen::Vector3d& pole);

const EmbeddedCoordinateChart<3, 2> sphereChart_stereo{
    sphereChart_stereo_impl, sphereChart_stereo_inv_impl, sphereChart_stereo_diff0_impl,
    sphereChart_stereo_inv_diff0_impl};

Eigen::Vector2d sphereChart_stereo_impl(const Eigen::Vector3d& eta, const Eigen::Vector3d& pole) {
    const SO3d sphereRot = SO3d::SO3FromVectors(-pole, Eigen::Vector3d::Unit(2));
    const Vector3d etaRotated = sphereRot * eta;
    return e3ProjectSphere(etaRotated);
}

Eigen::Vector3d sphereChart_stereo_inv_impl(const Eigen::Vector2d& y, const Eigen::Vector3d& pole) {
    const Vector3d etaRotated = e3ProjectSphereInv(y);
    const SO3d sphereRot = SO3d::SO3FromVectors(-pole, Eigen::Vector3d::Unit(2));
    return sphereRot.inverse() * etaRotated;
}

Eigen::Matrix<double, 2, 3> sphereChart_stereo_diff0_impl(const Eigen::Vector3d& pole) {
    const SO3d sphereRot = SO3d::SO3FromVectors(-pole, Eigen::Vector3d::Unit(2));
    const Vector3d etaRotated = sphereRot * pole;
    return e3ProjectSphereDiff(etaRotated) * sphereRot.asMatrix();
}

Eigen::Matrix<double, 3, 2> sphereChart_stereo_inv_diff0_impl(const Eigen::Vector3d& pole) {
    const SO3d sphereRot = SO3d::SO3FromVectors(-pole, Eigen::Vector3d::Unit(2));
    return sphereRot.inverse().asMatrix() * e3ProjectSphereInvDiff(Vector2d::Zero());
}

const EmbeddedCoordinateChart<3, 2> sphereChart_normal{
    [](const Eigen::Vector3d& eta, const Eigen::Vector3d& pole) {
        const static Vector3d e3 = Eigen::Vector3d::Unit(2);
        const SO3d sphereRot = SO3d::SO3FromVectors(pole, e3);
        const Vector3d y = sphereRot * eta;

        const double sin_th = (SO3d::skew(y) * e3).norm();
        const double cos_th = y.transpose() * e3;

        const double th = atan2(sin_th, cos_th);
        Vector3d omega;
        if (abs(th) < 1e-8) {
            omega = SO3d::skew(y) * e3;
        } else {
            omega = SO3d::skew(y) * e3 * (th / sin_th);
        }

        return Vector2d(omega.segment<2>(0));
    },
    [](const Eigen::Vector2d& eps, const Eigen::Vector3d& pole) {
        const static Vector3d e3 = Eigen::Vector3d::Unit(2);
        const Vector3d omega = (Vector3d() << eps, 0.0).finished();
        const Vector3d y = SO3d::exp(-omega) * e3;

        const SO3d sphereRot = SO3d::SO3FromVectors(pole, e3);
        const Vector3d eta = sphereRot.inverse() * y;

        return eta;
    },
    [](const Eigen::Vector3d& pole) {
        const static Vector3d e3 = Eigen::Vector3d::Unit(2);
        const SO3d sphereRot = SO3d::SO3FromVectors(pole, e3);
        Matrix<double, 2, 3> diff;
        diff << 0.0, 1.0, 0.0, -1.0, 0.0, 0.0;
        diff = diff * sphereRot.asMatrix();
        return diff;
    },
    [](const Eigen::Vector3d& pole) {
        const static Vector3d e3 = Eigen::Vector3d::Unit(2);
        const SO3d sphereRot = SO3d::SO3FromVectors(pole, e3);
        Matrix<double, 3, 2> diff;
        diff << 0.0, -1.0, 1.0, 0.0, 0.0, 0.0;
        diff = sphereRot.inverse().asMatrix() * diff;
        return diff;
    }};

const MatrixXd coordinateDifferential_invdepth_euclid(const VIOState& Xi0) {
    // Coordinate can be changed between euclidean and inverse depth by
    // x_{id} = eps_{id} o eps_{eu}^{-1} (x_{eu})
    // This function computes the differential of the above operation and the chart origin xi0.

    // Rows and their corresponding state components
    // I am using zero indexing and half open ranges
    // [0,6): Pose deviation from Xi0.pose
    // [6,9) Body-fixed velocity
    // [9,15): Pose deviation from Xi0.cameraOffset
    // [9+3i,9+3(i+1)): Body-fixed landmark i (euclidean)

    // Cols and their corresponding state components
    // [0,6): Pose deviation from Xi0.pose
    // [6,9) Body-fixed velocity
    // [9,15): Pose deviation from Xi0.cameraOffset
    // [15+3i,15+3(i+1)): Body-fixed landmark i (inverse depth)

    const int& N = Xi0.cameraLandmarks.size();
    MatrixXd M = MatrixXd::Identity(VIOSensorState::CompDim + 3 * N, VIOSensorState::CompDim + 3 * N);

    for (int i = 0; i < N; ++i) {
        const double& rho0i = 1. / Xi0.cameraLandmarks[i].p.norm();
        const Vector3d& y0i = Xi0.cameraLandmarks[i].p * rho0i;

        Matrix3d Mi;
        Mi.block<2, 3>(0, 0) =
            rho0i * sphereChart_stereo.chartDiff0(y0i) * (Matrix3d::Identity() - y0i * y0i.transpose());
        Mi.block<1, 3>(2, 0) = -rho0i * rho0i * y0i.transpose();

        M.block<3, 3>(VIOSensorState::CompDim + 3 * i, VIOSensorState::CompDim + 3 * i) = Mi;
    }

    return M;
}

const MatrixXd coordinateDifferential_normal_euclid(const VIOState& Xi0) {

    auto coordChange = [&](const VectorXd& eps) {
        // return VIOChart_euclid.bundleChart(VIOChart_normal.bundleChart.inv(eps, Xi0), Xi0);
        return VIOChart_normal(VIOChart_euclid.inv(eps, Xi0), Xi0);
    };
    const int& N = Xi0.cameraLandmarks.size();
    const MatrixXd M = numericalDifferential(coordChange, VectorXd::Zero(VIOSensorState::CompDim + 3 * N));

    return M;
}
