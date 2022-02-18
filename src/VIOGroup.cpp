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

#include "eqvio/VIOGroup.h"
#include "liepp/SEn3.h"

using namespace Eigen;
using namespace std;
using namespace liepp;

[[nodiscard]] VIOSensorState sensorStateGroupAction(const VIOGroup& X, const VIOSensorState& sensor) {
    VIOSensorState result;
    result.inputBias = sensor.inputBias + X.beta;
    result.pose = sensor.pose * X.A;
    result.velocity = X.A.R.inverse() * (sensor.velocity - X.w);
    result.cameraOffset = X.A.inverse() * sensor.cameraOffset * X.B;
    return result;
}

VIOState stateGroupAction(const VIOGroup& X, const VIOState& state) {
    VIOState newState;
    newState.sensor = sensorStateGroupAction(X, state.sensor);

    // Check the landmarks and transforms are aligned.
    assert(X.Q.size() == state.cameraLandmarks.size());
    for (size_t i = 0; i < X.Q.size(); ++i)
        assert(X.id[i] == state.cameraLandmarks[i].id);

    // Transform the body-fixed landmarks
    newState.cameraLandmarks.resize(state.cameraLandmarks.size());
    transform(
        state.cameraLandmarks.begin(), state.cameraLandmarks.end(), X.Q.begin(), newState.cameraLandmarks.begin(),
        [](const Landmark& lm, const SOT3d& Q) {
            Landmark result;
            result.p = Q.inverse() * lm.p;
            result.id = lm.id;
            return result;
        });

    return newState;
}

VisionMeasurement outputGroupAction(const VIOGroup& X, const VisionMeasurement& measurement) {
    // Transform the measurements
    VisionMeasurement newMeasurements;
    for (size_t i = 0; i < X.Q.size(); ++i) {
        const auto it = measurement.camCoordinates.find(X.id[i]);
        if (it != measurement.camCoordinates.end()) {
            const Vector3d bearing = measurement.cameraPtr->undistortPoint(it->second);
            newMeasurements.camCoordinates[X.id[i]] = measurement.cameraPtr->projectPoint(X.Q[i].R.inverse() * bearing);
        }
    }
    newMeasurements.cameraPtr = measurement.cameraPtr;
    return newMeasurements;
}

VIOGroup VIOGroup::operator*(const VIOGroup& other) const {
    VIOGroup result;

    result.beta = this->beta + other.beta;
    result.A = this->A * other.A;
    result.B = this->B * other.B;
    result.w = this->w + this->A.R * other.w;

    // Check the transforms are aligned.
    assert(this->Q.size() == other.Q.size());
    assert(this->id.size() == other.id.size());
    for (size_t i = 0; i < this->Q.size(); ++i)
        assert(this->id[i] == other.id[i]);

    result.Q.resize(this->Q.size());
    transform(
        this->Q.begin(), this->Q.end(), other.Q.begin(), result.Q.begin(),
        [](const SOT3d& Qi1, const SOT3d& Qi2) { return Qi1 * Qi2; });
    result.id = this->id;

    return result;
}

VIOGroup VIOGroup::Identity(const vector<int>& id) {
    VIOGroup result;
    result.beta = Eigen::Matrix<double, 6, 1>::Zero();
    result.A = SE3d::Identity();
    result.B = SE3d::Identity();
    result.w = Vector3d::Zero();
    result.id = id;
    result.Q.resize(id.size());
    for (SOT3d& Qi : result.Q) {
        Qi.setIdentity();
    }
    return result;
}

VIOGroup VIOGroup::inverse() const {
    VIOGroup result;
    result.beta = -beta;
    result.A = A.inverse();
    result.B = B.inverse();
    result.w = -(A.R.inverse() * w);

    result.Q = Q;
    for_each(result.Q.begin(), result.Q.end(), [](SOT3d& Qi) { Qi = Qi.inverse(); });
    result.id = this->id;

    return result;
}

bool VIOGroup::hasNaN() const {
    bool nanFound = false;
    nanFound |= beta.hasNaN();
    nanFound |= A.asMatrix().hasNaN();
    nanFound |= B.asMatrix().hasNaN();
    nanFound |= w.hasNaN();
    nanFound |= std::any_of(Q.begin(), Q.end(), [](const SOT3d& Qi) { return Qi.asMatrix().hasNaN(); });
    return nanFound;
}

CSVLine& operator<<(CSVLine& line, const VIOGroup& X) {
    line << X.beta << X.A << X.w << X.B;
    line << X.id.size();
    for (size_t i = 0; i < X.id.size(); ++i) {
        line << X.id[i] << X.Q[i];
    }
    return line;
}

VIOAlgebra VIOAlgebra::operator*(const double& c) const {
    VIOAlgebra result;
    result.u_beta = this->u_beta * c;
    result.U_A = this->U_A * c;
    result.U_B = this->U_B * c;
    result.u_w = this->u_w * c;
    result.W.resize(this->W.size());
    transform(W.begin(), W.end(), result.W.begin(), [&c](const Vector4d& Wi) { return c * Wi; });
    result.id = this->id;

    return result;
}

VIOAlgebra VIOAlgebra::operator-() const {
    VIOAlgebra result;
    result.u_beta = -this->u_beta;
    result.U_A = -this->U_A;
    result.U_B = -this->U_B;
    result.u_w = -this->u_w;
    result.W.resize(this->W.size());
    transform(W.begin(), W.end(), result.W.begin(), [](const Vector4d& Wi) { return -Wi; });
    result.id = this->id;

    return result;
}

VIOAlgebra VIOAlgebra::operator+(const VIOAlgebra& other) const {
    VIOAlgebra result;
    result.u_beta = this->u_beta + other.u_beta;
    result.U_A = this->U_A + other.U_A;
    result.U_B = this->U_B + other.U_B;
    result.u_w = this->u_w + other.u_w;

    assert(this->id.size() == other.id.size());
    for (size_t i = 0; i < this->id.size(); ++i) {
        assert(this->id[i] == other.id[i]);
    }

    result.W.resize(this->W.size());
    transform(
        this->W.begin(), this->W.end(), other.W.begin(), result.W.begin(),
        [](const Vector4d& Wi1, const Vector4d& Wi2) { return Wi1 + Wi2; });
    result.id = this->id;

    return result;
}

VIOAlgebra VIOAlgebra::operator-(const VIOAlgebra& other) const { return *this + (-other); }

[[nodiscard]] VIOAlgebra liftVelocity(const VIOState& state, const IMUVelocity& velocity) {
    VIOAlgebra lift;

    const VIOSensorState& sensor = state.sensor;
    const auto v_est = velocity - state.sensor.inputBias;

    lift.u_beta = (Matrix<double, 6, 1>() << velocity.gyrBiasVel, velocity.accBiasVel).finished();

    // Set the SE(3) velocity
    lift.U_A.setZero();
    lift.U_A.block<3, 1>(0, 0) = v_est.gyr;
    lift.U_A.block<3, 1>(3, 0) = sensor.velocity;

    lift.U_B = sensor.cameraOffset.inverse().Adjoint() * lift.U_A;

    // Set the R3 velocity
    lift.u_w = -v_est.acc + sensor.gravityDir() * GRAVITY_CONSTANT;

    // Set the landmark transform velocities
    const se3d U_C = sensor.cameraOffset.inverse().Adjoint() * lift.U_A;
    lift.W.resize(state.cameraLandmarks.size());
    transform(state.cameraLandmarks.begin(), state.cameraLandmarks.end(), lift.W.begin(), [&U_C](const Landmark& blm) {
        Vector4d result;
        const Vector3d& omega_C = U_C.block<3, 1>(0, 0);
        const Vector3d& v_C = U_C.block<3, 1>(3, 0);
        result.block<3, 1>(0, 0) = omega_C + SO3d::skew(blm.p) * v_C / blm.p.squaredNorm();
        result(3) = blm.p.dot(v_C) / blm.p.squaredNorm();
        return result;
    });

    // Set the lift ids
    lift.id.resize(state.cameraLandmarks.size());
    transform(state.cameraLandmarks.begin(), state.cameraLandmarks.end(), lift.id.begin(), [](const Landmark& blm) {
        return blm.id;
    });

    return lift;
}

[[nodiscard]] VIOGroup liftVelocityDiscrete(const VIOState& state, const IMUVelocity& velocity, const double& dt) {
    // Lift the velocity discretely
    VIOGroup lift;

    const VIOSensorState& sensor = state.sensor;
    const auto v_est = velocity - state.sensor.inputBias;

    lift.beta = dt * (Matrix<double, 6, 1>() << velocity.gyrBiasVel, velocity.accBiasVel).finished();

    // Set the SE(3) velocity
    Matrix<double, 6, 1> AVel;
    AVel << v_est.gyr, sensor.velocity + 0.5 * dt * (v_est.acc - sensor.gravityDir() * GRAVITY_CONSTANT);
    lift.A = SE3d::exp(dt * AVel);

    lift.B = sensor.cameraOffset.inverse() * lift.A * sensor.cameraOffset;

    // Set the R3 velocity
    Vector3d bodyVelocityDiff = v_est.acc - sensor.gravityDir() * GRAVITY_CONSTANT;
    lift.w = sensor.velocity - (sensor.velocity + dt * bodyVelocityDiff);

    // Set the landmark transform velocities
    const int N = state.cameraLandmarks.size();
    const se3d U_C = sensor.cameraOffset.inverse().Adjoint() * AVel;
    const SE3d cameraPoseChangeInv = SE3d::exp(-dt * U_C);
    lift.Q.resize(N);
    lift.id.resize(N);
    for (int i = 0; i < N; ++i) {
        const Landmark& blm0 = state.cameraLandmarks[i];
        Landmark blm1;
        blm1.id = blm0.id;
        blm1.p = cameraPoseChangeInv * blm0.p;

        // Find the transform to take blm1 to blm0
        lift.Q[i].R = SO3d::SO3FromVectors(blm1.p.normalized(), blm0.p.normalized());
        lift.Q[i].a = blm0.p.norm() / blm1.p.norm();
        lift.id[i] = blm1.id;
    }

    return lift;
}

[[nodiscard]] VIOGroup VIOExp(const VIOAlgebra& lambda) {
    SE23d::VectorDS extPoseAwVel;
    extPoseAwVel << lambda.U_A, lambda.u_w;
    SE23d extPoseAw = SE23d::exp(extPoseAwVel);

    VIOGroup result;
    result.beta = lambda.u_beta;
    result.A = SE3d(extPoseAw.R, extPoseAw.x[0]);
    result.w = extPoseAw.x[1];

    result.B = SE3d::exp(lambda.U_B);

    result.id = lambda.id;
    result.Q.resize(lambda.W.size());
    transform(lambda.W.begin(), lambda.W.end(), result.Q.begin(), [](const Vector4d& Wi) { return SOT3d::exp(Wi); });

    return result;
}

[[nodiscard]] VIOAlgebra operator*(const double& c, const VIOAlgebra& lambda) { return lambda * c; }