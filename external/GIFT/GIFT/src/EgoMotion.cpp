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
#include <iostream>
#include <utility>

using namespace std;
using namespace Eigen;
using namespace cv;
using namespace GIFT;

EgoMotion::EgoMotion(const vector<pair<Vector3T, Vector3T>>& sphereFlows) {
    Vector3T linVel(0, 0, 1);
    Vector3T angVel(0, 0, 0);

    pair<int, ftype> stepResPair = optimize(sphereFlows, linVel, angVel);

    this->optimisedResidual = stepResPair.second;
    this->optimisationSteps = stepResPair.first;
    this->linearVelocity = linVel;
    this->angularVelocity = angVel;
    this->numberOfFeatures = sphereFlows.size();
}

EgoMotion::EgoMotion(const vector<pair<Vector3T, Vector3T>>& sphereFlows, const Vector3T& initLinVel) {
    Vector3T linVel = initLinVel;
    Vector3T angVel = estimateAngularVelocity(sphereFlows, linVel);

    pair<int, ftype> stepResPair = optimize(sphereFlows, linVel, angVel);

    this->optimisedResidual = stepResPair.second;
    this->optimisationSteps = stepResPair.first;
    this->linearVelocity = linVel;
    this->angularVelocity = angVel;
    this->numberOfFeatures = sphereFlows.size();
}

EgoMotion::EgoMotion(
    const vector<pair<Vector3T, Vector3T>>& sphereFlows, const Vector3T& initLinVel, const Vector3T& initAngVel) {
    Vector3T linVel = initLinVel;
    Vector3T angVel = initAngVel;

    pair<int, ftype> stepResPair = optimize(sphereFlows, linVel, angVel);

    this->optimisedResidual = stepResPair.second;
    this->optimisationSteps = stepResPair.first;
    this->linearVelocity = linVel;
    this->angularVelocity = angVel;
    this->numberOfFeatures = sphereFlows.size();
}

EgoMotion::EgoMotion(const std::vector<Feature>& features, const ftype& dt) {
    vector<pair<Vector3T, Vector3T>> sphereFlows;
    for (const auto& lm : features) {
        if (lm.lifetime < 2)
            continue;
        sphereFlows.emplace_back(make_pair(lm.sphereCoordinates(), lm.opticalFlowSphere() / dt));
    }

    Vector3T linVel(0, 0, 1);
    Vector3T angVel(0, 0, 0);

    pair<int, ftype> stepResPair = optimize(sphereFlows, linVel, angVel);

    this->optimisedResidual = stepResPair.second;
    this->optimisationSteps = stepResPair.first;
    this->linearVelocity = linVel;
    this->angularVelocity = angVel;
    this->numberOfFeatures = sphereFlows.size();
}

EgoMotion::EgoMotion(const vector<GIFT::Feature>& features, const Vector3T& initLinVel, const ftype& dt) {
    vector<pair<Vector3T, Vector3T>> sphereFlows;
    for (const auto& lm : features) {
        if (lm.lifetime < 2)
            continue;
        sphereFlows.emplace_back(make_pair(lm.sphereCoordinates(), lm.opticalFlowSphere() / dt));
    }

    Vector3T linVel = initLinVel;
    Vector3T angVel = estimateAngularVelocity(sphereFlows, linVel);

    pair<int, ftype> stepResPair = optimize(sphereFlows, linVel, angVel);

    this->optimisedResidual = stepResPair.second;
    this->optimisationSteps = stepResPair.first;
    this->linearVelocity = linVel;
    this->angularVelocity = angVel;
    this->numberOfFeatures = sphereFlows.size();
}

EgoMotion::EgoMotion(
    const std::vector<Feature>& features, const Vector3T& initLinVel, const Vector3T& initAngVel, const ftype& dt) {
    vector<pair<Vector3T, Vector3T>> sphereFlows;
    for (const auto& lm : features) {
        if (lm.lifetime < 2)
            continue;
        sphereFlows.emplace_back(make_pair(lm.sphereCoordinates(), lm.opticalFlowSphere() / dt));
    }
    Vector3T linVel = initLinVel;
    Vector3T angVel = initAngVel;

    pair<int, ftype> stepResPair = optimize(sphereFlows, linVel, angVel);

    this->optimisedResidual = stepResPair.second;
    this->optimisationSteps = stepResPair.first;
    this->linearVelocity = linVel;
    this->angularVelocity = angVel;
    this->numberOfFeatures = sphereFlows.size();
}

pair<int, ftype> EgoMotion::optimize(
    const vector<pair<Vector3T, Vector3T>>& flows, Vector3T& linVel, Vector3T& angVel) {
    ftype lastResidual = 1e8;
    ftype residual = computeResidual(flows, linVel, angVel);

    Vector3T bestLinVel = linVel;
    Vector3T bestAngVel = angVel;
    ftype bestResidual = INFINITY;
    int optimisationSteps = 0;

    int iteration = 0;
    while ((abs(lastResidual - residual) > optimisationThreshold) && (iteration < maxIterations)) {
        lastResidual = residual;
        optimizationStep(flows, linVel, angVel);
        residual = computeResidual(flows, linVel, angVel);
        ++iteration;

        // cout << "residual: " << residual << endl;

        if (residual < bestResidual) {
            bestResidual = residual;
            bestLinVel = linVel;
            bestAngVel = angVel;
            optimisationSteps = iteration;
        }
    }

    linVel = bestLinVel;
    angVel = bestAngVel;

    if (voteForLinVelInversion(flows, linVel, angVel)) {
        linVel = -linVel;
    }

    return make_pair(optimisationSteps, bestResidual);
}

ftype EgoMotion::computeResidual(
    const vector<pair<Vector3T, Vector3T>>& flows, const Vector3T& linVel, const Vector3T& angVel) {
    Vector3T wHat = linVel.normalized();

    ftype residual = 0;
    int normalisationFactor = 0;
    for (const auto& flow : flows) {
        const Vector3T& phi = flow.second;
        const Vector3T& eta = flow.first;

        ftype res_i = wHat.dot((phi + angVel.cross(eta)).cross(eta));
        residual += pow(res_i, 2);
        ++normalisationFactor;
    }
    normalisationFactor = max(normalisationFactor, 1);
    residual = residual / normalisationFactor;

    return residual;
}

void EgoMotion::optimizationStep(
    const std::vector<pair<Vector3T, Vector3T>>& flows, Vector3T& linVel, Vector3T& angVel) {
    auto Proj3 = [](const Vector3T& vec) { return Matrix3T::Identity() - vec * vec.transpose() / vec.squaredNorm(); };

    Vector3T wHat = linVel.normalized();

    Matrix3T tempHess11 = Matrix3T::Zero();
    Matrix3T tempHess12 = Matrix3T::Zero();
    Matrix3T tempHess22 = Matrix3T::Zero();
    Vector3T tempGrad2 = Vector3T::Zero();

    for (const auto& flow : flows) {
        // Each flow is a pair of spherical bearing eta and perpendicular flow vector phi.
        const Vector3T& phi = flow.second;
        const Vector3T& eta = flow.first;

        Vector3T ZOmega = (phi + angVel.cross(eta)).cross(eta);
        Matrix3T ProjEta = Proj3(eta);

        tempHess11 += ZOmega * ZOmega.transpose();
        tempHess12 += wHat.transpose() * ZOmega * ProjEta + ZOmega * wHat.transpose() * ProjEta;
        tempHess22 += ProjEta * wHat * wHat.transpose() * ProjEta;

        tempGrad2 += wHat.transpose() * ZOmega * ProjEta * wHat;
    }

    Matrix<ftype, 6, 6> hessian;
    Matrix<ftype, 6, 1> gradient;

    Matrix3T ProjWHat = Proj3(wHat);
    hessian.block<3, 3>(0, 0) = ProjWHat * tempHess11 * ProjWHat;
    hessian.block<3, 3>(0, 3) = -ProjWHat * tempHess12;
    hessian.block<3, 3>(3, 0) = hessian.block<3, 3>(0, 3).transpose();
    hessian.block<3, 3>(3, 3) = tempHess22;

    gradient.block<3, 1>(0, 0) = ProjWHat * tempHess11 * wHat;
    gradient.block<3, 1>(3, 0) = -tempGrad2;

    // Step with Newton's method
    // Compute the solution to Hess^{-1} * grad
    Matrix<ftype, 6, 1> step = hessian.bdcSvd(ComputeFullU | ComputeFullV).solve(gradient);
    wHat += -step.block<3, 1>(0, 0);

    linVel = linVel.norm() * wHat.normalized();
    angVel += -step.block<3, 1>(3, 0);
}

vector<pair<Point2f, Vector2T>> EgoMotion::estimateFlowsNorm(const vector<GIFT::Feature>& features) const {
    vector<pair<Vector3T, Vector3T>> flowsSphere = estimateFlows(features);
    vector<pair<Point2f, Vector2T>> flowsNorm;

    for (const auto& flowSphere : flowsSphere) {
        const Vector3T& eta = flowSphere.first;
        const Vector3T& phi = flowSphere.second;

        ftype eta3 = eta.z();
        Point2f etaNorm(eta.x() / eta3, eta.y() / eta3);
        Vector2T phiNorm;
        phiNorm << 1 / eta3 * (phi.x() - etaNorm.x * phi.z()), 1 / eta3 * (phi.y() - etaNorm.y * phi.z());
        flowsNorm.emplace_back(make_pair(etaNorm, phiNorm));
    }

    return flowsNorm;
}

vector<pair<Vector3T, Vector3T>> EgoMotion::estimateFlows(const vector<GIFT::Feature>& features) const {
    auto Proj3 = [](const Vector3T& vec) { return Matrix3T::Identity() - vec * vec.transpose() / vec.squaredNorm(); };

    vector<pair<Vector3T, Vector3T>> estFlows;

    for (const auto& lm : features) {
        const Vector3T& eta = lm.sphereCoordinates();
        const Vector3T etaVel = Proj3(eta) * this->linearVelocity;

        ftype invDepth = 0;
        if (etaVel.norm() > 0)
            invDepth = -etaVel.dot(lm.opticalFlowSphere() + this->angularVelocity.cross(eta)) / etaVel.squaredNorm();

        Vector3T flow = -this->angularVelocity.cross(eta) + invDepth * etaVel;
        estFlows.emplace_back(make_pair(eta, flow));
    }

    return estFlows;
}

Vector3T EgoMotion::estimateAngularVelocity(
    const vector<pair<Vector3T, Vector3T>>& sphereFlows, const Vector3T& linVel) {
    // Uses ordinary least squares to estimate angular velocity from linear velocity and flows.
    auto Proj3 = [](const Vector3T& vec) { return Matrix3T::Identity() - vec * vec.transpose() / vec.squaredNorm(); };
    auto skew = [](const Vector3T& v) {
        Matrix3T m;
        m << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
        return m;
    };

    bool velocityFlag = (linVel.norm() > 1e-4);

    Matrix3T tempA = Matrix3T::Zero();
    Vector3T tempB = Vector3T::Zero();

    for (const auto& flow : sphereFlows) {
        const Vector3T& eta = flow.first;
        const Vector3T& phi = flow.second;

        tempA += Proj3(eta);
        if (velocityFlag)
            tempB += eta.cross(Proj3(Proj3(eta) * linVel) * phi);
        else
            tempB += eta.cross(phi);
    }

    Vector3T angVel = -tempA.inverse() * tempB;
    return angVel;
}

bool EgoMotion::voteForLinVelInversion(
    const vector<pair<Vector3T, Vector3T>>& flows, const Vector3T& linVel, const Vector3T& angVel) {
    auto Proj3 = [](const Vector3T& vec) { return Matrix3T::Identity() - vec * vec.transpose() / vec.squaredNorm(); };
    int invertVotes = 0;

    for (const auto& flowPair : flows) {
        const Vector3T& eta = flowPair.first;
        const Vector3T etaVel = Proj3(eta) * linVel;

        ftype scaledInvDepth = -etaVel.dot(flowPair.second + angVel.cross(eta));
        if (scaledInvDepth > 0)
            --invertVotes;
        else if (scaledInvDepth < 0)
            ++invertVotes;
    }

    bool invertDecision = (invertVotes > 0);
    return invertDecision;
}