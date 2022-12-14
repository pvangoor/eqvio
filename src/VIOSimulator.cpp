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

#include "eqvio/VIOSimulator.h"
#include "eqvio/common/LieYaml.h"
#include "eqvio/common/safeConfig.h"
#include "eqvio/csv/CSVReader.h"

#include <filesystem>
#include <fstream>
#include <random>

using namespace Eigen;
using namespace std;
using namespace liepp;

#if __cplusplus <= 201703L
#include <experimental/map>
using namespace std::experimental;
#endif

std::vector<StampedPose>::const_iterator getTimeIndex(const vector<StampedPose>& stampedPoses, const double& t) {
    std::vector<StampedPose>::const_iterator it = std::lower_bound(
        stampedPoses.begin(), stampedPoses.end(), t, [](const auto& elem, const auto& val) { return elem.t < val; });
    return it;
}

VIOSimulator::VIOSimulator(
    const std::vector<StampedPose>& poses, const GIFT::GICameraPtr& camPtr, const YAML::Node& settings,
    const VIOFilter::Settings& filterSettings)
    : poses(poses), filterSettings(filterSettings), cameraPtr(camPtr) {

    const int numPoints = configOrDefault(settings["numPoints"], 1000);
    const double wallDistance = configOrDefault(settings["wallDistance"], 2.0);
    std::random_device rd;
    randomSeed = configOrDefault(settings["randomSeed"], rd());
    srand(randomSeed);

    const int numWalls = configOrDefault(settings["numWalls"], 1);

    inertialPoints = generateWorldPoints(numPoints, wallDistance, numWalls);

    safeConfig(settings, "maxFeatures", maxFeatures);
    safeConfig(settings["initialNoise"], initialNoise);
    safeConfig(settings["inputNoise"], inputNoise);
    safeConfig(settings["outputNoise"], outputNoise);
}

vector<Landmark> VIOSimulator::generateWorldPoints(const int num, const double distance, const int numWalls) const {
    // Find the limits of the trajectory
    Vector3d trajectoryMin = 1e8 * Vector3d::Ones();
    Vector3d trajectoryMax = -1e8 * Vector3d::Ones();
    for (const StampedPose& pose : poses) {
        if (pose.pose.x.x() < trajectoryMin.x())
            trajectoryMin.x() = pose.pose.x.x();
        if (pose.pose.x.x() > trajectoryMax.x())
            trajectoryMax.x() = pose.pose.x.x();

        if (pose.pose.x.y() < trajectoryMin.y())
            trajectoryMin.y() = pose.pose.x.y();
        if (pose.pose.x.y() > trajectoryMax.y())
            trajectoryMax.y() = pose.pose.x.y();

        if (pose.pose.x.z() < trajectoryMin.z())
            trajectoryMin.z() = pose.pose.x.z();
        if (pose.pose.x.z() > trajectoryMax.z())
            trajectoryMax.z() = pose.pose.x.z();
    }

    // Place points at distance outside of the trajectory
    vector<Landmark> points(num);
    const Vector3d temp = 0.8 * Vector3d(numWalls > 0, numWalls > 1, numWalls > 3) + 0.2 * Vector3d::Ones();
    const Vector3d scaling = trajectoryMax - trajectoryMin + 2 * distance * temp;
    const Vector3d offset = trajectoryMin - distance * temp;

    for (int i = 0; i < num; ++i) {
        points[i].id = i;
        points[i].p = 0.5 * (Vector3d::Random() + Vector3d::Ones()); // Random point in [0,1]^3
        points[i].p = points[i].p.cwiseProduct(scaling);
        points[i].p = points[i].p + offset;

        // Now the point is somwhere in the trajectory box, we project one of the coordinates to the limits.
        switch ((numWalls * i) / num) {
        case 0:
            points[i].p.x() = offset.x() + scaling.x();
            break;
        case 1:
            points[i].p.y() = offset.y() + scaling.y();
            break;
        case 2:
            points[i].p.y() = offset.y();
            break;
        case 3:
            points[i].p.x() = offset.x();
            break;
        case 4:
            points[i].p.z() = offset.z();
            break;
        case 5:
            points[i].p.z() = offset.z() + scaling.z();
            break;
        default:
            points[i].p.z() = offset.z();
            break;
        }
    }

    mt19937 g(randomSeed);
    shuffle(points.begin(), points.end(), g);

    return points;
}

IMUVelocity VIOSimulator::getIMU(const double& currentTime, const double& samplingFrequency) const {
    IMUVelocity imuVel;
    imuVel.stamp = currentTime;

    auto it = getTimeIndex(poses, imuVel.stamp);
    if (it == poses.end()) {
        imuVel.gyr.setZero();
        const SO3d& imuAtt = poses.rbegin()->pose.R;
        imuVel.acc = imuAtt.inverse() * Vector3d(0, 0, GRAVITY_CONSTANT);
        return imuVel;
    }

    // Deal with the boundary
    while (it + 1 >= poses.end()) {
        --it;
    }
    while (it - 2 <= poses.begin()) {
        ++it;
    }

    // Compute the angular velocity
    const StampedPose& pose1 = *(it - 1);
    const StampedPose& pose2 = *(it); // This is the pose we are moving towards

    imuVel.gyr = SO3d::log(pose1.pose.R.inverse() * pose2.pose.R) / (pose2.t - pose1.t);
    const SO3d imuAtt = pose1.pose.R * SO3d::exp((currentTime - pose1.t) * imuVel.gyr);

    // Compute the linear acceleration
    const Eigen::Matrix3d inertialStates = getInertialStates(it, currentTime);
    const Eigen::Vector3d inertialAccel = inertialStates.col(2);

    // Convert the inertial acceleration to the body-fixed measurement
    imuVel.acc = imuAtt.inverse() * (inertialAccel - Vector3d(0, 0, -GRAVITY_CONSTANT));

    // Corrupt with noise
    if (inputNoise) {
        Eigen::Matrix<double, 12, 1> inputNoiseSample =
            sampleGaussianDistribution(filterSettings.constructInputGainMatrix() * std::max(samplingFrequency, 0.0));
        imuVel = imuVel + inputNoiseSample;
    }

    return imuVel;
}

Eigen::Matrix3d VIOSimulator::getInertialStates(std::vector<StampedPose>::const_iterator it, const double& ct) const {
    const StampedPose& pose0 = *(it - 2);
    const StampedPose& pose1 = *(it - 1);
    const StampedPose& pose2 = *(it); // This is the pose we are moving towards
    const StampedPose& pose3 = *(it + 1);

    // Use least squares fitting for inertial acceleration
    const double& tau0 = pose0.t - ct;
    const double& tau1 = pose1.t - ct;
    const double& tau2 = pose2.t - ct;
    const double& tau3 = pose3.t - ct;

    Eigen::Matrix3d result;
    // if (tau0 > 0 || tau2 < 0) {
    //     // The requested currentTime is outside of the range of poses. We use zero.
    //     result.setZero();
    //     return result;
    // }

    // The idea is that we can write the inertial position as a 3rd order polynomial, uniquely determined by its value
    // at the 4 chosen poses at the given times.
    // In one dimension, x(t) = a0 + a1*t + a2*t^2/2 + a3*t^3/6
    // We centre the polynomial at the current time, so t=0 means x gives the current position.
    // This whole thing can be rewritten as a matrix equation, where the vector X = A T, with
    // X = (x_1,x_2,x_3)^T, A_ij= a{i-1}_j, T=(1,t,t^2,t^3)^T
    // Using a few values of X and T, we can create matrices XX and TT by stacking columns horizontally.
    // Now, A is determined by XX = A TT, so A = XX TT^T (TT*TT^T)^{-1}

    Matrix<double, 3, 4> positionMat;
    positionMat.col(0) = pose0.pose.x;
    positionMat.col(1) = pose1.pose.x;
    positionMat.col(2) = pose2.pose.x;
    positionMat.col(3) = pose3.pose.x;
    Matrix<double, 4, 4> timeMat;
    timeMat.col(0) << 1.0, tau0, tau0 * tau0 / 2.0, tau0 * tau0 * tau0 / 6.0;
    timeMat.col(1) << 1.0, tau1, tau1 * tau1 / 2.0, tau1 * tau1 * tau1 / 6.0;
    timeMat.col(2) << 1.0, tau2, tau2 * tau2 / 2.0, tau2 * tau2 * tau2 / 6.0;
    timeMat.col(3) << 1.0, tau3, tau3 * tau3 / 2.0, tau3 * tau3 * tau3 / 6.0;
    Matrix<double, 3, 4> AMat = positionMat * timeMat.transpose() * (timeMat * timeMat.transpose()).inverse();

    // The left 3x3 block gives the position, velocity and acceleration in the inertial frame.
    return AMat.block<3, 3>(0, 0);
}

VisionMeasurement VIOSimulator::getVision(const double& currentTime) const {
    VisionMeasurement measData;
    measData.stamp = currentTime;
    measData.cameraPtr = cameraPtr;

    // Compute the current pose through interpolation
    auto it = getTimeIndex(poses, currentTime);
    if (it == poses.end()) {
        return measData;
    }
    while (it - 1 < poses.begin()) {
        ++it;
    }
    const StampedPose& pose0 = *(it - 1);
    const StampedPose& pose1 = *(it);
    const se3d currentVel = SE3d::log(pose0.pose.inverse() * pose1.pose) / (pose1.t - pose0.t);
    const SE3d currentPose = pose0.pose * SE3d::exp(currentVel * (currentTime - pose0.t));

    // Identify all visible features
    const SE3d cameraPoseInv = (currentPose * cameraOffset).inverse();
    std::vector<Landmark> cameraFramePoints(inertialPoints.size());
    transform(
        inertialPoints.begin(), inertialPoints.end(), cameraFramePoints.begin(), [&cameraPoseInv](const Landmark& p) {
            return Landmark{cameraPoseInv * p.p, p.id};
        });
    const auto fovEnd = remove_if(cameraFramePoints.begin(), cameraFramePoints.end(), [this](const Landmark& p) {
        return !cameraPtr->isInDomain(p.p);
    });
    cameraFramePoints.erase(fovEnd, cameraFramePoints.end());

    // Use the points with the lowest id numbers. Since the list is already ordered, this is just the first n elements.
    if (cameraFramePoints.size() > maxFeatures) {
        cameraFramePoints.erase(cameraFramePoints.begin() + maxFeatures, cameraFramePoints.end());
    }

    // Project the points and create the vision measurement vector.
    transform(
        cameraFramePoints.begin(), cameraFramePoints.end(),
        std::inserter(measData.camCoordinates, measData.camCoordinates.begin()),
        [this](const Landmark& p) { return std::make_pair(p.id, cameraPtr->projectPoint(p.p)); });

    // Corrupt with noise
    if (outputNoise) {
        Eigen::VectorXd outputNoiseSample =
            sampleGaussianDistribution(filterSettings.constructOutputGainMatrix(measData.camCoordinates.size()));
        measData = measData + outputNoiseSample;
    }

    return measData;
}

const std::vector<StampedPose>& VIOSimulator::viewPoses() const { return poses; }

VIOState VIOSimulator::getFullState(const double& time, const bool& allowNoise) const {
    auto it = getTimeIndex(poses, time);
    // Deal with the boundary
    while (it + 1 >= poses.end()) {
        --it;
    }
    while (it - 2 <= poses.begin()) {
        ++it;
    }

    const StampedPose& pose0 = *(it - 1);
    const StampedPose& pose1 = *(it);
    const Eigen::Vector3d angularVel = SO3d::log(pose0.pose.R.inverse() * pose1.pose.R) / (pose1.t - pose0.t);

    VIOState xi;
    xi.sensor.inputBias.setZero();
    xi.sensor.pose.R = pose0.pose.R * SO3d::exp(angularVel * (time - pose0.t));

    const Matrix3d inertialStates = getInertialStates(it, time);
    xi.sensor.pose.x = inertialStates.col(0);
    xi.sensor.velocity = xi.sensor.pose.R.inverse() * inertialStates.col(1);

    xi.sensor.cameraOffset = cameraOffset;

    const SE3d cameraPoseInv = (xi.sensor.pose * cameraOffset).inverse();
    xi.cameraLandmarks.resize(inertialPoints.size());
    transform(
        inertialPoints.begin(), inertialPoints.end(), xi.cameraLandmarks.begin(), [&cameraPoseInv](const Landmark& p) {
            return Landmark{cameraPoseInv * p.p, p.id};
        });

    // Corrupt with noise
    if (allowNoise && initialNoise) {
        Eigen::VectorXd epsilon =
            sampleGaussianDistribution(filterSettings.constructInitialStateCovariance(xi.cameraLandmarks.size()));
        // const VIOAlgebra Delta = getCoordinates(filterSettings.coordinateChoice)->liftInnovation(epsilon, xi);
        // xi = stateGroupAction(VIOExp(Delta), xi);
        xi = getCoordinates(filterSettings.coordinateChoice)->stateChart.chartInv(epsilon, xi);
    }

    return xi;
}