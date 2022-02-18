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
#include "LieYaml.h"
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

auto getTimeIndex(const vector<StampedPose>& stampedPoses, const double& t) {
    auto it = find_if(stampedPoses.begin(), stampedPoses.end(), [&t](const StampedPose& sp) { return sp.t > t; });
    return it;
}

VIOSimulator::VIOSimulator(
    const std::string& posesFileName, const YAML::Node& settings, const double timeScale, const int column,
    const char delim) {
    // Find the poses file
    assert(std::filesystem::exists(posesFileName));
    std::ifstream poseFile = std::ifstream(posesFileName);
    CSVReader poseFileIter(poseFile, delim);
    ++poseFileIter; // skip header

    double prevPoseTime = -1e8;
    for (CSVLine row : poseFileIter) {
        // Ignore leading columns
        for (int i = 0; i < column; ++i) {
            std::string x;
            row >> x;
        }

        StampedPose pose;
        row >> pose.t >> pose.pose;
        pose.t *= timeScale;
        if (pose.t > prevPoseTime + 1e-8) {
            // Avoid poses with the same timestamp.
            poses.emplace_back(pose);
            prevPoseTime = pose.t;
        }
    }

    int numPoints = configOrDefault(settings["numPoints"], 1000);
    double wallDistance = configOrDefault(settings["wallDistance"], 5.0);
    randomSeed = configOrDefault(settings["randomSeed"], rand());
    srand(randomSeed);

    inertialPoints = generateWorldPoints(numPoints, wallDistance);

    safeConfig(settings, "fieldOfView", fieldOfView);
    safeConfig(settings, "maxFeatures", maxFeatures);
    safeConfig(settings, "cameraOffset", cameraOffset);
}

vector<Landmark> VIOSimulator::generateWorldPoints(const int num, const double distance) const {
    // Find the limits of the trajectory
    Vector3d trajectoryMin = 1e8 * Vector3d::Ones();
    Vector3d trajectoryMax = -1e8 * Vector3d::Ones();
    for (const StampedPose& pose : poses) {
        if (pose.pose.x.x() < trajectoryMin.x())
            trajectoryMin.x() = pose.pose.x.x();
        else if (pose.pose.x.x() > trajectoryMax.x())
            trajectoryMax.x() = pose.pose.x.x();

        if (pose.pose.x.y() < trajectoryMin.y())
            trajectoryMin.y() = pose.pose.x.y();
        else if (pose.pose.x.y() > trajectoryMax.y())
            trajectoryMax.y() = pose.pose.x.y();

        if (pose.pose.x.z() < trajectoryMin.z())
            trajectoryMin.z() = pose.pose.x.z();
        else if (pose.pose.x.z() > trajectoryMax.z())
            trajectoryMax.z() = pose.pose.x.z();
    }

    // Place points at distance outside of the trajectory
    vector<Landmark> points(num);
    const Vector3d scaling = trajectoryMax - trajectoryMin + 2 * distance * Vector3d::Ones();
    const Vector3d offset = trajectoryMin - distance * Vector3d::Ones();

    for (int i = 0; i < num; ++i) {
        points[i].id = i;
        points[i].p = 0.5 * (Vector3d::Random() + Vector3d::Ones()); // Random point in [0,1]^3
        points[i].p = points[i].p.cwiseProduct(scaling);
        points[i].p = points[i].p + offset;

        // Now the point is somwhere in the trajectory box, we project one of the coordinates to the limits.
        switch ((6 * i) / num) {
        case 0:
            points[i].p.x() = offset.x();
            break;
        case 1:
            points[i].p.x() = offset.x() + scaling.x();
            break;
        case 2:
            points[i].p.y() = offset.y();
            break;
        case 3:
            points[i].p.y() = offset.y() + scaling.y();
            break;
        case 4:
            points[i].p.z() = offset.z() + scaling.z();
            break;
        case 5:
            points[i].p.z() = offset.z();
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

IMUVelocity VIOSimulator::getIMU(const double& currentTime) const {
    IMUVelocity imuVel;
    imuVel.stamp = currentTime;

    auto it = getTimeIndex(poses, imuVel.stamp);
    if (it == poses.end()) {
        imuVel.gyr.setZero();
        imuVel.acc.setZero();
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
    const StampedPose& pose0 = *(it - 2);
    const StampedPose& pose1 = *(it - 1);
    const StampedPose& pose2 = *(it); // This is the pose we are moving towards
    const StampedPose& pose3 = *(it + 1);

    imuVel.gyr = SO3d::log(pose1.pose.R.inverse() * pose2.pose.R) / (pose2.t - pose1.t);
    const SO3d imuAtt = pose1.pose.R * SO3d::exp((currentTime - pose1.t) * imuVel.gyr);

    // Compute the linear acceleration

    // Use least squares fitting for inertial acceleration
    const double& tau0 = pose0.t - currentTime;
    const double& tau1 = pose1.t - currentTime;
    const double& tau2 = pose2.t - currentTime;
    const double& tau3 = pose3.t - currentTime;
    const Vector3d& x0 = pose0.pose.x;
    const Vector3d& x1 = pose1.pose.x;
    const Vector3d& x2 = pose2.pose.x;
    const Vector3d& x3 = pose3.pose.x;

    if (tau0 > 0 || tau2 < 0) {
        // The requested currentTime is outside of the range of poses. We use zero.
        imuVel.acc = imuAtt.inverse() * (-Vector3d(0, 0, -GRAVITY_CONSTANT));
        return imuVel;
    }

    Matrix<double, 4, 3> b;
    b.block<1, 3>(0, 0) = x0.transpose();
    b.block<1, 3>(1, 0) = x1.transpose();
    b.block<1, 3>(2, 0) = x2.transpose();
    b.block<1, 3>(3, 0) = x3.transpose();
    Matrix<double, 4, 4> tMat;
    tMat.block<1, 4>(0, 0) << tau0 * tau0 * tau0, tau0 * tau0, tau0, 1;
    tMat.block<1, 4>(1, 0) << tau1 * tau1 * tau1, tau1 * tau1, tau1, 1;
    tMat.block<1, 4>(2, 0) << tau2 * tau2 * tau2, tau2 * tau2, tau2, 1;
    tMat.block<1, 4>(3, 0) << tau3 * tau3 * tau3, tau3 * tau3, tau3, 1;
    Matrix<double, 4, 3> aVecs = (tMat.transpose() * tMat).inverse() * tMat.transpose() * b;

    const Vector3d& inertialAccel = 2 * aVecs.block<1, 3>(1, 0).transpose();

    // Convert the inertial acceleration to the body-fixed measurement
    imuVel.acc = imuAtt.inverse() * (inertialAccel - Vector3d(0, 0, -GRAVITY_CONSTANT));

    return imuVel;
}

VisionMeasurement VIOSimulator::getVision(const double& currentTime) const {
    VisionMeasurement measData;
    measData.stamp = currentTime;

    auto it = getTimeIndex(poses, currentTime);
    if (it == poses.end()) {
        return measData;
    }

    // Compute the current pose through interpolation
    while (it - 1 < poses.begin()) {
        ++it;
    }
    const StampedPose& pose0 = *(it - 1);
    const StampedPose& pose1 = *(it);

    const se3d currentVel = SE3d::log(pose0.pose.inverse() * pose1.pose) / (pose1.t - pose0.t);
    const SE3d currentPose = pose0.pose * SE3d::exp(currentVel * (currentTime - pose0.t));
    const SE3d cameraPoseInv = (currentPose * cameraOffset).inverse();

    // Identify all visible features
    std::vector<Landmark> cameraFramePoints(inertialPoints.size());
    transform(
        inertialPoints.begin(), inertialPoints.end(), cameraFramePoints.begin(), [&cameraPoseInv](const Landmark& p) {
            return Landmark{cameraPoseInv * p.p, p.id};
        });
    const auto fovEnd = remove_if(cameraFramePoints.begin(), cameraFramePoints.end(), [this](const Landmark& p) {
        return atan2(p.p.segment<2>(0).norm(), p.p.z()) > this->fieldOfView * 3.14 / 180.;
    });
    cameraFramePoints.erase(fovEnd, cameraFramePoints.end());

    // Remove tracking points that are no longer visible
    erase_if(pointId2TrackingId, [&cameraFramePoints](const auto& keyValPair) {
        int key = keyValPair.first;
        return !any_of(
            cameraFramePoints.begin(), cameraFramePoints.end(), [&key](const Landmark& p) { return p.id == key; });
    });

    // Add new points to track
    for (const Landmark& p : cameraFramePoints) {
        if (pointId2TrackingId.size() >= maxFeatures) {
            break;
        }
        if (pointId2TrackingId.find(p.id) == pointId2TrackingId.end()) {
            // Point id is already in the tracking map. Add it.
            pointId2TrackingId[p.id] = ++currentTrackingId;
        }
    }

    // Keep only the points that are being tracked
    const auto trackingEnd = remove_if(cameraFramePoints.begin(), cameraFramePoints.end(), [this](const Landmark& p) {
        return (this->pointId2TrackingId.find(p.id) == this->pointId2TrackingId.end());
    });
    cameraFramePoints.erase(trackingEnd, cameraFramePoints.end());

    // Create the measurement bearings, sorted on their tracking ids.
    // transform(
    //     cameraFramePoints.begin(), cameraFramePoints.end(), std::inserter(measData.camCoordinates,
    //     measData.camCoordinates.begin()), [this](const Landmark& p) { return
    //     std::make_pair(this->pointId2TrackingId[p.id], p.p.normalized()); });

    /// @todo Add camera support to the simulator

    return measData;
}

const std::vector<StampedPose>& VIOSimulator::viewPoses() const { return poses; }