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

#include "eqvio/VIOVisualiser.h"

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <numeric>
#include <algorithm>

liepp::SE3d alignUmeyama(const std::vector<Eigen::Vector3d>& points1, const std::vector<Eigen::Vector3d>& points2) {
    // Find the SE(3) alignment matrix between points 1 and points 2.
    // The points should be aligned in terms of the index: point 1_i corresponds to point 2_i.
    // We find $(R,x) \in \mathbf{SE}(3)$ to minimise $\sum_{i=1}^n \vert (R p_i + x) - q_i \vert^2$ where $p_i$ are the points1 and $q_i$ are the points2.

    assert(points2.size() == points1.size());
    const size_t& n = points2.size();

    Eigen::Vector3d mu1 = Eigen::Vector3d::Zero();
    Eigen::Vector3d mu2 = Eigen::Vector3d::Zero();
    for (size_t i=0;i<n;++i){
        mu1 += 1./n * points1.at(i);
        mu2 += 1./n * points2.at(i);
    }

    double sigma1Squared = 0.0;
    Eigen::Matrix3d sigma12 = Eigen::Matrix3d::Zero();
    for (size_t i=0;i<n;++i){
        sigma1Squared += 1./n * (points1.at(i) - mu1).squaredNorm();
        sigma12 += 1./n * (points2.at(i) - mu2) * (points1.at(i) - mu1).transpose();
    }

    // The rotation is computed from the SVD of sigma12
#if EIGEN_MAJOR_VERSION >= 4 && EIGEN_MINOR_VERSION > 90
    Eigen::BDCSVD<Eigen::Matrix3d, Eigen::ComputeFullU | Eigen::ComputeFullV> svd(sigma12);
#else
    Eigen::BDCSVD<Eigen::Matrix3d> svd(sigma12, Eigen::ComputeFullU | Eigen::ComputeFullV);
#endif

    // S is used to ensure the rotation matrix is oriented (right handed)
    Eigen::Matrix3d S = Eigen::Matrix3d::Identity();
    double scaleSum = svd.singularValues().sum();
    if (sigma12.determinant() < 0) {
        S(2,2) = -1.;
        scaleSum += -2.*svd.singularValues()(2);
    }

    const Eigen::Matrix3d RMat = svd.matrixU() * S * svd.matrixV().transpose();
    const liepp::SO3d R = liepp::SO3d(RMat);
    const double s = 1./sigma1Squared * scaleSum;
    const Eigen::Vector3d x = mu2 - s * (R * mu1);

    return liepp::SE3d(R, x);
}

liepp::SE3d alignTrajectories(const std::vector<StampedPose>& estTrajectory, const std::vector<StampedPose>& refTrajectory) {
    // Find an alignment between an estimated and a reference trajectory.
    // To disambiguate, we try to align the estimated trajectory to the reference.

    // Aalign the times. The trajectory vectors should already be ordered.
    const double minTime = std::max(estTrajectory.front().t, refTrajectory.front().t);
    const double maxTime = std::min(estTrajectory.back().t, refTrajectory.back().t);
    // Find the trajectory with the slowest rate to use for matching
    const double refPeriod = (refTrajectory.back().t - refTrajectory.front().t) / refTrajectory.size();
    const double estPeriod = (estTrajectory.back().t - estTrajectory.front().t) / estTrajectory.size();
    const double usePeriod = std::max(refPeriod, estPeriod);

    // Match the trajectories based on the slowest rate.
    std::vector<Eigen::Vector3d> estTrajectoryMatched;
    std::vector<Eigen::Vector3d> refTrajectoryMatched;
    auto estTrajectoryIt = estTrajectory.begin();
    auto refTrajectoryIt = refTrajectory.begin();
    for (double time = minTime; time < maxTime; time += usePeriod) {
        // Iterate until the suggested poses match the time.
        while (estTrajectoryIt != estTrajectory.end() && estTrajectoryIt->t < time) {
            ++estTrajectoryIt;
        }
        while (refTrajectoryIt != refTrajectory.end() && refTrajectoryIt->t < time) {
            ++refTrajectoryIt;
        }
        if (refTrajectoryIt == refTrajectory.end() || estTrajectoryIt == estTrajectory.end())
            break;

        // Add the ref and estimated pose to the lists
        estTrajectoryMatched.emplace_back(estTrajectoryIt->pose.x);
        refTrajectoryMatched.emplace_back(refTrajectoryIt->pose.x);
    }
    if (estTrajectoryMatched.size() == 0) {
        return liepp::SE3d::Identity();
    }
    if (estTrajectoryMatched.size() <= 100) {
        return refTrajectory.begin()->pose * estTrajectory.begin()->pose.inverse();
    }

    return alignUmeyama(estTrajectoryMatched, refTrajectoryMatched);
}

void VIOVisualiser::displayFeatureImage(const std::vector<GIFT::Feature>& features, const cv::Mat& baseImage) {
    cv::Mat featureImage = baseImage;
    if (featureImage.channels() == 1) {
        cv::cvtColor(baseImage, featureImage, cv::COLOR_GRAY2BGR);
    }

    auto drawingLambda = [&featureImage](const GIFT::Feature& feature) {
        const int radius = 8;
        const int thickness = 2;
        cv::drawMarker(
            featureImage, feature.camCoordinates, cv::Scalar(0, 255, 255), cv::MARKER_TILTED_CROSS, radius, thickness);
        cv::circle(featureImage, feature.camCoordinates, radius, cv::Scalar(0, 255, 255), thickness / 2);
        cv::putText(
            featureImage, std::to_string(feature.idNumber), feature.camCoordinates + cv::Point2f(radius, radius),
            cv::FONT_HERSHEY_SIMPLEX, 0.33, cv::Scalar(0, 255, 0));
    };
    for_each(features.begin(), features.end(), drawingLambda);

    cv::imshow("Features", featureImage);
    cv::waitKey(1);
}

void VIOVisualiser::setGroundTruthTrajectory(const std::vector<StampedPose>& groundTruthTrajectory) {
    this->groundTruthTrajectory = groundTruthTrajectory;
}

#if EQVIO_BUILD_VISUALISATION
void VIOVisualiser::updateMapDisplay(const VIOState& state, const double& time) {
    if (!plotter) {
        plotter = std::make_unique<Plotter>();
    }
    if (time <= 0.)
        return;

    // Save the IMU pose to the trajectory
    estimatedTrajectory.emplace_back(StampedPose{time, state.sensor.pose});

    // Find the current groundtruth pose
    auto GTPoseIt = std::find_if(groundTruthTrajectory.begin(), groundTruthTrajectory.end(), 
        [time](const auto& tPose){return tPose.t >= time;});
    if (GTPoseIt != groundTruthTrajectory.begin()) {
        --GTPoseIt;
    }

    // Align the final pose.
    if (!groundTruthTrajectory.empty()) {
        alignmentMatrix = alignTrajectories(estimatedTrajectory, groundTruthTrajectory);
    }
    const liepp::SE3d aligned_imu_pose = alignmentMatrix * state.sensor.pose;

    // Construct the estimated and groundtruth position trails.
    std::vector<Eigen::Vector3d> groundTruthPositionTrail;
    for (const StampedPose & tPose : groundTruthTrajectory) {
        groundTruthPositionTrail.emplace_back(tPose.pose.x);
        if (tPose.t > time) {
            break;
        }
    }
    std::vector<Eigen::Vector3d> positionTrail(estimatedTrajectory.size());
    std::transform(estimatedTrajectory.begin(), estimatedTrajectory.end(), positionTrail.begin(),
    [this](const auto & tPose) {
        const auto& [t, pose] = tPose;
        return (alignmentMatrix * pose).x;
    });

    constexpr int minimumLife = 3;
    liepp::SE3d cameraPose = aligned_imu_pose * state.sensor.cameraOffset;

    // Get the inertial points
    std::vector<Eigen::Vector3d> currentPoints(state.cameraLandmarks.size());
    std::transform(
        state.cameraLandmarks.begin(), state.cameraLandmarks.end(), currentPoints.begin(),
        [&cameraPose](const Landmark& pt) { return cameraPose * pt.p; });

    // Update the lifetime of the current points
    for (const Landmark& lm : state.cameraLandmarks) {
        auto it = pointLifetimeCounter.find(lm.id);
        if (it == pointLifetimeCounter.end()) {
            pointLifetimeCounter[lm.id] = 1;
        } else {
            ++it->second;
        }
    }

    // Save and display persistent points with sufficient lifetime
    for (const Landmark& lm : state.cameraLandmarks) {
        if (pointLifetimeCounter.at(lm.id) > minimumLife) {
            persistentPoints[lm.id] = state.sensor.pose * state.sensor.cameraOffset * lm.p;
        }
    }
    std::vector<Eigen::Vector3d> worldPoints(persistentPoints.size());
    std::transform(persistentPoints.begin(), persistentPoints.end(), worldPoints.begin(), [this](const auto& lm) {
        const auto& [id, p] = lm;
        return alignmentMatrix * p;
    });

    // Draw the points
    plotter->drawPoints(currentPoints, Eigen::Vector4d(1, 1, 0, 1), 5);
    plotter->hold = true;
    plotter->drawPoints(worldPoints, Eigen::Vector4d(0, 0, 0, 0), 3);

    // Draw the IMU and camera axes
    plotter->drawLine(positionTrail, Eigen::Vector4d(1, 1, 1, 1), 2);
    plotter->drawLine({aligned_imu_pose.x, cameraPose.x}, Eigen::Vector4d(1, 0, 1, 1), 2);
    plotter->drawAxes(cameraPose.asMatrix());
    plotter->drawAxes(aligned_imu_pose.asMatrix(), 1.0, 5);

    // Add the groundtruth trajectory and pose
    if (groundTruthPositionTrail.size() >= 2) {
        plotter->drawLine(groundTruthPositionTrail, Eigen::Vector4d(0, 0, 0, 1), 2);
        plotter->drawAxes(GTPoseIt->pose.asMatrix());
    }

    // Turn off the hold and lock the view to be centred on the estimated pose.
    plotter->hold = false;
    plotter->lockOrigin(aligned_imu_pose.x);
}

VIOVisualiser::~VIOVisualiser() {
    if (plotter) {
        plotter->unlockOrigin();
    }
}

#else

void VIOVisualiser::updateMapDisplay(const VIOState& state, const double& time) {}
VIOVisualiser::~VIOVisualiser() = default;

#endif