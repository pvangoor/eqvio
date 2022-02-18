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

#if EQVIO_BUILD_VISUALISATION
void VIOVisualiser::updateMapDisplay(const VIOState& state) {
    if (!plotter) {
        plotter = std::make_unique<Plotter>();
    }

    constexpr int minimumLife = 3;
    liepp::SE3d cameraPose = state.sensor.pose * state.sensor.cameraOffset;

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
            persistentPoints[lm.id] = cameraPose * lm.p;
        }
    }
    std::vector<Eigen::Vector3d> worldPoints(persistentPoints.size());
    std::transform(persistentPoints.begin(), persistentPoints.end(), worldPoints.begin(), [](const auto& lm) {
        const auto& [id, p] = lm;
        return p;
    });

    // Save the IMU position to the trail
    positionTrail.emplace_back(state.sensor.pose.x);

    // Draw the points
    plotter->drawPoints(currentPoints, Eigen::Vector4d(1, 1, 0, 1), 5);
    plotter->hold = true;
    plotter->drawPoints(worldPoints, Eigen::Vector4d(0, 0, 0, 0), 3);

    // Draw the IMU and camera axes
    plotter->drawLine(positionTrail, Eigen::Vector4d(1, 1, 1, 1), 2);
    plotter->drawLine({state.sensor.pose.x, cameraPose.x}, Eigen::Vector4d(1, 0, 1, 1), 2);
    plotter->drawAxes(cameraPose.asMatrix());
    plotter->drawAxes(state.sensor.pose.asMatrix(), 1.0, 5);
    plotter->hold = false;
    plotter->lockOrigin(state.sensor.pose.x);
}

VIOVisualiser::~VIOVisualiser() {
    if (plotter) {
        plotter->unlockOrigin();
    }
}

#else

void VIOVisualiser::updateMapDisplay(const VIOState& state) {}
VIOVisualiser::~VIOVisualiser() = default;

#endif