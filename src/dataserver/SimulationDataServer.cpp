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

#include "eqvio/dataserver/SimulationDataServer.h"
#include <random>

double initialTime;

std::vector<StampedPose>
generateLineTrajectory(const double& endTime, const double& frequency, const double initialTime) {
    liepp::SE3d pose = liepp::SE3d(liepp::SO3d::Identity(), {1.0, 0.0, 0.0});
    const int numPoses = std::floor(endTime * frequency);

    const double sinTime = 10.0;

    std::vector<StampedPose> trajectory(numPoses);
    for (int i = 0; i < numPoses; ++i) {
        const double t0 = i / frequency + initialTime;

        const double newCoord = 5 * (2 * (t0 + std::sin(t0 * 3.14 * 2 / sinTime)) / endTime - 1);
        // const double newCoord = 5 * std::sin(4 * 3.14 * t0 / endTime);
        pose.x = Eigen::Vector3d(0, newCoord, 0);
        // pose.x.x() = 0.1 * std::sin(2 * t0 / sinTime * 2 * 3.14);
        // pose.x.z() = 0.1 * std::sin(1.5 * t0 / sinTime * 2 * 3.14);
        // pose.x = Eigen::Vector3d(newCoord, 0, 0);
        trajectory.at(i) = StampedPose{t0 - initialTime, pose};
    }

    return trajectory;
}

std::vector<StampedPose>
generateWaveTrajectory(const double& endTime, const double& frequency, const double initialTime) {
    liepp::SE3d pose = liepp::SE3d(liepp::SO3d::Identity(), {1.0, 0.0, 0.0});
    const int numPoses = std::floor(endTime * frequency);

    constexpr double circleTime = 20.0;

    std::vector<StampedPose> trajectory(numPoses);
    for (int i = 0; i < numPoses; ++i) {
        const double t0 = i / frequency + initialTime;

        const double angle = 3.14 * 2 * t0 / circleTime;
        pose.R = liepp::SO3d::exp(Eigen::Vector3d(0, 0, angle));
        pose.x = Eigen::Vector3d(std::cos(angle), std::sin(angle), 0.2 * std::sin(10 * angle));

        trajectory.at(i) = StampedPose{t0 - initialTime, pose};
    }

    return trajectory;
}

std::vector<StampedPose>
generateSquareTrajectory(const double& endTime, const double& frequency, const double initialTime) {
    liepp::SO3d attitude = liepp::SO3d::Identity();
    Eigen::Vector3d position = Eigen::Vector3d(1.0, 0.0, 0.0);
    const int numPoses = std::floor(endTime * frequency);

    constexpr double squareTime = 20.0;

    std::vector<StampedPose> trajectory(numPoses);
    for (int i = 0; i < numPoses; ++i) {
        const double t0 = i / frequency + initialTime;

        // attitude = liepp::SO3d::exp(Eigen::Vector3d(0, 0, (std::sin(t0 * 2 / squareTime) + 1) * 3.14));
        attitude = liepp::SO3d::exp(Eigen::Vector3d(0, 0, (-t0 * 2 / squareTime) * 3.14));

        const double time01AlongSide = ((t0 / squareTime * 4) - int(t0 / squareTime * 4));
        assert(time01AlongSide >= 0 && time01AlongSide <= 1);
        const double distAlongSide = (-1 + 2 * std::pow(std::sin(time01AlongSide / 2 * 3.14), 2));

        const int squareSide = int(t0 / squareTime * 4) % 4;
        switch (squareSide) {
        case 0:
            position.x() = distAlongSide;
            position.y() = 1.0;
            break;
        case 1:
            position.x() = 1.0;
            position.y() = -distAlongSide;
            break;
        case 2:
            position.x() = -distAlongSide;
            position.y() = -1.0;
            break;
        case 3:
            position.x() = -1.0;
            position.y() = distAlongSide;
            break;
        }

        trajectory.at(i) = StampedPose{t0 - initialTime, liepp::SE3d(attitude, position)};
    }

    return trajectory;
}

std::vector<StampedPose>
generateSinTrajectory(const double& endTime, const double& frequency, const double initialTime) {
    liepp::SO3d attitude = liepp::SO3d::Identity();
    Eigen::Vector3d position = Eigen::Vector3d(0.0, 0.0, 0.0);
    const int numPoses = std::floor(endTime * frequency);

    constexpr double sinTime = 20.0;

    std::vector<StampedPose> trajectory(numPoses);
    for (int i = 0; i < numPoses; ++i) {
        const double t0 = i / frequency + initialTime;

        position.y() = 0.5 * std::cos(t0 / sinTime * 2 * 3.14);
        position.x() = 0.5 * std::cos(2 * t0 / sinTime * 2 * 3.14);
        position.z() = 0.5 * std::cos(1.5 * t0 / sinTime * 2 * 3.14);

        Eigen::Vector3d attVec = Eigen::Vector3d::Zero();
        attVec.x() = std::cos(5 * t0 / sinTime) * 3.14 / 4;
        attVec.y() = std::cos(-6 * t0 / sinTime) * 3.14 / 4;
        attVec.z() = std::cos(4 * t0 / sinTime) * 3.14 / 4;
        attitude = liepp::SO3d::exp(attVec);
        trajectory.at(i) = StampedPose{t0 - initialTime, liepp::SE3d(attitude, position)};
    }

    return trajectory;
}

std::vector<StampedPose> SimulationDataServer::generateTrajectory(const YAML::Node& choice) const {
    double desiredFreq = 10 * std::max(imuFreq, imageFreq);
    std::random_device rd;
    initialTime = double(rd()) * maxSimulationTime / rd.max();
    // initialTime = 7 + 1.0 / (5.0 * desiredFreq);
    // initialTime = 0.00333333333333333333333;
    initialTime = 0.5 / imuFreq;

    std::string choiceString = "none";
    safeConfig(choice, choiceString);
    if (choiceString == "wave") {
        return generateWaveTrajectory(maxSimulationTime, desiredFreq, initialTime);
    } else if (choiceString == "square") {
        return generateSquareTrajectory(maxSimulationTime, desiredFreq, initialTime);
    } else if (choiceString == "sine") {
        return generateSinTrajectory(maxSimulationTime, desiredFreq, initialTime);
    } else if (choiceString == "line") {
        return generateLineTrajectory(maxSimulationTime, desiredFreq, initialTime);
    } else {
        return generateWaveTrajectory(maxSimulationTime, desiredFreq, initialTime);
    }
}

GIFT::PinholeCamera generatePinholeCameraSquare() {
    // const cv::Size imageSize = cv::Size(500, 500);
    // const double fx = 250.0;
    // const double fy = 250.0;
    // const double cx = 250.0;
    // const double cy = 250.0;
    const cv::Size imageSize = cv::Size(752, 480);
    const double fx = 458.654;
    const double fy = 457.296;
    const double cx = 367.215;
    const double cy = 248.375;
    const cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

    return GIFT::PinholeCamera(imageSize, K);
}

double SimulationDataServer::nextImageTime() const { return imageMeasCount / imageFreq; }

double SimulationDataServer::nextIMUTime() const { return imuMeasCount / imuFreq; }

MeasurementType SimulationDataServer::nextMeasurementType() const {
    if (std::min(nextImageTime(), nextIMUTime()) >= maxSimulationTime) {
        return MeasurementType::None;
    } else if (nextImageTime() <= nextIMUTime()) {
        return MeasurementType::Image;
    } else {
        return MeasurementType::IMU;
    }
}

VIOState SimulationDataServer::getInitialCondition() const { return simulator.getFullState(0.0, true); }

VIOState SimulationDataServer::getTrueState(const double& stamp, const bool& withNoise) const {
    return simulator.getFullState(stamp, withNoise);
}

std::shared_ptr<liepp::SE3d> SimulationDataServer::cameraExtrinsics() const {
    return std::make_shared<liepp::SE3d>(simulator.cameraOffset);
}

StampedImage SimulationDataServer::getImage() { throw std::runtime_error("Cannot request images from the simulator."); }

IMUVelocity SimulationDataServer::getIMU() { return getSimIMU(); }

double SimulationDataServer::nextTime() const {
    const double nextMeasTime = std::min(nextImageTime(), nextIMUTime());
    return nextMeasTime < maxSimulationTime ? nextMeasTime : std::nan("");
}

VisionMeasurement SimulationDataServer::getSimVision() {
    const auto measurement = simulator.getVision(nextImageTime());
    ++imageMeasCount;
    return measurement;
}

IMUVelocity SimulationDataServer::getSimIMU() {
    const auto measurement = simulator.getIMU(nextIMUTime(), imuFreq);
    ++imuMeasCount;
    return measurement;
}

SimulationDataServer::SimulationDataServer(const YAML::Node& simSettings, const VIOFilter::Settings& filterSettings) {
    safeConfig(simSettings["duration"], maxSimulationTime);

    const std::vector<StampedPose> poses = generateTrajectory(simSettings["trajectory"]);

    GIFT::GICameraPtr camPtr = std::make_shared<GIFT::PinholeCamera>(generatePinholeCameraSquare());
    simulator = VIOSimulator(poses, camPtr, simSettings, filterSettings);

    safeConfig(simSettings["imuFreq"], imuFreq);
    safeConfig(simSettings["imageFreq"], imageFreq);

    Eigen::Matrix3d cameraRotation;
    cameraRotation << 0, 0, 1, -1, 0, 0, 0, -1, 0;
    simulator.cameraOffset.R = liepp::SO3d(cameraRotation);
}