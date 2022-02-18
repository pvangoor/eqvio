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

#pragma once

#include <memory>
#include <ostream>
#include <string>

#include "LieYaml.h"
#include "yaml-cpp/yaml.h"

#include "VIOFilter.h"
#include "common.h"

/** @brief The local coordinate charts available to the EqF
 */
enum class CoordinateChoice { Euclidean, InvDepth, Normal };

/** @brief Choose coordinates for the EqF based on a YAML node
 *
 * @param ccNode The YAML node with the coordinate choice.
 */
static CoordinateChoice coordinateSelection(const YAML::Node& ccNode) {
    std::string choice;
    safeConfig(ccNode, "settings:coordinateChoice", choice);

    if (choice == "Euclidean")
        return CoordinateChoice::Euclidean;
    else if (choice == "InvDepth")
        return CoordinateChoice::InvDepth;
    else if (choice == "Normal")
        return CoordinateChoice::Normal;
    else
        throw std::runtime_error("Invalid coordinate choice. Valid choices are Euclidean, InvDepth, Normal.");
}

/** @brief The settings of the EqF for VIO.
 *
 * This includes settings for:
 * - Initial covariance
 * - Process covariance
 * - Input (IMU velocity) covariance
 * - Output (Features) covariance
 * - Implementation settings
 * - Initial camera offset estimate
 */
struct VIOFilter::Settings {
    double biasOmegaProcessVariance = 0.001; ///< The variance of the Wiener process of the gyroscope bias error
    double biasAccelProcessVariance = 0.001; ///< The variance of the Wiener process of the accelerometer bias error
    double attitudeProcessVariance = 0.001;  ///< The variance of the Wiener process of the attitude error
    double positionProcessVariance = 0.001;  ///< The variance of the Wiener process of the position error
    double velocityProcessVariance = 0.001;  ///< The variance of the Wiener process of the velocity error
    double cameraAttitudeProcessVariance = 0.001; ///< The variance of the Wiener process of the camera attitude error
    double cameraPositionProcessVariance = 0.001; ///< The variance of the Wiener process of the camera position error
    double pointProcessVariance = 0.001;          ///< The variance of the Wiener process of the landmark position error

    double velGyrNoise = 0.1;      ///< The noise of the gyroscope measurements
    double velAccNoise = 0.1;      ///< The noise of the accelerometer measurements
    double velGyrBiasWalk = 0.001; ///< The random walk of the gyroscope measurements
    double velAccBiasWalk = 0.001; ///< The random walk of the accelerometer measurements

    double measurementNoise = 0.1;     ///< The noise of the feature pixel coordinate measurements
    double outlierThresholdAbs = 1e8;  ///< The absolute outlier threshold. A lower value means more outliers.
    double outlierThresholdProb = 1e8; ///< The relative outlier threshold. A lower value means more outliers.
    double featureRetention = 0.3; ///< The minimum proportion of features that are always kept, regardless of outliers.

    double initialAttitudeVariance = 1.0;       ///< The initial variance of the attitude error
    double initialPositionVariance = 1.0;       ///< The initial variance of the position error
    double initialVelocityVariance = 1.0;       ///< The initial variance of the velocity error
    double initialCameraAttitudeVariance = 0.1; ///< The initial variance of the camera attitude error
    double initialCameraPositionVariance = 0.1; ///< The initial variance of the camera position error
    double initialPointVariance = 1.0;          ///< The initial variance of the body-fixed landmark position error
    double initialBiasOmegaVariance = 1.0;      ///< The initial variance of the gyroscope bias error
    double initialBiasAccelVariance = 1.0;      ///< The initial variance of the accelerometer bias error
    double initialSceneDepth = 1.0;             ///< The depth value used to initialise new features as landmarks

    bool useDiscreteInnovationLift = true; ///< If true, compute the EqF correction using a discrete action inverse.
    bool useDiscreteVelocityLift = true;   ///< If true, use a discretised version of the EqF lift function.
    bool fastRiccati = false;   ///< If true, then only propagate the Riccati matrix when receiving vision measurements.
    bool useMedianDepth = true; ///< If true, use the median depth of all current landmarks to initialise new landmarks.
    bool useFeaturePredictions = false; ///< If true, pass the estimated positions of features to the front-end.
    bool useEquivariantOutput = true;   ///< If true, use the equivariant output approximation.
    CoordinateChoice coordinateChoice = CoordinateChoice::Euclidean; ///< The local coordinate chart choice.
    liepp::SE3d cameraOffset = liepp::SE3d::Identity(); ///< The initial estimate of the camera w.r.t. the IMU.

    Settings() = default;

    /** @brief Configure the settings from a YAML Node
     *
     * @see The example configuration YAML file provided.
     */
    Settings(const YAML::Node& configNode);
};

inline VIOFilter::Settings::Settings(const YAML::Node& configNode) {
    // Configure gain matrices

    safeConfig(configNode, "processVariance:biasGyr", biasOmegaProcessVariance);
    safeConfig(configNode, "processVariance:biasAcc", biasAccelProcessVariance);
    safeConfig(configNode, "processVariance:attitude", attitudeProcessVariance);
    safeConfig(configNode, "processVariance:position", positionProcessVariance);
    safeConfig(configNode, "processVariance:velocity", velocityProcessVariance);
    safeConfig(configNode, "processVariance:point", pointProcessVariance);
    safeConfig(configNode, "processVariance:cameraAttitude", cameraAttitudeProcessVariance);
    safeConfig(configNode, "processVariance:cameraPosition", cameraPositionProcessVariance);

    safeConfig(configNode, "measurementNoise:feature", measurementNoise);
    safeConfig(configNode, "measurementNoise:featureOutlierAbs", outlierThresholdAbs);
    safeConfig(configNode, "measurementNoise:featureOutlierProb", outlierThresholdProb);
    safeConfig(configNode, "measurementNoise:featureRetention", featureRetention);

    safeConfig(configNode, "velocityNoise:gyr", velGyrNoise);
    safeConfig(configNode, "velocityNoise:acc", velAccNoise);
    safeConfig(configNode, "velocityNoise:gyrBias", velGyrBiasWalk);
    safeConfig(configNode, "velocityNoise:accBias", velAccBiasWalk);

    // Configure initial gains
    safeConfig(configNode, "initialVariance:attitude", initialAttitudeVariance);
    safeConfig(configNode, "initialVariance:position", initialPositionVariance);
    safeConfig(configNode, "initialVariance:velocity", initialVelocityVariance);
    safeConfig(configNode, "initialVariance:point", initialPointVariance);
    safeConfig(configNode, "initialVariance:biasGyr", initialBiasOmegaVariance);
    safeConfig(configNode, "initialVariance:biasAcc", initialBiasAccelVariance);
    safeConfig(configNode, "initialVariance:cameraAttitude", initialCameraAttitudeVariance);
    safeConfig(configNode, "initialVariance:cameraPosition", initialCameraPositionVariance);

    // Configure computation methods
    safeConfig(configNode, "settings:useDiscreteInnovationLift", useDiscreteInnovationLift);
    safeConfig(configNode, "settings:useDiscreteVelocityLift", useDiscreteVelocityLift);
    safeConfig(configNode, "settings:fastRiccati", fastRiccati);
    safeConfig(configNode, "settings:useMedianDepth", useMedianDepth);
    safeConfig(configNode, "settings:useFeaturePredictions", useFeaturePredictions);
    safeConfig(configNode, "settings:useEquivariantOutput", useEquivariantOutput);

    coordinateChoice = coordinateSelection(configNode);

    // Configure initial settings
    safeConfig(configNode, "initialValue:sceneDepth", initialSceneDepth);
    safeConfig(configNode, "initialValue:cameraOffset", cameraOffset);
}
