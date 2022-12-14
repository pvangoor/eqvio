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

#include "eqvio/common/LieYaml.h"
#include "yaml-cpp/yaml.h"

#include "VIOFilter.h"
#include "eqvio/common/safeConfig.h"

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

    double velGyrNoise = 1e-4;    ///< The noise of the gyroscope measurements
    double velAccNoise = 1e-3;    ///< The noise of the accelerometer measurements
    double velGyrBiasWalk = 1e-5; ///< The random walk of the gyroscope measurements
    double velAccBiasWalk = 1e-3; ///< The random walk of the accelerometer measurements

    double measurementNoise = 2.0;     ///< The noise of the feature pixel coordinate measurements
    double outlierThresholdAbs = 1e8;  ///< The absolute outlier threshold. A lower value means more outliers.
    double outlierThresholdProb = 1e8; ///< The relative outlier threshold. A lower value means more outliers.
    double featureRetention = 0.3; ///< The minimum proportion of features that are always kept, regardless of outliers.

    double initialAttitudeVariance = 1.0e-4;       ///< The initial variance of the attitude error
    double initialPositionVariance = 1.0e-4;       ///< The initial variance of the position error
    double initialVelocityVariance = 1.0e-2;       ///< The initial variance of the velocity error
    double initialCameraAttitudeVariance = 1.0e-5; ///< The initial variance of the camera attitude error
    double initialCameraPositionVariance = 1.0e-4; ///< The initial variance of the camera position error
    double initialPointVariance = 1.0;             ///< The initial variance of the body-fixed landmark position error
    double initialPointDepthVariance = -1.0; ///< The initial variance of the body-fixed landmark depth error (optional)
    double initialBiasOmegaVariance = 0.1;   ///< The initial variance of the gyroscope bias error
    double initialBiasAccelVariance = 0.1;   ///< The initial variance of the accelerometer bias error
    double initialSceneDepth = 1.0;          ///< The depth value used to initialise new features as landmarks

    bool useDiscreteInnovationLift = true; ///< If true, compute the EqF correction using a discrete action inverse.
    bool useDiscreteVelocityLift = true;   ///< If true, use a discretised version of the EqF lift function.
    bool useDiscreteStateMatrix = false;   ///< If true, use a discretised version of the EqF State Matrix A0t.
    bool fastRiccati = false;   ///< If true, then only propagate the Riccati matrix when receiving vision measurements.
    bool useMedianDepth = true; ///< If true, use the median depth of all current landmarks to initialise new landmarks.
    bool useFeaturePredictions = false; ///< If true, pass the estimated positions of features to the front-end.
    bool useEquivariantOutput = true;   ///< If true, use the equivariant output approximation.
    bool removeLostLandmarks =
        true; ///< If true, remove landmarks from the state whenever they are lost by feature tracking.
    CoordinateChoice coordinateChoice = CoordinateChoice::Euclidean; ///< The local coordinate chart choice.
    liepp::SE3d cameraOffset = liepp::SE3d::Identity(); ///< The initial estimate of the camera w.r.t. the IMU.

    Settings() = default;

    /** @brief Configure the settings from a YAML Node
     *
     * @see The example configuration YAML file provided.
     */
    Settings(const YAML::Node& configNode);

    /** @brief Construct the initial state covariance from the provided settings.
     */
    Eigen::MatrixXd constructInitialStateCovariance(const size_t& numLandmarks = 0) const;

    /** @brief Construct the state process covariance from the provided settings.
     */
    Eigen::MatrixXd constructStateGainMatrix(const size_t& numLandmarks) const;

    /** @brief Construct the input covariance from the provided settings.
     */
    Eigen::Matrix<double, IMUVelocity::CompDim, IMUVelocity::CompDim> constructInputGainMatrix() const;

    /** @brief Construct the output covariance from the provided settings.
     */
    Eigen::MatrixXd constructOutputGainMatrix(const size_t& numLandmarks) const;
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
    safeConfig(configNode, "initialVariance:pointDepth", initialPointDepthVariance);
    safeConfig(configNode, "initialVariance:biasGyr", initialBiasOmegaVariance);
    safeConfig(configNode, "initialVariance:biasAcc", initialBiasAccelVariance);
    safeConfig(configNode, "initialVariance:cameraAttitude", initialCameraAttitudeVariance);
    safeConfig(configNode, "initialVariance:cameraPosition", initialCameraPositionVariance);

    // Configure computation methods
    safeConfig(configNode, "settings:useDiscreteInnovationLift", useDiscreteInnovationLift);
    safeConfig(configNode, "settings:useDiscreteVelocityLift", useDiscreteVelocityLift);
    safeConfig(configNode, "settings:useDiscreteStateMatrix", useDiscreteStateMatrix);
    safeConfig(configNode, "settings:fastRiccati", fastRiccati);
    safeConfig(configNode, "settings:useMedianDepth", useMedianDepth);
    safeConfig(configNode, "settings:useFeaturePredictions", useFeaturePredictions);
    safeConfig(configNode, "settings:useEquivariantOutput", useEquivariantOutput);
    safeConfig(configNode, "settings:removeLostLandmarks", removeLostLandmarks);

    coordinateChoice = coordinateSelection(configNode);

    // Configure initial settings
    safeConfig(configNode, "initialValue:sceneDepth", initialSceneDepth);
    safeConfig(configNode, "initialValue:cameraOffset", cameraOffset);
}

inline Eigen::MatrixXd VIOFilter::Settings::constructStateGainMatrix(const size_t& numLandmarks) const {
    Eigen::MatrixXd PMat = Eigen::MatrixXd::Identity(
        VIOSensorState::CompDim + 3 * numLandmarks, VIOSensorState::CompDim + 3 * numLandmarks);
    PMat.block<3, 3>(0, 0) *= this->biasOmegaProcessVariance;
    PMat.block<3, 3>(3, 3) *= this->biasAccelProcessVariance;
    PMat.block<3, 3>(6, 6) *= this->attitudeProcessVariance;
    PMat.block<3, 3>(9, 9) *= this->positionProcessVariance;
    PMat.block<3, 3>(12, 12) *= this->velocityProcessVariance;
    PMat.block<3, 3>(15, 15) *= this->cameraAttitudeProcessVariance;
    PMat.block<3, 3>(18, 18) *= this->cameraPositionProcessVariance;
    PMat.block(VIOSensorState::CompDim, VIOSensorState::CompDim, 3 * numLandmarks, 3 * numLandmarks) *=
        this->pointProcessVariance;

    return PMat;
}

inline Eigen::Matrix<double, IMUVelocity::CompDim, IMUVelocity::CompDim>
VIOFilter::Settings::constructInputGainMatrix() const {
    Eigen::Matrix<double, IMUVelocity::CompDim, IMUVelocity::CompDim> R =
        Eigen::Matrix<double, IMUVelocity::CompDim, IMUVelocity::CompDim>::Identity();
    R.block<3, 3>(0, 0) *= this->velGyrNoise * this->velGyrNoise;
    R.block<3, 3>(3, 3) *= this->velAccNoise * this->velAccNoise;
    R.block<3, 3>(6, 6) *= this->velGyrBiasWalk * this->velGyrBiasWalk;
    R.block<3, 3>(9, 9) *= this->velAccBiasWalk * this->velAccBiasWalk;
    return R;
}

inline Eigen::MatrixXd VIOFilter::Settings::constructOutputGainMatrix(const size_t& numLandmarks) const {
    return this->measurementNoise * this->measurementNoise *
           Eigen::MatrixXd::Identity(2 * numLandmarks, 2 * numLandmarks);
}

inline Eigen::MatrixXd VIOFilter::Settings::constructInitialStateCovariance(const size_t& numLandmarks) const {
    Eigen::MatrixXd Sigma(VIOSensorState::CompDim + 3 * numLandmarks, VIOSensorState::CompDim + 3 * numLandmarks);
    Sigma.setIdentity();
    Sigma.block<3, 3>(0, 0) *= this->initialBiasOmegaVariance;
    Sigma.block<3, 3>(3, 3) *= this->initialBiasAccelVariance;
    Sigma.block<3, 3>(6, 6) *= this->initialAttitudeVariance;
    Sigma.block<3, 3>(9, 9) *= this->initialPositionVariance;
    Sigma.block<3, 3>(12, 12) *= this->initialVelocityVariance;
    Sigma.block<3, 3>(15, 15) *= this->initialCameraAttitudeVariance;
    Sigma.block<3, 3>(18, 18) *= this->initialCameraPositionVariance;

    Sigma.block(VIOSensorState::CompDim, VIOSensorState::CompDim, 3 * numLandmarks, 3 * numLandmarks) *=
        this->initialPointVariance;

    if (initialPointDepthVariance > 0) {
        for (size_t i = 0; i < numLandmarks; ++i) {
            Sigma(VIOSensorState::CompDim + 3 * i + 2, VIOSensorState::CompDim + 3 * i + 2) = initialPointDepthVariance;
        }
    }

    return Sigma;
}