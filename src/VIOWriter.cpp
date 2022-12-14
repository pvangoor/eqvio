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

#include "eqvio/VIOWriter.h"
#include "eqvio/csv/CSVLine.h"
#include <filesystem>

VIOWriter::VIOWriter(const std::string& providedOutputDir) : outputDir(providedOutputDir) {

    if (outputDir.back() != '/') {
        outputDir = outputDir + "/";
    }

    if (!std::filesystem::is_directory(outputDir)) {
        std::filesystem::create_directories(outputDir);
    }
}

void VIOWriter::writeStates(const double& stamp, const VIOState& xi) {
    { // Write the IMU pose to file
        if (!IMUStateFile.is_open()) {
            IMUStateFile.open(outputDir + "IMUState.csv");
            IMUStateFile << "time, px, py, pz, qw, qx, qy, qz, vx, vy, vz\n";
        }

        IMUStateFile << std::setprecision(20) << stamp << ", " << std::setprecision(6);
        CSVLine line;
        line << xi.sensor.pose << xi.sensor.velocity;
        IMUStateFile << line << '\n';
    }

    { // Write the Camera offset to file
        if (!cameraFile.is_open()) {
            cameraFile.open(outputDir + "camera.csv");
            cameraFile << "time, px, py, pz, qw, qx, qy, qz\n";
        }
        cameraFile << std::setprecision(20) << stamp << ", " << std::setprecision(6);
        CSVLine line;
        line << xi.sensor.cameraOffset;
        cameraFile << line << '\n';
    }

    { // Write the biases to file
        if (!biasFile.is_open()) {
            biasFile.open(outputDir + "bias.csv");
            biasFile << "time, bias_gyr_x, bias_gyr_y, bias_gyr_z, bias_acc_x, bias_acc_y, bias_acc_z\n";
        }
        biasFile << std::setprecision(20) << stamp << ", " << std::setprecision(6);
        CSVLine line;
        line << xi.sensor.inputBias;
        biasFile << line << '\n';
    }

    { // Write the points in world frame to file
        if (!pointsFile.is_open()) {
            pointsFile.open(outputDir + "points.csv");
            pointsFile << "time, p1id, p1x, p1y, p1z, ...\n";
        }
        pointsFile << std::setprecision(20) << stamp << ", " << std::setprecision(6);
        CSVLine line;
        const liepp::SE3d PC = xi.sensor.pose * xi.sensor.cameraOffset;
        for (const auto& q : xi.cameraLandmarks) {
            line << q.id << PC * q.p;
        }
        pointsFile << line << '\n';
    }
}

void VIOWriter::writeFeatures(const VisionMeasurement& y) {
    // Write the features to file
    if (!featuresFile.is_open()) {
        featuresFile.open(outputDir + "features.csv");
        featuresFile << "time, z1id, z1x, z1y, ...\n";
    }
    featuresFile << std::setprecision(20) << y.stamp << ", " << std::setprecision(6);
    CSVLine line;
    for (const auto& [id, z] : y.camCoordinates) {
        line << id << z;
    }
    featuresFile << line << '\n';
}

void VIOWriter::writeTiming(const LoopTimer::LoopTimingData& timingData) {
    if (!timingFile.is_open()) {
        timingFile.open(outputDir + "timing.csv");

        CSVLine header;
        header << "time";
        for (const auto& [lab, timing] : timingData.timings) {
            header << lab;
        }
        timingFile << header << '\n';
    }

    timingFile << std::setprecision(20) << timingData.loopTimeStart.count() << ", " << std::setprecision(6);
    CSVLine line;
    for (const auto& [lab, timing] : timingData.timings) {
        line << timing.count();
    }
    timingFile << line << '\n';
}

void VIOWriter::writeLandmarkError(const double& stamp, const VIOState& trueState, const VIOState& estState) {
    if (!landmarkErrorFile.is_open()) {
        landmarkErrorFile.open(outputDir + "landmarkError.csv");
        landmarkErrorFile << "time, lm_err_1, lm_err_2, ...\n";
    }

    landmarkErrorFile << std::setprecision(20) << stamp << ", " << std::setprecision(6);
    CSVLine line;

    for (const Landmark& lm : trueState.cameraLandmarks) {
        auto it = std::find_if(
            estState.cameraLandmarks.begin(), estState.cameraLandmarks.end(),
            [&lm](const Landmark& lmEst) { return lmEst.id == lm.id; });
        if (it == estState.cameraLandmarks.end()) {
            line << std::nan("");
        } else {
            line << (it->p - lm.p).norm();
        }
    }

    landmarkErrorFile << line << '\n';
}

void VIOWriter::writeConsistency(const double& stamp, const VIOState& trueState, const VIO_eqf& filter) {

    {
        if (!trueStateFile.is_open()) {
            trueStateFile.open(outputDir + "trueState.csv");
            trueStateFile << "time, pose_tx, pose_ty, pose_tz, pose_qw, pose_qx, pose_qy, pose_qz,"
                             "pose_vx, pose_vy, pose_vz, cam_tx, cam_ty, cam_tz, cam_qw, cam_qx, cam_qy, cam_qz,"
                             "bias_gyr_x, bias_gyr_y, bias_gyr_z, bias_acc_x, bias_acc_y, bias_acc_z"
                             "num_lm, lm_1_id, lm_1_x, lm_1_y, lm_1_z, lm_2_id, lm_2_x, lm_2_y, lm_2_z, ...\n";
        }
        trueStateFile << std::setprecision(20) << stamp << ", " << std::setprecision(6);
        CSVLine line;
        line << trueState;
        trueStateFile << line << '\n';
    }

    {
        if (!neesFile.is_open()) {
            neesFile.open(outputDir + "nees.csv");
            neesFile << "time, NEES, DoF, PoseNEES, AttitudeNEES\n";
        }
        neesFile << std::setprecision(20) << stamp << ", " << std::setprecision(6);
        VIOState truePoseState = filter.stateEstimate();
        truePoseState.sensor.pose = trueState.sensor.pose;
        VIOState trueAttitudeState = filter.stateEstimate();
        trueAttitudeState.sensor.pose.R = trueState.sensor.pose.R;

        const double fullNEES = filter.computeNEES(trueState);

        const liepp::SE3d errorPose = trueState.sensor.pose * filter.X.A.inverse();
        const liepp::se3d poseEpsilon = liepp::SE3d::log(filter.xi0.sensor.pose.inverse() * errorPose);
        const double poseNEES = poseEpsilon.transpose() * filter.Sigma.block<6, 6>(6, 6).inverse() * poseEpsilon;

        const Eigen::Vector3d attEpsilon = liepp::SO3d::log(filter.xi0.sensor.pose.R.inverse() * errorPose.R);
        const double attitudeNEES = attEpsilon.transpose() * filter.Sigma.block<3, 3>(6, 6).inverse() * attEpsilon;

        CSVLine line;
        line << fullNEES << filter.xi0.Dim() << poseNEES << attitudeNEES;
        neesFile << line << '\n';
    }

    {
        if (!poseConsistencyFile.is_open()) {
            poseConsistencyFile.open(outputDir + "poseConsistency.csv");
            poseConsistencyFile << "time, eps_rx, eps_ry, eps_rz, eps_px, eps_py, eps_pz,"
                                   "Sigma2_rx, Sigma2_ry, Sigma2_rz, Sigma2_px, Sigma2_py, Sigma2_pz\n";
        }
        poseConsistencyFile << std::setprecision(20) << stamp << ", " << std::setprecision(6);
        const liepp::SE3d errorPose = trueState.sensor.pose * filter.X.A.inverse();
        const liepp::se3d epsilon = liepp::SE3d::log(filter.xi0.sensor.pose.inverse() * errorPose);
        const Eigen::Vector<double, 6> sigmaPoseVector = filter.Sigma.diagonal().segment<6>(6);

        CSVLine line;
        line << epsilon << sigmaPoseVector;
        poseConsistencyFile << line << '\n';
    }

    {
        if (!cameraConsistencyFile.is_open()) {
            cameraConsistencyFile.open(outputDir + "cameraConsistency.csv");
            cameraConsistencyFile << "time, eps_rx, eps_ry, eps_rz, eps_px, eps_py, eps_pz,"
                                     "Sigma2_rx, Sigma2_ry, Sigma2_rz, Sigma2_px, Sigma2_py, Sigma2_pz\n";
        }
        cameraConsistencyFile << std::setprecision(20) << stamp << ", " << std::setprecision(6);
        const liepp::SE3d errorCamera = filter.X.A * trueState.sensor.cameraOffset * filter.X.B.inverse();
        const liepp::se3d epsilon = liepp::SE3d::log(filter.xi0.sensor.cameraOffset.inverse() * errorCamera);
        const Eigen::Vector<double, 6> sigmaCameraVector = filter.Sigma.diagonal().segment<6>(15);

        CSVLine line;
        line << epsilon << sigmaCameraVector;
        cameraConsistencyFile << line << '\n';
    }

    {
        if (!biasConsistencyFile.is_open()) {
            biasConsistencyFile.open(outputDir + "biasConsistency.csv");
            biasConsistencyFile
                << "time, eps_gyr_x, eps_gyr_y, eps_gyr_z, eps_acc_x, eps_acc_y, eps_acc_z,"
                   "Sigma2_gyr_x, Sigma2_gyr_y, Sigma2_gyr_z, Sigma2_acc_x, Sigma2_acc_y, Sigma2_acc_z\n";
        }
        biasConsistencyFile << std::setprecision(20) << stamp << ", " << std::setprecision(6);
        const liepp::se3d epsilon = trueState.sensor.inputBias - filter.X.beta - filter.xi0.sensor.inputBias;
        const Eigen::Vector<double, 6> sigmaBiasVector = filter.Sigma.diagonal().segment<6>(0);

        CSVLine line;
        line << epsilon << sigmaBiasVector;
        biasConsistencyFile << line << '\n';
    }
}