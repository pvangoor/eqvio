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

VIOWriter::VIOWriter(std::string outputDir) {

    if (outputDir.back() != '/') {
        outputDir = outputDir + "/";
    }

    if (!std::filesystem::is_directory(outputDir)) {
        std::filesystem::create_directories(outputDir);
    }

    IMUStateFile.open(outputDir + "IMUState.csv");
    cameraFile.open(outputDir + "camera.csv");
    biasFile.open(outputDir + "bias.csv");
    pointsFile.open(outputDir + "points.csv");
    featuresFile.open(outputDir + "features.csv");
    timingFile.open(outputDir + "timing.csv");

    IMUStateFile << "time, px, py, pz, qw, qx, qy, qz, vx, vy, vz\n";
    cameraFile << "time, px, py, pz, qw, qx, qy, qz\n";
    biasFile << "time, bias_gyr_x, bias_gyr_y, bias_gyr_z, bias_acc_x, bias_acc_y, bias_acc_z\n";
    pointsFile << "time, p1id, p1x, p1y, p1z, ...\n";
    featuresFile << "time, z1id, z1x, z1y, ...\n";
}

void VIOWriter::writeStates(const double& stamp, const VIOState& xi) {
    { // Write the IMU pose to file
        IMUStateFile << std::setprecision(20) << stamp << ", " << std::setprecision(6);
        CSVLine line;
        line << xi.sensor.pose << xi.sensor.velocity;
        IMUStateFile << line << '\n';
    }

    { // Write the Camera offset to file
        cameraFile << std::setprecision(20) << stamp << ", " << std::setprecision(6);
        CSVLine line;
        line << xi.sensor.cameraOffset;
        cameraFile << line << '\n';
    }

    { // Write the biases to file
        biasFile << std::setprecision(20) << stamp << ", " << std::setprecision(6);
        CSVLine line;
        line << xi.sensor.inputBias;
        biasFile << line << '\n';
    }

    { // Write the points in world frame to file
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
    featuresFile << std::setprecision(20) << y.stamp << ", " << std::setprecision(6);
    CSVLine line;
    for (const auto& [id, z] : y.camCoordinates) {
        line << id << z;
    }
    featuresFile << line << '\n';
}

void VIOWriter::writeTiming(const LoopTimer::LoopTimingData& timingData) {
    if (!timingFileHeaderIsWritten) {
        CSVLine header;
        header << "time";
        for (const auto& [lab, timing] : timingData.timings) {
            header << lab;
        }
        timingFile << header << '\n';

        timingFileHeaderIsWritten = true;
    }

    timingFile << std::setprecision(20) << timingData.loopTimeStart.count() << ", " << std::setprecision(6);
    CSVLine line;
    for (const auto& [lab, timing] : timingData.timings) {
        line << timing.count();
    }
    timingFile << line << '\n';
}
