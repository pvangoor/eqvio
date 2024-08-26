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

#include "eqvio/VIOFilterSettings.h"
#include "eqvio/VIOWriter.h"
#include "eqvio/dataserver/SimulationDataServer.h"

#include "yaml-cpp/yaml.h"

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <thread>

#include "argparse/argparse.hpp"

#include "eqvio/VIOVisualiser.h"

VisionMeasurement convertGIFTFeatures(const std::vector<GIFT::Feature>& GIFTFeatures, const double& stamp);

int main(int argc, char const* argv[]) {
    argparse::ArgumentParser program("EqVIO Simulation");
    program.add_argument("-d", "--display")
        .help("Display a live plot of the system state")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("config").help("The yaml file with the filter and feature tracker configuration.");
    program.add_argument("-o", "--output")
        .help("Directory for storing the filter output files.")
        .default_value(std::string("-"));
    program.add_argument("--limitRate")
        .help("Limit the playback rate of the data. Default 0.0 (no limit).")
        .default_value(0.0)
        .scan<'g', double>();

    program.add_argument("--landmarkReset")
        .help("Reset the landmarks every n seconds. Default 0.0 (never).")
        .default_value(0.0)
        .scan<'g', double>();
    program.add_argument("--fullState")
        .help("All landmarks are always part of the state.")
        .default_value(false)
        .implicit_value(true);

    // Read the arguments
    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cout << err.what() << '\n' << program;
        exit(0);
    }

    const std::string configFileName = program.get("config");
    YAML::Node eqvioConfig = YAML::LoadFile(configFileName);
    bool writeStateFlag = configOrDefault(eqvioConfig, "main:writeState", true);
    const double cameraLag = configOrDefault(eqvioConfig, "main:cameraLag", 0.0);

    const bool displayFlag = program.get<bool>("--display");
    const double landmarkResetTime = program.get<double>("--landmarkReset");
    double lastLandmarkReset = landmarkResetTime > 0 ? 0.0 : std::nan("");

    VIOFilter::Settings filterSettings(eqvioConfig["eqf"]);
    SimulationDataServer simDataServer = SimulationDataServer(eqvioConfig["sim"], filterSettings);

    loopTimer.initialise(
        {"correction", "features", "preprocessing", "propagation", "total", "total vision update", "write output"});

    const double limitRateSetting = program.get<double>("--limitRate");

    // Read I/O settings

    if (displayFlag && !EQVIO_BUILD_VISUALISATION) {
        std::cout << "Visualisations have been requested, but the necessary module is not set to be built."
                  << std::endl;
    }
    VIOVisualiser visualiser(displayFlag);

    // Initialise the filter
    if (simDataServer.cameraExtrinsics()) {
        std::cout << "Camera extrinsics were provided by the dataset, overriding those in the filter settings."
                  << std::endl;
        filterSettings.cameraOffset = *simDataServer.cameraExtrinsics();
    }
    std::cout << "The camera extrinsics (pose of the camera w.r.t. the IMU) are:\n"
              << filterSettings.cameraOffset.asMatrix() << std::endl;

    VIOFilter filter(simDataServer.getInitialCondition(), filterSettings);

    // Set up output files

    std::stringstream outputFileNameStream, internalFileNameStream;
    std::time_t t0 = std::time(nullptr);
    if (program.is_used("--output")) {
        outputFileNameStream << program.get<std::string>("--output");
        if (!writeStateFlag) {
            writeStateFlag = true;
            std::cout
                << "User provided an output file, so the output writing flag from the configuration will be ignored."
                << std::endl;
        }
    } else {
        outputFileNameStream << "EQVIO_output_" << std::put_time(std::localtime(&t0), "%F_%T") << "/";
    }
    VIOWriter vioWriter(outputFileNameStream.str());

    int imuDataCounter = 0, visionDataCounter = 0;
    const std::chrono::steady_clock::time_point loopStartTime = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point rateLimitTimer = std::chrono::steady_clock::now();

    std::cout << "NEES:\n";

    while (true) {
        MeasurementType measType = simDataServer.nextMeasurementType();

        if (measType == MeasurementType::None) {
            // End of data
            std::cout << "\n\n";
            break;
        }

        if (measType == MeasurementType::Image) {
            VisionMeasurement measData = simDataServer.getSimVision();
            if (!program.get<bool>("--fullState")) {
                filter.augmentLandmarkStates(measData.getIds(), simDataServer.getTrueState(measData.stamp, true));
            }
            filter.processVisionData(measData);
            ++visionDataCounter;

            // Output filter data
            VIOState estimatedState = filter.stateEstimate();
            const VIOState trueState = simDataServer.getTrueState(filter.getTime());
            const double NEES = filter.viewEqFState().computeNEES(trueState);
            if (writeStateFlag) {
                vioWriter.writeStates(filter.getTime(), estimatedState);
                vioWriter.writeFeatures(measData);
                vioWriter.writeLandmarkError(filter.getTime(), trueState, estimatedState);
                vioWriter.writeConsistency(filter.getTime(), trueState, filter.viewEqFState());
            }

            std::cout << '\r' << NEES << std::flush;

            // Optionally visualise the filter data
            if (displayFlag) {
                visualiser.updateMapDisplay(estimatedState, filter.getTime());
            }

            // Limit the loop rate
            if (limitRateSetting > 0) {
                std::this_thread::sleep_until(rateLimitTimer + std::chrono::duration<double>(1.0 / limitRateSetting));
                rateLimitTimer = std::chrono::steady_clock::now();
            }

        } else if (measType == MeasurementType::IMU) {
            IMUVelocity imuData = simDataServer.getIMU();
            assert(!imuData.acc.hasNaN());
            assert(!imuData.gyr.hasNaN());

            filter.processIMUData(imuData);
            ++imuDataCounter;

            // Reset landmarks if requested
            if (filter.getTime() >= lastLandmarkReset + landmarkResetTime) {
                lastLandmarkReset += landmarkResetTime;
                const VIOState xi = simDataServer.getTrueState(filter.getTime(), true);
                filter.setLandmarks(xi.cameraLandmarks);
            }
        }
    }

    const auto elapsedTime =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - loopStartTime);
    std::cout << "Processed " << imuDataCounter << " IMU and " << visionDataCounter << " vision measurements.\n"
              << "Time taken: " << elapsedTime.count() * 1e-3 << " seconds." << std::endl;

    return 0;
}