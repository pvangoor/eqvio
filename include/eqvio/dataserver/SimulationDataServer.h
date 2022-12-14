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

#include "eqvio/dataserver/DataServerBase.h"

/** @brief A data server that provides purely simulated data
 *
 * This data server creates a simulator instance using an artificial trajectory. It provides data at whatever
 * frequencies are chosen.
 */
class SimulationDataServer : public DataServerBase {
  protected:
    double imageFreq = 20.0;          /// The frequency of image measurements in Hz
    double imuFreq = 200.0;           /// The frequency of IMU measurements in Hz
    double maxSimulationTime = 100.0; /// The time at which the simulation should end in s.

    int imuMeasCount = 0;   /// The number of IMU measurements provided.
    int imageMeasCount = 0; /// The number of image measurements provided.

    double nextImageTime() const;
    double nextIMUTime() const;

  public:
    virtual MeasurementType nextMeasurementType() const override;
    virtual StampedImage getImage() override;
    virtual IMUVelocity getIMU() override;
    virtual double nextTime() const override;

    virtual VisionMeasurement getSimVision() override;
    virtual IMUVelocity getSimIMU() override;

    /** @brief Get the exact initial condition of the VIO system. */
    virtual VIOState getInitialCondition() const;

    /** @brief Get the true state of the VIO State at a time
     *
     * @param stamp The time stamp for the desired state.
     * @param withNoise True if noise should be added to the requested state.
     * @return The VIO state at the time stamp with noise if requested.
     */
    virtual VIOState getTrueState(const double& stamp, const bool& withNoise = false) const;

    virtual std::shared_ptr<liepp::SE3d> cameraExtrinsics() const;

    std::vector<StampedPose> generateTrajectory(const YAML::Node& choice) const;

    /** @brief construct the simulation data server without provided trajectory.
     *
     * @param simSettings The settings to provide to the simulator.
     */
    SimulationDataServer(
        const YAML::Node& simSettings = YAML::Node(),
        const VIOFilter::Settings& filterSettings = VIOFilter::Settings());
};
