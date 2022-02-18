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

#include "eqvio/LoopTimer.h"
#include "eqvio/VIOState.h"
#include "eqvio/aofstream.h"

/** @brief A class to handle writing output for the VIO system
 *
 * The output is distributed over a number of files, which are organised in a given directory. The use of aofstream
 * rather than ofstream means that the VIO system usually does not have to slow down or wait for the OS to write to
 * files.
 */
class VIOWriter {
  protected:
    aofstream IMUStateFile; ///< The file for recording the IMU states
    aofstream cameraFile;   ///< The file for recording the camera offset
    aofstream biasFile;     ///< The file for recording the IMU biases
    aofstream pointsFile;   ///< The file for recording the landmark points

    aofstream featuresFile; ///< The file for recording the image features

    aofstream timingFile;                   ///< The file for recording processing times
    bool timingFileHeaderIsWritten = false; ///< True once the header of the timing file is written.

  public:
    /** @brief Create a number of files in the output directory to record VIO data.
     *
     * @param outputDir The output directory where files should be stored.
     *
     * This method creates the output directory if it does not already exist, and then creates output files for the VIO
     * data to be written to. It also creates header rows in each of the files to make them human-readable.
     */
    VIOWriter(std::string outputDir);

    /** @brief Write the VIO states to the relevant files.
     *
     * @param stamp The timestamp associated with the given state.
     * @param xi The current VIO system state estimate.
     *
     * The VIO states are divided into IMU states, IMU biases, camera offset, and landmark points. All four of these are
     * written to separate output files and time stamped.
     */
    void writeStates(const double& stamp, const VIOState& xi);

    /** @brief Write the feature measurements to file.
     *
     * @param y The vision measurement to be written.
     */
    void writeFeatures(const VisionMeasurement& y);

    /** @brief Write the system processing time to file.
     *
     * @param timingData The timing information for the latest loop of the VIO system.
     *
     * Note that, on the first call to this function, the timing data is also used to create a header for the timing
     * data file.
     */
    void writeTiming(const LoopTimer::LoopTimingData& timingData);
};