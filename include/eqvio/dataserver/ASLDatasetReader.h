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

#include "eqvio/csv/CSVFile.h"
#include "eqvio/dataserver/DatasetReaderBase.h"

/** @brief The ASL dataset reader.
 *
 * This class can be used to read datasets in the EuRoC (ASL) format. For details, please see
 * <a href="https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets">EuRoC</a>
 */
class ASLDatasetReader : public DatasetReaderBase {
  protected:
    std::string cam_dir;  ///< The directory where camera images are stored.
    CSVFile IMUCSVFile;   ///< The CSV file containing IMU velocities
    CSVFile ImageCSVFile; ///< The CSV file containing image stamps and relative file names.

  public:
    virtual std::unique_ptr<StampedImage> nextImage() override;
    virtual std::unique_ptr<IMUVelocity> nextIMU() override;
    virtual void readCamera(const std::string& cameraFileName) override;

    /** @brief Construct the ASL dataset reader from the given dataset directory and simulation settings. */
    ASLDatasetReader(const std::string& datasetFileName, const YAML::Node& simSettings = YAML::Node());
};