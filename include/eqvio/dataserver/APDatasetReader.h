#pragma once

#include "eqvio/csv/CSVFile.h"
#include "eqvio/dataserver/DatasetReaderBase.h"

/** @brief The ardupilot dataset reader.
 *
 * This class can be used to read datasets in the ArduPilot (AP) format.
 *
 * @todo Add a clear description of this format.
 */
class APDatasetReader : public DatasetReaderBase {
  protected:
    std::string cam_dir;  ///< The directory where camera images are stored.
    CSVFile IMUCSVFile;   ///< The CSV file containing IMU velocities
    CSVFile ImageCSVFile; ///< The CSV file containing image stamps and relative file names.

  public:
    virtual std::unique_ptr<StampedImage> nextImage() override;
    virtual std::unique_ptr<IMUVelocity> nextIMU() override;

    APDatasetReader() = default;

    /** @brief Construct the AP dataset reader from the given dataset directory and simulation settings.*/
    APDatasetReader(const std::string& datasetFileName, const YAML::Node& simSettings = YAML::Node());
    virtual void readCamera(const std::string& cameraFileName) override;
};
