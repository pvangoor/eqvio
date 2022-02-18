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