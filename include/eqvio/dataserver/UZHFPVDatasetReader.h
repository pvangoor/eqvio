#pragma once

#include "eqvio/csv/CSVFile.h"
#include "eqvio/dataserver/DatasetReaderBase.h"

/** @brief The UZH-FPV dataset reader.
 *
 * This class can be used to read datasets in the UZH-FPV format. For details, please see
 * <a href="https://fpv.ifi.uzh.ch/">UZH-FPV</a>
 */
class UZHFPVDatasetReader : public DatasetReaderBase {
  protected:
    std::string baseDir;  ///< The base directory of the dataset.
    CSVFile IMUCSVFile;   ///< The CSV file containing IMU velocities
    CSVFile ImageCSVFile; ///< The CSV file containing image stamps and relative file names.

  public:
    virtual std::unique_ptr<StampedImage> nextImage();
    virtual std::unique_ptr<IMUVelocity> nextIMU();
    virtual void readCamera(const std::string& cameraFileName);

    /** @brief Construct the ASL dataset reader from the given dataset directory and simulation settings. */
    UZHFPVDatasetReader(const std::string& datasetFileName, const YAML::Node& simSettings = YAML::Node());
};
