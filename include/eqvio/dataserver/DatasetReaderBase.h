#pragma once

#include "eqvio/IMUVelocity.h"
#include "eqvio/VIOSimulator.h"
#include "eqvio/VisionMeasurement.h"

#include "opencv4/opencv2/core.hpp"
#include "opencv4/opencv2/highgui.hpp"
#include "yaml-cpp/yaml.h"

/** @brief The possible measurement types encountered in a VIO dataset */
enum class MeasurementType { Image, IMU, None };

/** @brief A struct carrying an openCV image and a timestamp */
struct StampedImage {
    cv::Mat image; ///< The image data
    double stamp;  ///< The time stamp of the image
};

/** @brief The common interface of all dataset readers.
 *
 * This abstraction helps to create a common interface between different dataset formats like EuRoC and UZH-FPV. It is
 * an abstract base class and cannot be instantiated itself, although pointers are allowed.
 */
class DatasetReaderBase {
  public:
    /** @brief get the next image data.
     *
     * @return a pointer to the next image data in the dataset. The pointer is null if no data remains
     */
    virtual std::unique_ptr<StampedImage> nextImage() = 0;

    /** @brief get the next IMU data.
     *
     * @return a pointer to the next IMU data in the dataset. The pointer is null if no data remains
     */
    virtual std::unique_ptr<IMUVelocity> nextIMU() = 0;

    /** @brief Read a camera file.
     *
     * Many datasets have default camera file locations, but some do not (e.g. ROSBags, UZH-FPV). This method lets the
     * user pass different camera parameters easily.
     */
    virtual void readCamera(const std::string& cameraFileName) = 0;

    std::unique_ptr<VIOSimulator>
        simulator; ///< A simulator that can be used to generate measurements from groundtruth data.

    GIFT::GICameraPtr camera;                      ///< A shared pointer to the camera associated with the dataset.
    std::shared_ptr<liepp::SE3d> cameraExtrinsics; ///< A shared pointer to the camera pose w.r.t. the IMU.
    double cameraLag = 0.0;                        ///< The lag of the image data in the dataset w.r.t. the IMU data.
};
