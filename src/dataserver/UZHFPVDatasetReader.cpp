#include "eqvio/dataserver/UZHFPVDatasetReader.h"
#include "yaml-cpp/yaml.h"
#include <filesystem>
#include <iostream>

UZHFPVDatasetReader::UZHFPVDatasetReader(const std::string& datasetMainDir, const YAML::Node& simSettings) {
    // Set up required structures
    // --------------------------

    baseDir = datasetMainDir;

    // Set up the data iterator for IMU
    IMUCSVFile = CSVFile(datasetMainDir + "imu.txt", ' ');
    IMUCSVFile.skipLine(); // skip the header

    // Read the image data file
    ImageCSVFile = CSVFile(datasetMainDir + "left_images.txt", ' ');
    ImageCSVFile.skipLine(); // skip the header

    // Get the ground truth
    const std::string groundtruthFileName = datasetMainDir + "groundtruth.txt";
    if (std::filesystem::exists(groundtruthFileName)) {
        // simulator = VIOSimulator(groundtruthFileName, simSettings, 1.0, 1, ' ');
    }

    // Read the camera file
    std::string cameraFileName =
        baseDir + "../indoor_forward_calib_snapdragon/camchain-imucam-..indoor_forward_calib_snapdragon_imu.yaml";
    if (std::filesystem::exists(cameraFileName)) {
        readCamera(cameraFileName);
    } else {
        std::cout << "Camera file was not found at:\n" << cameraFileName << std::endl;
        throw std::runtime_error("The camera file could not be found.");
    }
}

std::unique_ptr<IMUVelocity> UZHFPVDatasetReader::nextIMU() {
    if (!IMUCSVFile) {
        return nullptr;
    }

    CSVLine imuLine = IMUCSVFile.nextLine();
    int num;
    IMUVelocity temp;
    imuLine >> num >> temp;
    return std::make_unique<IMUVelocity>(temp);
}

std::unique_ptr<StampedImage> UZHFPVDatasetReader::nextImage() {
    if (!ImageCSVFile) {
        return nullptr;
    }
    StampedImage temp;

    CSVLine imageLine = ImageCSVFile.nextLine();
    std::string nextImageFname;
    double rawStamp;
    int num;
    imageLine >> num >> rawStamp >> nextImageFname;

    nextImageFname = baseDir + nextImageFname;
    if (*nextImageFname.rbegin() == '\r') {
        nextImageFname.erase(nextImageFname.end() - 1);
    }

    temp.stamp = rawStamp - cameraLag;
    temp.image = cv::imread(nextImageFname);

    return std::make_unique<StampedImage>(temp);
}

void UZHFPVDatasetReader::readCamera(const std::string& cameraFileName) {
    YAML::Node cameraFileNode = YAML::LoadFile(cameraFileName);

    // Read the camera intrinsics
    cameraFileNode = cameraFileNode["cam0"];

    cv::Size imageSize;
    imageSize.width = cameraFileNode["resolution"][0].as<int>();
    imageSize.height = cameraFileNode["resolution"][1].as<int>();

    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = cameraFileNode["intrinsics"][0].as<double>();
    K.at<double>(1, 1) = cameraFileNode["intrinsics"][1].as<double>();
    K.at<double>(0, 2) = cameraFileNode["intrinsics"][2].as<double>();
    K.at<double>(1, 2) = cameraFileNode["intrinsics"][3].as<double>();

    std::array<double, 4> distortion;
    distortion = cameraFileNode["distortion_coeffs"].as<std::array<double, 4>>();

    camera = std::make_shared<GIFT::EquidistantCamera>(imageSize, K, distortion);

    // Read the camera extrinsics
    const std::vector<std::vector<double>> extrinsics_rows =
        cameraFileNode["T_cam_imu"].as<std::vector<std::vector<double>>>();
    Eigen::Matrix4d extrinsics_matrix;
    assert(extrinsics_rows.size() == 4);
    for (int i = 0; i < 4; ++i) {
        assert(extrinsics_rows[i].size() == 4);
        extrinsics_matrix.row(i) = Eigen::Vector4d(extrinsics_rows[i].data()).transpose();
    }
    // The UZH FPV dataset reports the pose of the IMU w.r.t. the camera, so it needs to be inverted.
    cameraExtrinsics = std::make_unique<liepp::SE3d>(extrinsics_matrix.inverse());
}