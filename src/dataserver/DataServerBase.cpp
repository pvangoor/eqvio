#include "eqvio/dataserver/DataServerBase.h"

DataServerBase::DataServerBase(std::unique_ptr<DatasetReaderBase>&& datasetReader)
    : datasetReaderPtr(std::move(datasetReader)) {}

void DataServerBase::readCamera(const std::string& cameraFileName) { datasetReaderPtr->readCamera(cameraFileName); }

std::shared_ptr<liepp::SE3d> DataServerBase::cameraExtrinsics() const { return datasetReaderPtr->cameraExtrinsics; }

GIFT::GICameraPtr DataServerBase::camera() const { return datasetReaderPtr->camera; }

VisionMeasurement DataServerBase::getSimVision() {
    const StampedImage image = getImage();
    assert(datasetReaderPtr->simulator);
    return datasetReaderPtr->simulator->getVision(image.stamp);
}

IMUVelocity DataServerBase::getSimIMU() {
    const IMUVelocity imu = getIMU();
    assert(datasetReaderPtr->simulator);
    return datasetReaderPtr->simulator->getIMU(imu.stamp);
}