#include "eqvio/dataserver/SimpleDataServer.h"

MeasurementType SimpleDataServer::nextMeasurementType() const {
    if (nextImageData && nextIMUData) {
        return (nextImageData->stamp <= nextIMUData->stamp) ? MeasurementType::Image : MeasurementType::IMU;
    } else if (nextImageData) {
        return MeasurementType::Image;
    } else if (nextIMUData) {
        return MeasurementType::IMU;
    }
    return MeasurementType::None;
}

StampedImage SimpleDataServer::getImage() {
    StampedImage retImageData = *nextImageData;
    nextImageData = datasetReaderPtr->nextImage();
    return retImageData;
}

IMUVelocity SimpleDataServer::getIMU() {
    IMUVelocity retIMUData = *nextIMUData;
    nextIMUData = datasetReaderPtr->nextIMU();
    return retIMUData;
}

SimpleDataServer::SimpleDataServer(std::unique_ptr<DatasetReaderBase>&& datasetReader)
    : DataServerBase(std::move(datasetReader)) {
    nextImageData = datasetReaderPtr->nextImage();
    nextIMUData = datasetReaderPtr->nextIMU();
}

double SimpleDataServer::nextTime() const {
    MeasurementType nextMT = nextMeasurementType();
    if (nextMT == MeasurementType::Image) {
        return nextImageData->stamp;
    } else if (nextMT == MeasurementType::IMU) {
        return nextIMUData->stamp;
    }
    return std::nan("");
}