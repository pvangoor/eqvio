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