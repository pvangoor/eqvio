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

#include "eqvio/dataserver/DataServerBase.h"

DataServerBase::DataServerBase(std::unique_ptr<DatasetReaderBase>&& datasetReader, const YAML::Node& simSettings)
    : datasetReaderPtr(std::move(datasetReader)) {
    simulator = VIOSimulator(datasetReaderPtr->groundtruth(), datasetReaderPtr->camera, simSettings);
    if (cameraExtrinsics()) {
        simulator.cameraOffset = *cameraExtrinsics();
    }
}

void DataServerBase::readCamera(const std::string& cameraFileName) {
    datasetReaderPtr->readCamera(cameraFileName);
    simulator.cameraPtr = datasetReaderPtr->camera;
    if (cameraExtrinsics()) {
        simulator.cameraOffset = *cameraExtrinsics();
    }
}

std::shared_ptr<liepp::SE3d> DataServerBase::cameraExtrinsics() const { return datasetReaderPtr->cameraExtrinsics; }

GIFT::GICameraPtr DataServerBase::camera() const { return datasetReaderPtr->camera; }

VisionMeasurement DataServerBase::getSimVision() {
    const StampedImage image = getImage();
    return simulator.getVision(image.stamp);
}

IMUVelocity DataServerBase::getSimIMU() {
    const IMUVelocity imu = getIMU();
    return simulator.getIMU(imu.stamp);
}

std::shared_ptr<liepp::SE3d> DataServerBase::groundTruthPose(const double stamp) const {
    if (simulator.viewPoses().empty()) {
        return nullptr;
    } else {
        return std::make_shared<liepp::SE3d>(simulator.getFullState(stamp, false).sensor.pose);
    }
}