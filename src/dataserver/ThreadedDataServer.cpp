#include "eqvio/dataserver/ThreadedDataServer.h"

MeasurementType ThreadedDataServer::nextMeasurementType() const {
    std::unique_lock lck(ioMutex);
    queuesReady_cv.wait(
        lck, [this] { return (IMUDataFinished || !IMUQueue.empty()) && (imageDataFinished || !imageQueue.empty()); });

    // Check if any of the data is finished
    if (imageQueue.empty() && IMUQueue.empty()) {
        return MeasurementType::None;
    } else if (imageQueue.empty() && !IMUQueue.empty()) {
        return MeasurementType::IMU;
    } else if (!imageQueue.empty() && IMUQueue.empty()) {
        return MeasurementType::Image;
    }

    // Otherwise, compare the stamps
    if (imageQueue.front().stamp <= IMUQueue.front().stamp) {
        return MeasurementType::Image;
    } else {
        return MeasurementType::IMU;
    }
}

StampedImage ThreadedDataServer::getImage() {
    std::unique_lock lck(ioMutex);
    StampedImage data = imageQueue.front();
    imageQueue.pop();
    return data;
}

IMUVelocity ThreadedDataServer::getIMU() {
    std::unique_lock lck(ioMutex);
    IMUVelocity data = IMUQueue.front();
    IMUQueue.pop();
    return data;
}

double ThreadedDataServer::nextTime() const {
    const MeasurementType measType = nextMeasurementType();
    std::unique_lock lck(ioMutex);
    if (measType == MeasurementType::IMU) {
        return IMUQueue.front().stamp;
    }
    if (measType == MeasurementType::Image) {
        return imageQueue.front().stamp;
    }
    return std::nan("");
}

ThreadedDataServer::ThreadedDataServer(std::unique_ptr<DatasetReaderBase>&& datasetReader)
    : DataServerBase(std::move(datasetReader)) {
    ioThread = std::thread(&ThreadedDataServer::fillQueues, this);
}

void ThreadedDataServer::fillQueues() {
    while ((!IMUDataFinished || !imageDataFinished) && !destructorCalled) {

        while (!queuesFilled()) {
            std::unique_lock lck(ioMutex);

            // Fill up the image and IMU queues
            if (imageQueue.size() < maxImageQueueSize) {
                std::unique_ptr<StampedImage> nextImageData = datasetReaderPtr->nextImage();
                if (nextImageData) {
                    imageQueue.emplace(*nextImageData);
                } else {
                    imageDataFinished = true;
                }
            }

            if (IMUQueue.size() < maxIMUQueueSize) {
                std::unique_ptr<IMUVelocity> nextIMUData = datasetReaderPtr->nextIMU();
                if (nextIMUData) {
                    IMUQueue.emplace(*nextIMUData);
                } else {
                    IMUDataFinished = true;
                }
            }

            if (IMUDataFinished && imageDataFinished) {
                break;
            }
        }
        queuesReady_cv.notify_one();
    }
}

bool ThreadedDataServer::queuesFilled() const {
    std::unique_lock lck(ioMutex);
    return (IMUDataFinished || IMUQueue.size() == maxIMUQueueSize) &&
           (imageDataFinished || imageQueue.size() == maxImageQueueSize);
}

ThreadedDataServer::~ThreadedDataServer() {
    destructorCalled = true;
    ioThread.join();
}