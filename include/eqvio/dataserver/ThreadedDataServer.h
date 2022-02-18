#pragma once

#include "eqvio/dataserver/DataServerBase.h"

#include "eqvio/csv/CSVReader.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

/** @brief A threaded implementation of a data server
 *
 * This data server uses a thread to continuously buffer IMU and image data. This can greatly increase program speed as
 * there is rarely a need to wait for new measurements to be read in the main thread.
 */
class ThreadedDataServer : public DataServerBase {
  protected:
    const size_t maxImageQueueSize = 200; ///< The maximum image data buffer size
    const size_t maxIMUQueueSize = 2000;  ///< The maximum IMU data buffer size

    mutable std::condition_variable queuesReady_cv; ///< A CV that is triggered when image and IMU data is ready.
    mutable std::mutex ioMutex;                     ///< A mutex to govern access to the image and IMU buffers.
    std::thread ioThread;                           ///< The thread that reads image and IMU data.
    std::atomic_bool destructorCalled = false;      ///< A flag to break the thread when the destructor is called.

    bool IMUDataFinished = false;   ///< A flag that is false so long more IMU data remains to be read.
    bool imageDataFinished = false; ///< A flag that is false so long more image data remains to be read.

    std::queue<IMUVelocity> IMUQueue;    ///< The buffer of IMU data.
    std::queue<StampedImage> imageQueue; ///< The buffer of image data.

    /** @brief Return true if the queues are at their maximum size or the data is finished.*/
    bool queuesFilled() const;

    /** @brief Fill the image and IMU data buffers.
     *
     * This function is intended to be run by the ioThread.
     */
    void fillQueues();

  public:
    /** @brief Construct a threaded data server from a dataset reader
     *
     * @param datasetReader An rvalue reference to a dataset reader unique_ptr.
     */
    ThreadedDataServer(std::unique_ptr<DatasetReaderBase>&& datasetReader);

    /** @brief Destroy the dataset reader, but ensure the io thread is stopped gracefully. */
    ~ThreadedDataServer();

    virtual MeasurementType nextMeasurementType() const override;
    virtual StampedImage getImage() override;
    virtual IMUVelocity getIMU() override;
    virtual double nextTime() const override;
};
