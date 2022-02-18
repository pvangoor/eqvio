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

#pragma once

#include <chrono>
#include <condition_variable>
#include <fstream>
#include <mutex>
#include <sstream>
#include <thread>

/** @brief asynchronous output file stream
 *
 * Uses a std thread to write data to an output stream asynchronously. When data is received through << by an aofstream,
 * it is buffered into a stringstream. A separate thread (started upon opening the output file) is used to flush the
 * buffer to the output file every few seconds. Upon destruction, the buffer is flushed completely and the thread is
 * destroyed. By using this class instead of a std::ofstream to write output, the program can mostly avoid waiting on
 * the OS.
 */
class aofstream {
  protected:
    /// The output file to which data is written
    std::ofstream outputFile;

    /// The buffer of data to write to the file
    std::stringstream streamBuffer;
    /// A flag to break the output thread loop when the destructor is called.
    bool destructorCalled = false;
    /// A condition variable that starts the output writing thread when new data is available.
    std::condition_variable buffer_cv;

    /// The thread that writes the streamBuffer to the outputFile.
    std::thread writerThread;
    /// A mutex to govern access to the stream buffer.
    std::mutex streamMutex;

    /** @brief Regularly flush the output buffer to the output file.
     *
     * Every few seconds, the stream buffer is flushed to the output file. This loop is paused until new data is
     * available and the condition variable is triggered. This function is intended to be run from the writerThread
     * only.
     */
    void flushStreamToFile() {
        std::chrono::steady_clock::time_point lastFlush = std::chrono::steady_clock::now();
        while (true) {
            std::unique_lock lck(streamMutex);
            buffer_cv.wait(lck, [this]() { return !(streamBuffer.rdbuf()->in_avail() == 0) || destructorCalled; });
            outputFile << streamBuffer.rdbuf();

            // Flush every 5 seconds
            if ((std::chrono::steady_clock::now() - lastFlush) > std::chrono::duration<double>(5.0)) {
                lastFlush = std::chrono::steady_clock::now();
                outputFile << std::flush;
            }

            if (this->destructorCalled) {
                break;
            }
        }
        if (!(streamBuffer.rdbuf()->in_avail() == 0)) {
            outputFile << streamBuffer.rdbuf();
        }
        outputFile << std::flush;
    }

  public:
    aofstream() = default;
    /** @brief Create an aofstream and immediately open fname for writing. */
    aofstream(const std::string& fname) {
        outputFile.open(fname);
        writerThread = std::thread(&aofstream::flushStreamToFile, this);
    }
    /** @brief Empty the stream buffer and finish the writing thread before destruction. */
    ~aofstream() {
        {
            std::unique_lock lck(streamMutex);
            this->destructorCalled = true;
        }
        buffer_cv.notify_one();
        if (writerThread.joinable()) {
            writerThread.join();
        }
        outputFile.close();
    }

    /** @brief Opens the given file for writing.
     *
     * @param fname The name of the file to open.
     */
    void open(const std::string& fname) {
        outputFile.open(fname);
        writerThread = std::thread(&aofstream::flushStreamToFile, this);
    }

    /** @brief Checks if the associated output file is open.
     *
     * @return true if the file is open.
     */
    bool is_open() { return outputFile.is_open(); }

    /** @brief Write data to the stream buffer to pass to the output file.
     *
     * @param data The data to be written to the output file.
     * @return this instance.
     *
     * Acquires the buffer mutex and writes data to the streambuffer. This data is flushed to the output file by the
     * writer thread. This function returns *this so that operator<< can be used in typical c++ fashion:
     *
     * myAofstream << data1 << data2;
     */
    aofstream& operator<<(const auto& data) {
        std::unique_lock lck(streamMutex);
        streamBuffer << data;
        buffer_cv.notify_one();
        return *this;
    }
};