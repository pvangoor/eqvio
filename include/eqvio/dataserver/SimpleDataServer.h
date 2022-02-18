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

#include "eqvio/dataserver/DataServerBase.h"

/** @brief The simplest implementation of a data server
 *
 * This data server reads the next image and IMU data, and then passes them on as requested. When the image data is
 * requested, the currently stored data is provided and the next data is read in anticipation of the next request.
 */
class SimpleDataServer : public DataServerBase {
  protected:
    std::unique_ptr<StampedImage> nextImageData; ///< The next image measurement
    std::unique_ptr<IMUVelocity> nextIMUData;    ///< The next IMU measurement

  public:
    virtual MeasurementType nextMeasurementType() const override;
    virtual StampedImage getImage() override;
    virtual IMUVelocity getIMU() override;
    virtual double nextTime() const override;

    /** @brief construct the simple data server from a dataset reader.
     *
     * @param datasetReader An rvalue reference to a dataset reader unique_ptr.
     */
    SimpleDataServer(std::unique_ptr<DatasetReaderBase>&& datasetReader);
};
