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

#include "eqvio/IMUVelocity.h"

IMUVelocity IMUVelocity::Zero() {
    IMUVelocity result;
    result.stamp = 0;
    result.gyr.setZero();
    result.acc.setZero();
    return result;
}

IMUVelocity::IMUVelocity(const Eigen::Matrix<double, 6, 1>& vec) {
    stamp = 0;
    gyr = vec.segment<3>(0);
    acc = vec.segment<3>(3);
}

IMUVelocity::IMUVelocity(const Eigen::Matrix<double, 12, 1>& vec) {
    stamp = 0;
    gyr = vec.segment<3>(0);
    acc = vec.segment<3>(3);
    gyrBiasVel = vec.segment<3>(6);
    accBiasVel = vec.segment<3>(9);
}

IMUVelocity IMUVelocity::operator+(const IMUVelocity& other) const {
    IMUVelocity result;
    result.stamp = (this->stamp > 0) ? this->stamp : other.stamp;
    result.gyr = this->gyr + other.gyr;
    result.acc = this->acc + other.acc;
    result.gyrBiasVel = this->gyrBiasVel + other.gyrBiasVel;
    result.accBiasVel = this->accBiasVel + other.accBiasVel;
    return result;
}

IMUVelocity IMUVelocity::operator-(const Eigen::Matrix<double, 6, 1>& vec) const {
    IMUVelocity result;
    result.stamp = this->stamp;
    result.gyr = this->gyr - vec.segment<3>(0);
    result.acc = this->acc - vec.segment<3>(3);
    return result;
}
IMUVelocity IMUVelocity::operator-(const Eigen::Matrix<double, 12, 1>& vec) const {
    IMUVelocity result;
    result.stamp = this->stamp;
    result.gyr = this->gyr - vec.segment<3>(0);
    result.acc = this->acc - vec.segment<3>(3);
    result.gyrBiasVel = this->gyrBiasVel - vec.segment<3>(6);
    result.accBiasVel = this->accBiasVel - vec.segment<3>(9);
    return result;
}

IMUVelocity IMUVelocity::operator*(const double& c) const {
    IMUVelocity result;
    result.stamp = this->stamp;
    result.gyr = this->gyr * c;
    result.acc = this->acc * c;
    result.gyrBiasVel = this->gyrBiasVel * c;
    result.accBiasVel = this->accBiasVel * c;
    return result;
}

CSVLine& operator<<(CSVLine& line, const IMUVelocity& imu) {
    return line << imu.stamp << imu.gyr << imu.acc << imu.gyrBiasVel << imu.accBiasVel;
}
CSVLine& operator>>(CSVLine& line, IMUVelocity& imu) {
    line >> imu.stamp >> imu.gyr >> imu.acc;
    if (line.empty()) {
        return line;
    } else {
        return line >> imu.gyrBiasVel >> imu.accBiasVel;
    }
}