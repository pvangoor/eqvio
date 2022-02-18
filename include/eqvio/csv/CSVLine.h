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

#include <deque>
#include <fstream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "eigen3/Eigen/Core"
#include "liepp/SE3.h"
#include "liepp/SO3.h"
#include "liepp/SOT3.h"

/** @file */

/** @brief A line of csv values.
 *
 *  Reads a line as comma separated values and provides access to the data. By specialising operator<< and operator>>,
 * many types of data can be easily read from and written to csv lines. Additionally, it is easy to read data in a csv
 * that is listed in sequence. For example
 *
 * @code myCSVLine >> myString >> myInt; @endcode
 *
 * will read the first entry in the csv line as a string into myString, and then the next entry as an int into myInt.
 */
class CSVLine {
  protected:
    friend class CSVReader; ///< This is to permit access to readLine

    std::deque<std::string> data; ///< The buffer of data from the file as strings.

    /** @brief Read an istream into the data buffer.
     *
     * @param lineStream The input line to parse.
     * @param delim The delimiter separating the values in the line.
     */
    void readLine(std::istream& lineStream, const char& delim = ',') {
        data.clear();
        std::string entry;
        while (std::getline(lineStream, entry, delim)) {
            data.emplace_back(entry);
        }
    }

  public:
    CSVLine() = default;

    /** @brief Construct a CSVLine and read the provided line immediately
     *
     * @param lineStream The input line to parse.
     * @param delim The delimiter separating the values in the line.
     */
    CSVLine(std::istream& lineStream, const char& delim = ',') { readLine(lineStream, delim); }

    /** @brief Access the data buffer at idx.
     *
     * @param idx The index where the data buffer should be accessed.
     * @return The data in the buffer at idx.
     */
    std::string operator[](const size_t& idx) const { return data[idx]; }

    /** @brief Get the size of the csv data buffer
     */
    size_t size() const { return data.size(); }

    /** @brief Return true if the csv data buffer is empty.
     */
    bool empty() const { return data.empty(); }

    /** @brief Write the front of the data buffer to an arithmetic type
     *
     * @param d A reference to the arithmetic variable where the data should be placed.
     * @return A reference to this csv line.
     *
     * @todo Does this need to be part of the class itself?
     */
    template <typename T, std::enable_if_t<std::is_arithmetic<T>::value, bool> = true> CSVLine& operator>>(T& d) {
        std::stringstream(data.front()) >> d;
        data.pop_front();
        return *this;
    }

    /** @brief Read an arithmetic type to the back of the data buffer
     *
     * @param d The arithmetic variable to be read.
     * @return A reference to this csv line.
     *
     * @todo Does this need to be part of the class itself?
     */
    template <typename T, std::enable_if_t<std::is_arithmetic<T>::value, bool> = true> CSVLine& operator<<(const T& d) {
        std::stringstream ss;
        ss << d;
        data.emplace_back(ss.str());
        return *this;
    }

    /** @brief Write the front of the data buffer to a string
     *
     * @param s A reference to the string where the data should be placed.
     * @return A reference to this csv line.
     */
    CSVLine& operator>>(std::string& s) {
        s = data.front();
        data.pop_front();
        return *this;
    }

    /** @brief Read a stringstream to the back of the data buffer
     *
     * @param ss The stringstream to be read.
     * @return A reference to this csv line.
     */
    CSVLine& operator<<(const std::stringstream& ss) {
        data.emplace_back(ss.str());
        return *this;
    }

    /** @brief Read a string to the back of the data buffer
     *
     * @param s The string to be read.
     * @return A reference to this csv line.
     */
    CSVLine& operator<<(const std::string& s) {
        data.emplace_back(s);
        return *this;
    }
};

/** @brief Write a csv line to an output stream.
 *
 * @param os The output stream where the line should be written.
 * @param line The line to be written to the stream.
 * @return A reference to the output stream after writing.
 */
inline std::ostream& operator<<(std::ostream& os, const CSVLine& line) {
    if (!line.empty()) {
        for (size_t i = 0; i < line.size() - 1; ++i) {
            os << line[i] << ", ";
        }
        os << line[line.size() - 1];
    }
    return os;
}

/** @brief Read an Eigen Matrix from a csv line.
 *
 * @param line The line to be written to the stream.
 * @param a A reference to the Eigen Matrix where the data should be placed.
 * @return A reference to the csv line with the matrix data removed.
 */
template <typename Derived> CSVLine& operator>>(CSVLine& line, Eigen::MatrixBase<Derived>& a) {
    for (int i = 0; i < a.rows(); ++i) {
        for (int j = 0; j < a.cols(); ++j) {
            line >> a(i, j);
        }
    }
    return line;
}

/** @brief Read an Eigen Matrix to a csv line
 *
 * @param line The CSV line where the data should be placed.
 * @param a The Eigen Matrix to read.
 * @return A reference to the csv line with the data emplaced.
 *
 * Eigen matrices are stored into a csv in row-major order. Note that no information about the matrix size is included.
 */
template <typename Derived> CSVLine& operator<<(CSVLine& line, const Eigen::MatrixBase<Derived>& a) {
    for (int i = 0; i < a.rows(); ++i) {
        for (int j = 0; j < a.cols(); ++j) {
            line << a(i, j);
        }
    }
    return line;
}

/** @brief Write a csv line to a quaternion
 *
 * @param line The line containing the quaternion data.
 * @param q A reference to the quaternion where the data should be placed.
 * @return A reference to the csv line with the quaternion data removed.
 *
 * @note The order of quaternion representation used is (w,x,y,z). This is similar to the typical representation of
 * complex numbers, where the real part precedes the imaginary part.
 */
template <typename Derived> CSVLine& operator>>(CSVLine& line, Eigen::QuaternionBase<Derived>& q) {
    return line >> q.w() >> q.x() >> q.y() >> q.z();
}

/** @brief Write a quaternion to a csv line
 *
 * @param line The CSV line where the data should be placed.
 * @param q The quaternion to read.
 * @return A reference to the csv line with the data emplaced.
 *
 * @note The order of quaternion representation used is (w,x,y,z). This is similar to the typical representation of
 * complex numbers, where the real part precedes the imaginary part.
 */
template <typename Derived> CSVLine& operator<<(CSVLine& line, const Eigen::QuaternionBase<Derived>& q) {
    return line << q.w() << q.x() << q.y() << q.z();
}

/** @brief Write an SO(3) element to a csv line
 *
 * @param line The CSV line where the data should be placed.
 * @param R The SO(3) element to write.
 * @return A reference to the csv line with the data emplaced.
 */
inline CSVLine& operator<<(CSVLine& line, const liepp::SO3d& R) { return line << R.asQuaternion(); }

/** @brief Write a csv line to an SO(3) element
 *
 * @param line The line containing the SO(3) data.
 * @param R A reference to the SO(3) element where the data should be placed.
 * @return A reference to the csv line with the quaternion data removed.
 */
inline CSVLine& operator>>(CSVLine& line, liepp::SO3d& R) {
    Eigen::Quaterniond q;
    line >> q;
    R.fromQuaternion(q);
    return line;
}

/** @brief Write an SE(3) element to a csv line
 *
 * @param line The CSV line where the data should be placed.
 * @param P The SE(3) element to write.
 * @return A reference to the csv line with the data emplaced.
 */
inline CSVLine& operator<<(CSVLine& line, const liepp::SE3d& P) { return line << P.x << P.R; }
/** @brief Write a csv line to an SE(3) element
 *
 * @param line The line containing the SE(3) data.
 * @param P A reference to the SE(3) element where the data should be placed.
 * @return A reference to the csv line with the quaternion data removed.
 */
inline CSVLine& operator>>(CSVLine& line, liepp::SE3d& P) { return line >> P.x >> P.R; }

/** @brief Write a SOT(3) element to a csv line
 *
 * @param line The CSV line where the data should be placed.
 * @param Q The SOT(3) element to write.
 * @return A reference to the csv line with the data emplaced.
 */
inline CSVLine& operator<<(CSVLine& line, const liepp::SOT3d& Q) { return line << Q.a << Q.R; }
/** @brief Write a csv line to an SOT(3) element
 *
 * @param line The line containing the SOT(3) data.
 * @param Q A reference to the SOT(3) element where the data should be placed.
 * @return A reference to the csv line with the quaternion data removed.
 */
inline CSVLine& operator>>(CSVLine& line, liepp::SOT3d& Q) { return line >> Q.a >> Q.R; }
