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

#include "eqvio/csv/CSVReader.h"
#include <memory>

/** @brief A class for opening and reading CSV files
 *
 * @todo Consider a simpler interface for creating a view of a file as a csv file, similar to python csv.
 */
class CSVFile {
  protected:
    std::shared_ptr<std::ifstream> filePtr; ///< The file to be read.

  public:
    CSVReader reader; ///< The iterator used to read the file.

    CSVFile() : reader(){};

    /** @brief Open the file at filename as a csv file with the chosen delimiter.
     *
     * @param fileName The name of the file to open.
     * @param delim The delimiter separating values in the file.
     */
    CSVFile(const std::string& fileName, const char& delim = ',') {
        filePtr = std::make_shared<std::ifstream>(fileName);
        std::istream& file = *filePtr.get();
        reader = CSVReader(file, delim);
    }

    /** @brief Return true if there are more lines to read in the file.
     */
    operator bool() const { return reader != CSVReader() && filePtr->good(); }

    /** @brief Read the next line in the csv file.
     *
     * @return the line that was read.
     */
    CSVLine nextLine() {
        CSVLine line = *reader;
        ++reader;
        return line;
    }

    /** @brief Skip a line in the csv file.
     *
     * @todo Remove this in favor of using nextLine.
     */
    void skipLine() { ++reader; }
};
