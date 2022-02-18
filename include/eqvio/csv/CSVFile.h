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
