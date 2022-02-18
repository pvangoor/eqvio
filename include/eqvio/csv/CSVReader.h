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

#include "eqvio/csv/CSVLine.h"

/// @todo decide if this is necessary or maybe rewrite as a viewer class

class CSVReader {
  protected:
    std::istream* csvPtr;
    CSVLine csvLine;
    char delim;

  public:
    CSVReader() { csvPtr = NULL; }
    CSVReader(std::istream& f, const char& delim = ',') : delim(delim) {
        csvPtr = f.good() ? &f : NULL;
        readNextLine();
    }

    CSVReader begin() { return *this; }
    CSVReader end() { return CSVReader(); }
    CSVLine operator*() { return csvLine; }
    bool operator==(const CSVReader& other) const {
        return (this->csvPtr == other.csvPtr) || ((this->csvPtr == NULL) && (other.csvPtr == NULL));
    }
    bool operator!=(const CSVReader& other) const { return !(*this == other); }

    CSVReader& operator++() {
        if (csvPtr) {
            // Try to read the next line
            if (!readNextLine()) {
                csvPtr = NULL;
            }
        }
        return *this;
    }

    bool readNextLine() {
        std::string lineString;
        bool fileNotEmpty = (bool)std::getline(*csvPtr, lineString, '\n');
        if (fileNotEmpty) {
            std::stringstream lineStream(lineString);
            csvLine.readLine(lineStream, delim);
        } else {
            csvLine = CSVLine();
        }
        return fileNotEmpty;
    }
};
