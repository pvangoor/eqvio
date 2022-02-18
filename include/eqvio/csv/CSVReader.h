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
