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

#include "eqvio/csv/CSVReader.h"
#include "gtest/gtest.h"

const std::string testData =
    R"(1370925.32417695,0.0127831735,0.013848438,-0.013848438,-0.16759412,9.299079,2.1739352
1370925.3447114,-0.025566347,-0.043675844,-0.004261058,0.15562311,9.436746,2.2110453
1370925.36550249,-0.07243799,-0.010652645,-0.040480047,-0.033518825,9.387665,2.6360161
1370925.38669978,-0.05539375,-0.013848438,-0.020240024,-0.32202014,9.314642,3.098097
1370925.40955832,0.033023197,-0.041545313,0.024501082,-0.18794483,9.337387,2.9855695
1370925.42916753,0.043675844,-0.046871636,0.026631612,-0.16759412,9.411607,2.565387
1370925.45042336,-0.0063915867,-0.040480047,0.026631612,-0.12090719,9.491813,2.457648
1370925.4716602,-0.027696876,-0.046871636,0.018109497,0.04189853,9.411607,3.1316159
1370925.49317653,0.036218993,-0.037284255,0.00958738,-0.05865794,9.356541,3.0430303
1370925.51373899,0.084155895,-0.07243799,0.040480047,-0.09935937,9.356541,2.8323407
1370925.53606574,0.0777643,-0.080960095,0.0479369,0.10654198,9.403227,2.255338)";

TEST(CSVReaderTest, Instantiation) {
    std::stringstream csvFile = std::stringstream(testData);
    EXPECT_NO_THROW(CSVReader reader(csvFile));
}

TEST(CSVReaderTest, IteratorLoop) {
    std::stringstream csvFile = std::stringstream(testData);
    CSVReader reader(csvFile);
    while (reader != CSVReader()) {
        CSVLine line = *reader;
        EXPECT_NO_THROW(std::string temp = line[0]);
        ++reader;
    }
}

TEST(CSVReaderTest, RangeLoop) {
    std::stringstream csvFile = std::stringstream(testData);
    CSVReader reader(csvFile);
    for (auto row : reader) {
        EXPECT_NO_THROW(std::string temp = row[0]);
    }
}