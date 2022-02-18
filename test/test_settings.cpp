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

#include "eqvio/VIOFilterSettings.h"
#include "gtest/gtest.h"

TEST(TestSettings, ReadTemplateSettings) {
    const std::string configTemplateFileName = EQVIO_DEFAULT_CONFIG_FILE;
    const YAML::Node configNode = YAML::LoadFile(configTemplateFileName);

    EXPECT_TRUE(configNode["eqf"]);
    for (const auto& subnode : configNode["eqf"]) {
        std::cout << subnode.first.as<std::string>() << " ";
    }
    std::cout << std::endl;

    testing::internal::CaptureStdout();
    VIOFilter::Settings settings(configNode["eqf"]);
    const std::string output = testing::internal::GetCapturedStdout();
    EXPECT_STREQ(output.data(), "");
}