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