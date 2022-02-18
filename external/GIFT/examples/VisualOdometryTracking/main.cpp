/*
    This file is part of GIFT.

    GIFT is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    GIFT is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with GIFT.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "GIFT/PointFeatureTracker.h"
#include "GIFT/Visualisation.h"

#include "opencv2/highgui/highgui.hpp"
#include "yaml-cpp/yaml.h"

#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 4) {
        std::cout << "Usage: VisualOdometryTracking camera.yaml video.mp4 (settings.yaml)" << std::endl;
        exit(1);
    }

    // Set up the feature tracker
    GIFT::StandardCamera camera{cv::String(argv[1])};
    GIFT::PointFeatureTracker ft = GIFT::PointFeatureTracker(camera);
    ft.settings.maxFeatures = 50;
    ft.settings.featureDist = 15;
    ft.settings.minHarrisQuality = 0.05;
    ft.settings.featureSearchThreshold = 0.8;

    if (argc == 4) {
        // std::cout << "Reading filter settings from " << argv[3] << std::endl;
        const YAML::Node configNode = YAML::LoadFile(std::string(argv[3]));
        ft.settings.configure(configNode["GIFT"]);
    }

    // Set up the video capture
    cv::VideoCapture cap;
    const cv::String videoFname(argv[2]);
    cap.open(videoFname);
    cv::Mat image;

    cv::VideoWriter writer;
    const cv::String outputVideoFname =
        videoFname.substr(0, videoFname.rfind('.')) + "_features" + videoFname.substr(videoFname.rfind('.'));

    // Set up the output file
    std::time_t t = std::time(nullptr);
    std::stringstream outputFileNameStream, internalFileNameStream;
    outputFileNameStream << "GIFT_Monocular_" << std::put_time(std::localtime(&t), "%F_%T") << ".csv";
    std::ofstream outputFile(outputFileNameStream.str());
    outputFile << "frame, N, eta1id, eta1x, eta1y, eta1z, ..., ..., ..., ..., etaNid, etaNx, etaNy, etaNz" << std::endl;

    int frameCounter = 0;
    while (cap.read(image)) {

        if (!writer.isOpened()) {
            const bool isColor = (image.type() == CV_8UC3);
            const int fourcc = cap.get(cv::CAP_PROP_FOURCC);
            writer.open(outputVideoFname, fourcc, cap.get(cv::CAP_PROP_FPS), image.size(), isColor);
        }

        ft.processImage(image);
        std::vector<GIFT::Feature> features = ft.outputFeatures();

        // Write features to file
        outputFile << frameCounter << ", ";
        outputFile << features.size();
        for (const GIFT::Feature f : features) {
            outputFile << ", " << f.idNumber;
            outputFile << ", " << f.sphereCoordinates().format(Eigen::IOFormat(-1, 0, ", ", ", "));
        }
        outputFile << std::endl;

        // Write image to file
        const cv::Mat featureImage = GIFT::drawFeatureImage(image, features, 3, cv::Scalar(0, 255, 0));
        writer.write(featureImage);

        ++frameCounter;
    }
}