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

#include "GIFT/PatchFeatureTracker.h"
#include "GIFT/Visualisation.h"

#include "opencv2/highgui/highgui.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    const GIFT::StandardCamera cameraParams{cv::String(argv[1])};
    GIFT::PatchFeatureTracker<GIFT::TranslationGroup> ft =
        GIFT::PatchFeatureTracker<GIFT::TranslationGroup>(cameraParams);

    cv::VideoCapture cap;
    cap.open(cv::String(argv[2]));
    cv::Mat image;
    int counter = 0;
    while (cap.read(image)) {
        ft.trackFeatures(image);
        ft.detectFeatures(image);

        std::vector<GIFT::Feature> features = ft.outputFeatures();

        cv::Mat featureImage = GIFT::drawFeatureImage(image, features);

        cv::imshow("debug", featureImage);
        int k = cv::waitKey(0);
        std::cout << "Read Image " << ++counter << std::endl;
        std::cout << "Number of features is  " << features.size() << std::endl;
        if (k == 's')
            cv::imwrite("FeatureImage.png", featureImage);
        if (k == 27)
            break;
    }
}