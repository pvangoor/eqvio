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

#include "iostream"
#include "string"
#include "vector"
#include <stdexcept>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "GIFT/EgoMotion.h"
#include "GIFT/PointFeatureTracker.h"
#include "GIFT/Visualisation.h"
#include <getopt.h>
#include <sys/time.h>

static double time_seconds() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return tp.tv_sec + tp.tv_usec * 1.0e-6;
}

int main(int argc, char* argv[]) {
    cv::String camConfigFile;
    cv::String videoFile;
    int c;
    extern int optind;
    extern char* optarg;
    bool show_images = true;
    bool quiet = false;
    bool show_fps = false;
    unsigned maxCount = 1000;

    while ((c = getopt(argc, argv, "qnM:")) != -1) {
        switch (c) {
        case 'n':
            show_images = false;
            break;
        case 'q':
            quiet = true;
            break;
        case 'M':
            maxCount = atoi(optarg);
            break;
        }
    }
    argv += optind;
    argc -= optind;

    if (argc < 2) {
        throw std::runtime_error("You must provide exactly the camera calibration and the video file.");
    }
    camConfigFile = argv[0];
    videoFile = argv[1];

    // Set up a monocular feature tracker
    GIFT::StandardCamera cam0 = GIFT::StandardCamera(camConfigFile);
    GIFT::PointFeatureTracker ft = GIFT::PointFeatureTracker(cam0);
    ft.settings.maxFeatures = 250;
    ft.settings.featureDist = 20;

    cv::VideoCapture cap(videoFile);

    cv::Mat image;
    unsigned count = 0;
    double tstart = time_seconds();
    unsigned report_count = 0;

    while (cap.read(image) && ++count < maxCount) {

        // Track the features
        ft.processImage(image);

        std::vector<GIFT::Feature> features = ft.outputFeatures();

        // Compute EgoMotion
        GIFT::EgoMotion egoMotion(features);
        if (!quiet) {
            std::cout << "Estimated Linear Velocity:" << std::endl;
            std::cout << egoMotion.linearVelocity << '\n';
            std::cout << "Estimated Angular Velocity:" << std::endl;
            std::cout << egoMotion.angularVelocity << '\n' << std::endl;
        }

        auto estFlows = egoMotion.estimateFlowsNorm(features);

        if (++report_count == 100) {
            double now = time_seconds();
            double dt = now - tstart;
            printf("%.2f fps\n", report_count / dt);
            tstart = now;
            report_count = 0;
        }

        if (!show_images) {
            continue;
        }

        // cv::Mat flowImage = ft.drawFlowImage(cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 255), 3, 2);
        // cv::imshow("flow", flowImage);

        // Draw the normalised flow and estimates
        constexpr int viewScale = 500;
        cv::Mat estFlowImage(viewScale * 2, viewScale * 2, CV_8UC3, cv::Scalar(255, 255, 255));
        for (const auto& flow : estFlows) {
            cv::Point2f p1 = flow.first;
            cv::Point2f p0 = p1 - cv::Point2f(flow.second.x(), flow.second.y());
            p0 = cv::Point2f(viewScale, viewScale) + p0 * viewScale;
            p1 = cv::Point2f(viewScale, viewScale) + p1 * viewScale;
            cv::line(estFlowImage, p0, p1, cv::Scalar(255, 0, 0));
        }
        for (const auto& lm : features) {
            cv::Point2f p1 = lm.camCoordinatesNorm();
            cv::Point2f p0 = p1 - cv::Point2f(lm.opticalFlowNorm.x(), lm.opticalFlowNorm.y());
            p0 = cv::Point2f(viewScale, viewScale) + p0 * viewScale;
            p1 = cv::Point2f(viewScale, viewScale) + p1 * viewScale;
            cv::line(estFlowImage, p0, p1, cv::Scalar(255, 0, 255));
            cv::circle(estFlowImage, p0, 2, cv::Scalar(255, 0, 0));
        }
        cv::imshow("estimated flow", estFlowImage);

        cv::waitKey(1);
    }

    std::cout << "Completed " << count << " frames" << std::endl;
}
