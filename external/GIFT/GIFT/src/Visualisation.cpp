#include "GIFT/Visualisation.h"

#include "opencv2/imgproc.hpp"
#include <algorithm>

using namespace GIFT;
using namespace std;
using namespace cv;

Mat GIFT::drawFeatureImage(
    const Mat& baseImage, const vector<Feature>& features, const int& radius, const Scalar& color) {
    Mat featureImage = baseImage.clone();

    auto drawingLambda = [&](const Feature& feature) { circle(featureImage, feature.camCoordinates, radius, color); };
    for_each(features.begin(), features.end(), drawingLambda);

    return featureImage;
}

Mat GIFT::drawFlowImage(const Mat& baseImage, const vector<Feature>& features0, const vector<Feature>& features1,
    const int& radius, const Scalar& circleColor, const int& thickness, const Scalar& lineColor) {
    Mat flowImage = baseImage.clone();

    auto flowDrawingLambda = [&](const Feature& lm0) {
        auto lm1it = find_if(
            features1.begin(), features1.end(), [&lm0](const Feature& lm1) { return lm1.idNumber == lm0.idNumber; });
        if (lm1it != features1.end()) {
            circle(flowImage, lm0.camCoordinates, radius, circleColor);
            line(flowImage, lm0.camCoordinates, lm1it->camCoordinates, lineColor, thickness);
        }
    };
    for_each(features0.begin(), features0.end(), flowDrawingLambda);

    return flowImage;
}

Mat GIFT::drawFlowImage(const Mat& image0, const Mat& image1, const vector<Feature>& features0,
    const vector<Feature>& features1, const int& radius, const Scalar& circleColor, const int& thickness,
    const Scalar& lineColor) {
    // draw the flow image on a red/blue merge of image0 and image1
    Mat gray0 = image0;
    if (image0.channels() > 1)
        cvtColor(image0, gray0, COLOR_BGR2GRAY);
    Mat gray1 = image1;
    if (image1.channels() > 1)
        cvtColor(image1, gray1, COLOR_BGR2GRAY);
    Mat black = Mat::zeros(Size(gray0.cols, gray0.rows), CV_8UC1);
    vector<Mat> redBlueImageVec = {gray0, black, gray1};
    Mat redBlueImage;
    merge(redBlueImageVec, redBlueImage);

    Mat flowImage = drawFlowImage(redBlueImage, features0, features1, radius, circleColor, thickness, lineColor);
    return flowImage;
}