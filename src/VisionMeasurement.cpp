#include "eqvio/VisionMeasurement.h"
#include "eqvio/VIOState.h"

using namespace std;
using namespace Eigen;

std::vector<int> VisionMeasurement::getIds() const {
    std::vector<int> ids(camCoordinates.size());
    transform(camCoordinates.begin(), camCoordinates.end(), ids.begin(), [](const auto& cc) { return cc.first; });
    return ids;
}

std::map<int, cv::Point2f> VisionMeasurement::ocvCoordinates() const {
    std::map<int, cv::Point2f> ocvPoints;
    for (const auto& [id, pt] : camCoordinates) {
        ocvPoints[id] = cv::Point2f(pt.x(), pt.y());
    }
    return ocvPoints;
}

CSVLine& operator>>(CSVLine& line, VisionMeasurement& vision) {
    line >> vision.stamp;
    int numBearings;
    line >> numBearings;
    for (int i = 0; i < numBearings; ++i) {
        int id;
        Vector2d y;
        line >> id >> y;
        vision.camCoordinates[id] = y;
    }
    return line;
}

CSVLine& operator<<(CSVLine& line, const VisionMeasurement& vision) {
    line << vision.stamp;
    line << vision.camCoordinates.size();
    for (const pair<int, Vector2d>& cc : vision.camCoordinates) {
        line << cc.first << cc.second;
    }
    return line;
}

VisionMeasurement operator-(const VisionMeasurement& y1, const VisionMeasurement& y2) {
    VisionMeasurement yDiff;
    for (const pair<int, Vector2d>& cc1 : y1.camCoordinates) {
        const auto it2 = y2.camCoordinates.find(cc1.first);
        if (it2 != y2.camCoordinates.end()) {
            yDiff.camCoordinates[cc1.first] = cc1.second - it2->second;
        }
    }
    assert(y1.cameraPtr == y2.cameraPtr);
    yDiff.cameraPtr = y1.cameraPtr;
    return yDiff;
}
VisionMeasurement::operator Eigen::VectorXd() const {
    vector<int> ids = getIds();
    Eigen::VectorXd result = Eigen::VectorXd(2 * ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        result.segment<2>(2 * i) = camCoordinates.at(ids[i]);
    }
    return result;
}
