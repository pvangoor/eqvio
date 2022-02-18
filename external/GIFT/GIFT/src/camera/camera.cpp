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

#include "GIFT/camera/camera.h"
using namespace GIFT;

GICameraPtr readCamera(const YAML::Node& config) {
    GICameraPtr camPtr;

    std::vector<ftype> intrinsics, distortion, resolution;
    if (!safeConfig(config["intrinsics"], intrinsics)) {
        throw std::runtime_error("Could not read 'intrinsics'.");
    }
    if (!safeConfig(config["distortion_coeffs"], distortion)) {
        throw std::runtime_error("Could not read 'distortion_coeffs'.");
    }
    if (!safeConfig(config["resolution"], resolution)) {
        throw std::runtime_error("Could not read 'resolution'.");
    }

    cv::Size sze(resolution[0], resolution[1]);
    const cv::Mat K =
        (cv::Mat_<ftype>(3, 3) << intrinsics[0], 0, intrinsics[2], 0, intrinsics[1], intrinsics[3], 0, 0, 1);

    if (config["distortion_model"]) {
        const std::string distortionModel = config["distortion_model"].as<std::string>();
        if (distortionModel == "radtan") {
            camPtr = std::make_shared<StandardCamera>(sze, K, distortion);

        } else if (distortionModel == "equidistant") {
            std::array<ftype, 4> distortionArray;
            std::copy(distortion.begin(), distortion.begin() + 4, distortionArray.begin());
            camPtr = std::make_shared<EquidistantCamera>(sze, K, distortionArray);

        } else if (distortionModel == "doublesphere") {
            intrinsics.insert(intrinsics.end(), distortion.begin(), distortion.end());
            std::array<ftype, 6> dsIntrinsicsArray;
            std::copy(intrinsics.begin(), intrinsics.begin() + 6, dsIntrinsicsArray.begin());
            camPtr = std::make_shared<DoubleSphereCamera>(dsIntrinsicsArray, sze);
        }
    } else {
        if (distortion.empty()) {
            camPtr = std::make_shared<PinholeCamera>(sze, K);
        } else {
            camPtr = std::make_shared<StandardCamera>(sze, K, distortion);
        }
    }

    return camPtr;
}