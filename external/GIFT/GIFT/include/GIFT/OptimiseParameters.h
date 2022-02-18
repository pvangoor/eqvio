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

#include "GIFT/ImagePyramid.h"
#include "opencv2/core/core.hpp"

namespace GIFT {

class ParameterGroup;

void optimiseParameters(ParameterGroup& params, const PyramidPatch& patch, const ImagePyramid& pyramid);
void optimiseParameters(
    std::vector<ParameterGroup>& params, const std::vector<PyramidPatch>& patches, const ImagePyramid& pyramid);
void optimiseParameters(
    std::vector<ParameterGroup>& params, const std::vector<PyramidPatch>& patches, const cv::Mat& image);
int clamp(const int x, const int a, const int b);

} // namespace GIFT