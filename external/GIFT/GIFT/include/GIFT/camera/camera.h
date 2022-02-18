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

#pragma once

#include "GIFT/camera/DoubleSphereCamera.h"
#include "GIFT/camera/EquidistantCamera.h"
#include "GIFT/camera/GICamera.h"
#include "GIFT/camera/PinholeCamera.h"
#include "GIFT/camera/StandardCamera.h"
#include "yaml-cpp/yaml.h"

namespace GIFT {
using GICameraPtr = std::shared_ptr<const GIFT::GICamera>;

template <class T> bool safeConfig(const YAML::Node& cfg, T& var) {
    if (cfg) {
        var = cfg.as<T>();
        return true;
    } else {
        return false;
    }
}

GICameraPtr readCamera(const YAML::Node& config);
}