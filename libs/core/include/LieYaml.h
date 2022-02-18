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

#pragma once

#include "yaml-cpp/yaml.h"

#include "liepp/SE3.h"

namespace YAML {

template <> struct convert<liepp::SE3d> {
    static Node encode(const liepp::SE3d& rhs) {
        const Eigen::Vector3d& position = rhs.x;
        const Eigen::Quaterniond& attitude = rhs.R.asQuaternion();
        Node node;
        node.push_back("xw");
        node.push_back(position.x());
        node.push_back(position.y());
        node.push_back(position.z());
        node.push_back(attitude.w());
        node.push_back(attitude.x());
        node.push_back(attitude.y());
        node.push_back(attitude.z());
        return node;
    }

    static bool decode(const Node& node, liepp::SE3d& rhs) {
        if (!node.IsSequence() || node.size() != 8) {
            return false;
        }
        Eigen::Vector3d position;
        Eigen::Quaterniond attitude;
        position.x() = node[1].as<double>();
        position.y() = node[2].as<double>();
        position.z() = node[3].as<double>();
        attitude.w() = node[4].as<double>();
        attitude.x() = node[5].as<double>();
        attitude.y() = node[6].as<double>();
        attitude.z() = node[7].as<double>();

        rhs.x = position;
        rhs.R = attitude.normalized();

        return true;
    }
};
} // namespace YAML