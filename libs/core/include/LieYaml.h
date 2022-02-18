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