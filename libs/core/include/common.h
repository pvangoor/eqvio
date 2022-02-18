#pragma once

#include "yaml-cpp/yaml.h"
#include <iostream>

template <class T> bool safeConfig(const YAML::Node& cfg, T& var) {
    if (cfg) {
        var = cfg.as<T>();
        return true;
    } else {
        return false;
    }
}

template <class T> bool safeConfig(const YAML::Node& cfg, const std::string& itm, T& var) {
    std::stringstream ss(itm);
    std::string tmp;
    YAML::Node subNode = YAML::Clone(cfg);
    while (getline(ss, tmp, ':')) {
        if (subNode[tmp]) {
            subNode = subNode[tmp];
        } else {
            std::cout << itm << " was not found in the configuration." << std::endl;
            std::cout << "The following items were found in the configuration provided:\n";
            for (const auto& sn : cfg) {
                std::cout << sn.first.as<std::string>() << " ";
            }
            std::cout << std::endl;
            return false;
        }
    }
    var = subNode.as<T>();
    return true;
}

template <class T> T configOrDefault(const YAML::Node& cfg, const T& defaultValue) {
    if (cfg) {
        return cfg.as<T>();
    } else {
        return defaultValue;
    }
}

template <class T> T configOrDefault(const YAML::Node& cfg, const std::string& itm, const T& defaultValue) {
    T temp;
    if (safeConfig(cfg, itm, temp)) {
        return temp;
    } else {
        return defaultValue;
    }
}
