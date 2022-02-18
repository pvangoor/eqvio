# General Invariant Feature Tracker (GIFT)

GIFT is an image feature tracking library for monocular cameras.
The goal is to provide a package which simplifies the process of detecting and tracking features from a sequence of images.
A number of variants have been implemented including LKT-based tracking and ORB feature tracking.
The library aims to provide a generic interface to each of these methods.

## Dependencies

Currently the GIFT depends on both Eigen 3 and OpenCV 3.
In the future the dependency on Eigen may be removed.

- Eigen3: `sudo apt install libeigen3-dev`
- OpenCV: `sudo apt install libopencv-dev`
- yaml-cpp: `sudo apt install libyaml-cpp-dev`

If you choose to build the tests, [googletest](https://github.com/google/googletest) will be automatically added to the build directory.

## Building and Installing

GIFT can be built and installed using cmake and make.

```bash
git clone https://github.com/pvangoor/GIFT
cd GIFT
mkdir build
cd build
cmake ..
sudo make install
```

## Citing

GIFT was developed for use in an academic paper.
If you use GIFT in an academic context, please cite the following publication:

van Goor, Pieter, et al. "A Geometric Observer Design for Visual Localisation and Mapping." 2019 IEEE 58th Conference on Decision and Control (CDC). IEEE, 2019.
