#pragma once
/*
  allow for selection of single versus double precision
 */

#include <eigen3/Eigen/Eigen>

#if defined(USE_SINGLE_FLOAT)

// use single precision
typedef float ftype;
namespace Eigen {
typedef Eigen::Vector2f Vector2T;
typedef Eigen::Vector3f Vector3T;
typedef Eigen::Vector4f Vector4T;
typedef Eigen::VectorXf VectorXT;
typedef Eigen::Matrix2f Matrix2T;
typedef Eigen::Matrix3f Matrix3T;
typedef Eigen::Matrix4f Matrix4T;
typedef Eigen::MatrixXf MatrixXT;
} // namespace Eigen

#else

// use double precision
typedef double ftype;
namespace Eigen {
typedef Eigen::Vector2d Vector2T;
typedef Eigen::Vector3d Vector3T;
typedef Eigen::Vector4d Vector4T;
typedef Eigen::VectorXd VectorXT;
typedef Eigen::Matrix2d Matrix2T;
typedef Eigen::Matrix3d Matrix3T;
typedef Eigen::Matrix4d Matrix4T;
typedef Eigen::MatrixXd MatrixXT;
} // namespace Eigen

#endif
