/*
    This file is part of LiePP.

    LiePP is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    LiePP is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with LiePP.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once

#include "SO3.h"

namespace liepp {

template <typename _Scalar = double> class SE3 {
  public:
    using Scalar = _Scalar;
    constexpr static int CDim = 6;
    using Vector3S = Eigen::Matrix<_Scalar, 3, 1>;
    using Matrix3S = Eigen::Matrix<_Scalar, 3, 3>;
    using MatrixNS = Eigen::Matrix<_Scalar, 4, 4>;
    using VectorDS = Eigen::Matrix<_Scalar, 6, 1>;
    using MatrixDS = Eigen::Matrix<_Scalar, 6, 6>;
    using SO3S = SO3<_Scalar>;

    static MatrixNS wedge(const VectorDS& u) {
        // u is in the format (omega, v)
        MatrixNS result;
        result.template block<3, 3>(0, 0) = SO3S::skew(u.template block<3, 1>(0, 0));
        result.template block<3, 1>(0, 3) = u.template block<3, 1>(3, 0);
        result.template block<1, 4>(3, 0) = Eigen::Matrix<_Scalar, 1, 4>::Zero();
        return result;
    }
    static VectorDS vee(const MatrixNS& U) {
        // u is in the format (omega, v)
        VectorDS result;
        result.template block<3, 1>(0, 0) = SO3S::vex(U.template block<3, 3>(0, 0));
        result.template block<3, 1>(3, 0) = U.template block<3, 1>(0, 3);
        return result;
    }
    static MatrixDS adjoint(const VectorDS& u) {
        // u is in the format (omega, v)
        MatrixDS result = MatrixDS::Zero();
        result.template block<3, 3>(0, 0) = SO3S::skew(u.template segment<3>(0));
        result.template block<3, 3>(3, 3) = SO3S::skew(u.template segment<3>(0));
        result.template block<3, 3>(3, 0) = SO3S::skew(u.template segment<3>(3));
        return result;
    }
    static SE3 exp(const VectorDS& u) {
        Vector3S w = u.template block<3, 1>(0, 0);
        Vector3S v = u.template block<3, 1>(3, 0);

        _Scalar th = w.norm();
        _Scalar A, B, C;
        if (abs(th) >= 1e-12) {
            A = sin(th) / th;
            B = (1 - cos(th)) / pow(th, 2);
            C = (1 - A) / pow(th, 2);
        } else {
            A = 1.0;
            B = 1.0 / 2.0;
            C = 1.0 / 6.0;
        }

        Matrix3S wx = SO3S::skew(w);
        Matrix3S R = Matrix3S::Identity() + A * wx + B * wx * wx;
        Matrix3S V = Matrix3S::Identity() + B * wx + C * wx * wx;

        MatrixNS expMat = MatrixNS::Identity();
        expMat.template block<3, 3>(0, 0) = R;
        expMat.template block<3, 1>(0, 3) = V * v;

        return SE3(expMat);
    }
    static VectorDS log(const SE3& P) {
        SO3S R = P.R;
        Vector3S x = P.x;

        Matrix3S Omega = SO3S::skew(SO3S::log(R));

        _Scalar theta = SO3S::vex(Omega).norm();
        _Scalar coefficient = 1.0 / 12.0;
        if (abs(theta) > 1e-6) {
            coefficient = 1 / (theta * theta) * (1 - (theta * sin(theta)) / (2 * (1 - cos(theta))));
        }

        Matrix3S VInv = Matrix3S::Identity() - 0.5 * Omega + coefficient * Omega * Omega;
        Vector3S v = VInv * x;

        MatrixNS U = MatrixNS::Zero();
        U.template block<3, 3>(0, 0) = Omega;
        U.template block<3, 1>(0, 3) = v;

        return SE3::vee(U);
    }
    static SE3 Identity() { return SE3(SO3S::Identity(), Vector3S::Zero()); }
    static SE3 Random() { return SE3(SO3S::Random(), Vector3S::Random()); }

    SE3() = default;
    SE3(const SE3& other) = default;
    SE3(const MatrixNS& mat) {
        R = SO3S(mat.template block<3, 3>(0, 0));
        x = mat.template block<3, 1>(0, 3);
    }
    SE3(const SO3S& R, const Vector3S& x) {
        this->R = R;
        this->x = x;
    }

    void setIdentity() {
        R.setIdentity();
        x.setZero();
    }
    Vector3S operator*(const Vector3S& point) const { return R * point + x; }
    SE3 operator*(const SE3& other) const { return SE3(R * other.R, x + R * other.x); }

    void invert() {
        x = -R.inverse() * x;
        R = R.inverse();
    }
    SE3 inverse() const { return SE3(R.inverse(), -(R.inverse() * x)); }
    MatrixDS Adjoint() const {
        MatrixDS AdMat;
        Matrix3S Rmat = R.asMatrix();
        AdMat.template block<3, 3>(0, 0) = Rmat;
        AdMat.template block<3, 3>(0, 3) = Matrix3S::Zero();
        AdMat.template block<3, 3>(3, 0) = SO3S::skew(x) * Rmat;
        AdMat.template block<3, 3>(3, 3) = Rmat;
        return AdMat;
    }

    // Set and get
    MatrixNS asMatrix() const {
        MatrixNS result;
        result.setIdentity();
        result.template block<3, 3>(0, 0) = R.asMatrix();
        result.template block<3, 1>(0, 3) = x;
        return result;
    }
    void fromMatrix(const MatrixNS& mat) {
        R.fromMatrix(mat.template block<3, 3>(0, 0));
        x = mat.template block<3, 1>(0, 3);
    }

    SO3S R;
    Vector3S x;

    static_assert(isLieGroup<SE3<_Scalar>>);
};

using SE3d = SE3<double>;
using SE3f = SE3<float>;
using SE3cd = SE3<Eigen::dcomplex>;
using SE3cf = SE3<Eigen::scomplex>;

using se3d = Eigen::Matrix<double, 6, 1>;
using se3f = Eigen::Matrix<float, 6, 1>;
using se3cd = Eigen::Matrix<Eigen::dcomplex, 6, 1>;
using se3cf = Eigen::Matrix<Eigen::scomplex, 6, 1>;

}