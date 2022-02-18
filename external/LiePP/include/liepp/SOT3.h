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
#include "eigen3/Eigen/Dense"

namespace liepp {

template <typename _Scalar = double> class SOT3 {
  public:
    using Scalar = _Scalar;
    constexpr static int CDim = 4;
    using Vector3S = Eigen::Matrix<_Scalar, 3, 1>;
    using Matrix3S = Eigen::Matrix<_Scalar, 3, 3>;
    using MatrixNS = Eigen::Matrix<_Scalar, 4, 4>;
    using VectorDS = Eigen::Matrix<_Scalar, 4, 1>;
    using MatrixDS = Eigen::Matrix<_Scalar, 4, 4>;
    using SO3S = SO3<_Scalar>;

    static MatrixNS wedge(const VectorDS& u) {
        MatrixNS U = MatrixNS::Zero();
        U.template block<3, 3>(0, 0) = SO3S::skew(u.template segment<3>(0));
        U(3, 3) = u(3);
        return U;
    }
    static VectorDS vee(const MatrixNS& U) {
        VectorDS u;
        u.template segment<3>(0) = SO3S::vex(U.template block<3, 3>(0, 0));
        u(3) = U(3, 3);
        return u;
    }
    static SOT3 exp(const VectorDS& w) {
        SOT3 result;
        result.R = SO3S::exp(w.template block<3, 1>(0, 0));
        result.a = std::exp(w(3));
        return result;
    }
    static VectorDS log(const SOT3& T) {
        VectorDS result;
        result.template block<3, 1>(0, 0) = SO3S::log(T.R);
        result(3) = std::log(T.a);
        return result;
    }

    static MatrixNS adjoint(const VectorDS& u) {
        MatrixNS ad_u = MatrixNS::Zero();
        ad_u.template block<3, 3>(0, 0) = SO3S::skew(u.template segment<3>(0));
        return ad_u;
    }

    MatrixNS Adjoint() const {
        MatrixNS AdMatrix = MatrixNS::Identity();
        AdMatrix.template block<3, 3>(0, 0) = this->R.asMatrix();
        return AdMatrix;
    }

    static SOT3 Identity() {
        SOT3 Q;
        Q.setIdentity();
        return Q;
    }
    static SOT3 Random() { return SOT3(SO3S::Random(), std::exp(_Scalar(rand() / RAND_MAX))); }

    SOT3() = default;
    SOT3(const SOT3& other) = default;
    SOT3(const SO3S& R, const _Scalar& a) {
        this->R = R;
        this->a = a;
    }
    SOT3(const MatrixNS& mat) {
        R.fromMatrix(mat.template block<3, 3>(0, 0));
        a = mat(3, 3);
    }

    void setIdentity() {
        R.setIdentity();
        a = 1.0;
    }
    Vector3S operator*(const Vector3S& point) const { return a * (R * point); }
    SOT3 operator*(const SOT3& other) const { return SOT3(R * other.R, a * other.a); }
    Vector3S applyInverse(const Vector3S& p) const { return 1.0 / a * R.applyInverse(p); }

    void invert() {
        R.invert();
        a = 1.0 / a;
    }
    SOT3 inverse() const { return SOT3(R.inverse(), 1.0 / a); }

    // Set and get
    MatrixNS asMatrix() const {
        MatrixNS result = MatrixNS::Identity();
        result.template block<3, 3>(0, 0) = R.asMatrix();
        result(3, 3) = a;
        return result;
    }
    Matrix3S asMatrix3() const {
        Matrix3S result = a * R.asMatrix();
        return result;
    }

    SO3S R;
    _Scalar a;
    static_assert(isLieGroup<SOT3<_Scalar>>);
};

using SOT3d = SOT3<double>;
using SOT3f = SOT3<float>;
using SOT3cd = SOT3<Eigen::dcomplex>;
using SOT3cf = SOT3<Eigen::scomplex>;

}