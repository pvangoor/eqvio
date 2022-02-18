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

#include "LieGroup.h"

namespace liepp {

template <typename _Scalar = double> class SO3 {
  public:
    using Scalar = _Scalar;
    constexpr static int CDim = 3;
    using VectorDS = Eigen::Matrix<_Scalar, 3, 1>;
    using MatrixDS = Eigen::Matrix<_Scalar, 3, 3>;
    using MatrixNS = Eigen::Matrix<_Scalar, 3, 3>;
    using QuaternionS = Eigen::Quaternion<_Scalar>;

    static MatrixDS skew(const VectorDS& v) {
        return (MatrixDS() << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0).finished();
    }
    static MatrixDS wedge(const VectorDS& v) { return skew(v); }

    static VectorDS vex(const MatrixDS& M) { return (VectorDS() << M(2, 1), M(0, 2), M(1, 0)).finished(); }
    static VectorDS vee(const MatrixDS& M) { return vex(M); }

    static MatrixDS adjoint(const VectorDS& Omega) { return skew(Omega); }

    static SO3 exp(const VectorDS& w) {
        _Scalar theta = w.norm() / 2.0;
        QuaternionS result;
        result.w() = cos(theta);
        result.vec() = sin(theta) * w.normalized();
        return SO3(result);
    }

    static VectorDS log(const SO3& rotation) {
        MatrixDS R = rotation.asMatrix();
        _Scalar theta = acos((R.trace() - 1.0) / 2.0);
        _Scalar coefficient = (abs(theta) >= 1e-6) ? theta / (2.0 * sin(theta)) : 0.5;

        MatrixDS Omega = coefficient * (R - R.transpose());
        return vex(Omega);
    }

    static SO3 SO3FromVectors(const VectorDS& origin, const VectorDS& dest) {
        SO3 result;
        result.quaternion.setFromTwoVectors(origin, dest);
        return result;
    }

    static SO3 Identity() { return SO3(QuaternionS::Identity()); }
    static SO3 Random() { return SO3(QuaternionS::UnitRandom()); }

    SO3() = default;
    SO3(const MatrixDS& mat) { quaternion = mat; }
    SO3(const QuaternionS& quat) { quaternion = quat; }
    SO3 inverse() const { return SO3(quaternion.inverse()); }
    MatrixDS Adjoint() const { return this->asMatrix(); }

    void setIdentity() { quaternion = QuaternionS::Identity(); }
    VectorDS operator*(const VectorDS& point) const { return quaternion * point; }
    SO3 operator*(const SO3& other) const { return SO3(quaternion * other.quaternion); }
    VectorDS applyInverse(const VectorDS& point) const { return quaternion.inverse() * point; }

    void invert() { quaternion = quaternion.inverse(); }

    // Set and get
    MatrixDS asMatrix() const { return quaternion.toRotationMatrix(); }
    QuaternionS asQuaternion() const { return quaternion; }
    void fromMatrix(const MatrixDS& mat) { quaternion = mat; }
    void fromQuaternion(const QuaternionS& quat) { quaternion = quat; }

    QuaternionS quaternion;
    static_assert(isLieGroup<SO3<_Scalar>>);
};

using SO3d = SO3<double>;
using SO3f = SO3<float>;
// using SO3cd = SO3<Eigen::dcomplex>;
// using SO3cf = SO3<Eigen::scomplex>;

}