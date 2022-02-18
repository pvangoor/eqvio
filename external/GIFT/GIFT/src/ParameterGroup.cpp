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

#include "GIFT/ParameterGroup.h"
#include "eigen3/unsupported/Eigen/MatrixFunctions"

using namespace Eigen;
using namespace std;
using namespace GIFT;

Matrix<ftype, 2, Dynamic> Affine2Group::actionJacobian(const Vector2T& point) const {
    Matrix<ftype, 2, Dynamic> jac(2, this->dim());
    jac << point.x(), point.y(), 0, 0, 1, 0, 0, 0, point.x(), point.y(), 0, 1;
    return jac;
}

void Affine2Group::applyStepOnRight(const VectorXT& step) {
    // Put the step in matrix form, exponentiate, and apply the result.
    Matrix3T stepMat = Matrix3T::Identity();
    stepMat(0, 0) = step(0);
    stepMat(0, 1) = step(1);
    stepMat(0, 2) = step(4);
    stepMat(1, 0) = step(2);
    stepMat(1, 1) = step(3);
    stepMat(1, 2) = step(5);

    stepMat = stepMat.exp();
    this->translation = this->translation + this->transformation * stepMat.block<2, 1>(0, 2);
    this->transformation = this->transformation * stepMat.block<2, 2>(0, 0);
}

Affine2Group Affine2Group::Identity() {
    Affine2Group identityElement;
    identityElement.transformation = Matrix2T::Identity();
    identityElement.translation = Vector2T::Zero();
    return identityElement;
}

Vector2T Affine2Group::applyLeftAction(const Vector2T& point) const {
    Vector2T transformedPoint = this->transformation * point + this->translation;
    return transformedPoint;
}

void Affine2Group::changeLevel(const int& newLevel) {
    const int levelDiff = newLevel - level;
    this->level = newLevel;
    this->translation = pow(2, -levelDiff) * this->translation;
}

Matrix<ftype, 2, Dynamic> TranslationGroup::actionJacobian(const Vector2T& point) const { return Matrix2T::Identity(); }

Vector2T TranslationGroup::applyLeftAction(const Vector2T& point) const { return point + translation; }

void TranslationGroup::applyStepOnRight(const VectorXT& step) { this->translation += step; }

void TranslationGroup::changeLevel(const int& newLevel) {
    const int levelDiff = newLevel - level;
    this->level = newLevel;
    this->translation = pow(2, -levelDiff) * this->translation;
}

TranslationGroup TranslationGroup::Identity() {
    TranslationGroup newElem;
    newElem.translation.setZero();
    return newElem;
}