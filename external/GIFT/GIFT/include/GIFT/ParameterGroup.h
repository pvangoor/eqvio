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

#include "GIFT/ftype.h"
#include "eigen3/Eigen/Core"

namespace GIFT {

class ParameterGroup {
  public:
    virtual int dim() const = 0;
    virtual Eigen::Matrix<ftype, 2, Eigen::Dynamic> actionJacobian(const Eigen::Vector2T& point) const = 0;
    virtual void applyStepOnRight(const Eigen::VectorXT& step) = 0;
    virtual Eigen::Vector2T applyLeftAction(const Eigen::Vector2T& point) const = 0;
    virtual void changeLevel(const int& newLevel) = 0;
    int level = 0;
};

class Affine2Group : public ParameterGroup {
  public:
    int dim() const { return 6; }

    Eigen::Matrix2T transformation;
    Eigen::Vector2T translation;

    Eigen::Matrix<ftype, 2, Eigen::Dynamic> actionJacobian(const Eigen::Vector2T& point) const;
    Eigen::Vector2T applyLeftAction(const Eigen::Vector2T& point) const;
    void applyStepOnRight(const Eigen::VectorXT& step);
    void changeLevel(const int& newLevel);

    static Affine2Group Identity();
};

class TranslationGroup : public ParameterGroup {
  public:
    int dim() const { return 2; }

    Eigen::Vector2T translation;

    Eigen::Matrix<ftype, 2, Eigen::Dynamic> actionJacobian(const Eigen::Vector2T& point) const;
    Eigen::Vector2T applyLeftAction(const Eigen::Vector2T& point) const;
    void applyStepOnRight(const Eigen::VectorXT& step);
    void changeLevel(const int& newLevel);

    static TranslationGroup Identity();
};

} // namespace GIFT