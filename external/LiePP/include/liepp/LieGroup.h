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

#include "eigen3/Eigen/Dense"

namespace liepp {

#if __cpp_concepts >= 201907
template <typename G>
concept isLieGroup = requires {
    { G::CDim } -> std::same_as<const int&>; // G must have a const int labelled CDim
    { typename G::MatrixDS() } -> std::same_as<Eigen::Matrix<typename G::Scalar, G::CDim, G::CDim>>;
    { typename G::VectorDS() } -> std::same_as<Eigen::Matrix<typename G::Scalar, G::CDim, 1>>;

    // { G::wedge }
    // ->std::same_as<typename G::MatrixDS (&)(const typename G::VectorDS&)>;
    // { G::vee }
    // ->std::same_as<typename G::VectorDS (&)(const typename G::MatrixDS&)>;

    { G::exp } -> std::same_as<G (&)(const typename G::VectorDS&)>;
    { G::log } -> std::same_as<typename G::VectorDS (&)(const G&)>;

    { G::adjoint } -> std::same_as<typename G::MatrixDS (&)(const typename G::VectorDS&)>;
    { &G::Adjoint } -> std::same_as<typename G::MatrixDS (G::*)() const>;

    { static_cast<G (G::*)(const G&) const>(&G::operator*) } -> std::same_as<G (G::*)(const G&) const>;
    { G::Identity } -> std::same_as<G (&)()>;
    { &G::inverse } -> std::same_as<G (G::*)() const>;
};
#else
template <typename G> const bool isLieGroup = true;
#endif

} // namespace liepp