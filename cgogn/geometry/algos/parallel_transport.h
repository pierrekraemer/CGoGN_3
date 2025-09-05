/*******************************************************************************
 * CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional Maps  *
 * Copyright (C), IGG Group, ICube, University of Strasbourg, France            *
 *                                                                              *
 * This library is free software; you can redistribute it and/or modify it      *
 * under the terms of the GNU Lesser General Public License as published by the *
 * Free Software Foundation; either version 2.1 of the License, or (at your     *
 * option) any later version.                                                   *
 *                                                                              *
 * This library is distributed in the hope that it will be useful, but WITHOUT  *
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
 * for more details.                                                            *
 *                                                                              *
 * You should have received a copy of the GNU Lesser General Public License     *
 * along with this library; if not, write to the Free Software Foundation,      *
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
 *                                                                              *
 * Web site: http://cgogn.unistra.fr/                                           *
 * Contact information: cgogn@unistra.fr                                        *
 *                                                                              *
 *******************************************************************************/

#ifndef CGOGN_GEOMETRY_ALGOS_PARALLEL_TRANSPORT_H_
#define CGOGN_GEOMETRY_ALGOS_PARALLEL_TRANSPORT_H_

#include <cgogn/core/functions/traversals/halfedge.h>

#include <cgogn/geometry/algos/angle.h>
#include <cgogn/geometry/algos/laplacian.h>

#include <cgogn/geometry/types/vector_traits.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace cgogn
{

namespace geometry
{

using Complex = std::complex<Scalar>;

template <typename MESH>
void compute_vector_heat_transport(
	MESH& m, const typename mesh_traits<MESH>::template Attribute<Vec3>* vertex_position,
	const typename mesh_traits<MESH>::template Attribute<Scalar>* halfedge_normalized_angle,
	const CellsSet<MESH, typename mesh_traits<MESH>::Vertex>* source_vertices,
	typename mesh_traits<MESH>::template Attribute<Complex>* vertex_tangent_vector, Scalar t_multiplier = 1.0)
{
	using Vertex = typename mesh_traits<MESH>::Vertex;
	using HalfEdge = typename mesh_traits<MESH>::HalfEdge;

	auto vertex_index = get_or_add_attribute<uint32, Vertex>(m, "__vertex_index");

	uint32 nb_vertices = 0;
	foreach_cell(m, [&](Vertex v) -> bool {
		value<uint32>(m, vertex_index, v) = nb_vertices;
		nb_vertices++;
		return true;
	});

	auto vertex_area = get_or_add_attribute<Scalar, Vertex>(m, "__vertex_area");
	compute_area<Vertex>(m, vertex_position, vertex_area.get(), VertexAreaPolicy::THIRD);

	// connection cotangent operator
	Eigen::SparseMatrix<Complex> connLc =
		connection_cotan_operator_matrix(m, vertex_index.get(), halfedge_normalized_angle, vertex_position);

	// cotangent operator
	Eigen::SparseMatrix<Scalar> Lc = cotan_operator_matrix(m, vertex_index.get(), vertex_position);

	Eigen::VectorXd A(nb_vertices);
	Eigen::VectorXcd cA(nb_vertices);
	parallel_foreach_cell(m, [&](Vertex v) -> bool {
		uint32 vidx = value<uint32>(m, vertex_index, v);
		Scalar a = value<Scalar>(m, vertex_area, v);
		A(vidx) = a;
		cA(vidx) = Complex(a, 0.0);
		return true;
	});

	Eigen::VectorXcd Y0(nb_vertices);
	Eigen::VectorXd u0(nb_vertices);
	Eigen::VectorXd phi0(nb_vertices);
	Y0.setZero();
	u0.setZero();
	phi0.setZero();
	source_vertices->foreach_cell([&](Vertex v) {
		uint32 vidx = value<uint32>(m, vertex_index, v);
		Y0(vidx) = value<Complex>(m, vertex_tangent_vector, v);
		u0(vidx) = std::abs(Y0(vidx));
		phi0(vidx) = 1.0;
	});

	Scalar h = mean_edge_length(m, vertex_position);
	Scalar t = h * h * t_multiplier;

	Eigen::SparseMatrix<Complex, Eigen::ColMajor> cAm(cA.asDiagonal());
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<Complex, Eigen::ColMajor>> vector_solver(cAm - t * connLc);
	Eigen::VectorXcd Y = vector_solver.solve(Y0);

	Eigen::SparseMatrix<Scalar, Eigen::ColMajor> Am(A.asDiagonal());
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar, Eigen::ColMajor>> heat_solver(Am - t * Lc);
	Eigen::VectorXd u = heat_solver.solve(u0);
	Eigen::VectorXd phi = heat_solver.solve(phi0);

	parallel_foreach_cell(m, [&](Vertex v) -> bool {
		uint32 vidx = value<uint32>(m, vertex_index, v);
		Scalar normY = std::abs(Y(vidx));
		Scalar mag = u(vidx) / phi(vidx);
		value<Complex>(m, vertex_tangent_vector, v) = mag * Y(vidx) / normY;
		return true;
	});

	remove_attribute<Vertex>(m, vertex_index);
	remove_attribute<Vertex>(m, vertex_area);
}

} // namespace geometry

} // namespace cgogn

#endif // CGOGN_GEOMETRY_ALGOS_PARALLEL_TRANSPORT_H_
