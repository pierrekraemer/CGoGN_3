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

#ifndef CGOGN_GEOMETRY_ALGOS_LAPLACIAN_H_
#define CGOGN_GEOMETRY_ALGOS_LAPLACIAN_H_

#include <cgogn/core/functions/traversals/vertex.h>

#include <cgogn/geometry/algos/area.h>
#include <cgogn/geometry/algos/length.h>
#include <cgogn/geometry/types/vector_traits.h>

#include <Eigen/Sparse>

namespace cgogn
{

struct MapBase;

namespace geometry
{

///////////////
// MapBase:2 //
///////////////

template <typename MESH, typename std::enable_if_t<std::is_convertible_v<MESH&, MapBase&> &&
												   (mesh_traits<MESH>::dimension == 2)>* = nullptr>
Scalar edge_cotan_weight(const MESH& m, typename MESH::Edge e,
						 const typename MESH::template Attribute<Vec3>* vertex_position)
{
	using Vertex = typename mesh_traits<MESH>::Vertex;

	Scalar result = 0.0;

	Dart d1 = e.dart_;
	Dart d2 = phi2(m, d1);

	const Vec3& p1 = value<Vec3>(m, vertex_position, Vertex(d1));
	const Vec3& p2 = value<Vec3>(m, vertex_position, Vertex(d2));

	const Vec3& p3 = value<Vec3>(m, vertex_position, Vertex(phi_1(m, d1)));
	Vec3 vecR = p1 - p3;
	Vec3 vecL = p2 - p3;
	Scalar e1value = vecR.dot(vecL) / vecR.cross(vecL).norm();

	result += e1value / 2.0;

	if (!is_boundary(m, d2))
	{
		const Vec3& p4 = value<Vec3>(m, vertex_position, Vertex(phi_1(m, d2)));
		Vec3 vecR = p2 - p4;
		Vec3 vecL = p1 - p4;
		Scalar e2value = vecR.dot(vecL) / vecR.cross(vecL).norm();

		result += e2value / 2.0;
	}

	return std::clamp(result, 0.0, 1.0);
}

/////////////
// GENERIC //
/////////////

template <typename MESH>
void compute_edge_cotan_weight(const MESH& m,
							   const typename mesh_traits<MESH>::template Attribute<Vec3>* vertex_position,
							   typename mesh_traits<MESH>::template Attribute<Scalar>* edge_weight)
{
	using Edge = typename mesh_traits<MESH>::Edge;
	parallel_foreach_cell(m, [&](Edge e) -> bool {
		value<Scalar>(m, edge_weight, e) = edge_cotan_weight(m, e, vertex_position);
		return true;
	});
}

template <typename MESH>
Eigen::SparseMatrix<Scalar, Eigen::ColMajor> cotan_operator_matrix(
	MESH& m, const typename mesh_traits<MESH>::template Attribute<uint32>* vertex_index,
	const typename mesh_traits<MESH>::template Attribute<Scalar>* edge_cotan_weight)
{
	static_assert(mesh_traits<MESH>::dimension == 2, "MESH dimension should be 2");

	using Vertex = typename mesh_traits<MESH>::Vertex;
	using Edge = typename mesh_traits<MESH>::Edge;

	uint32 nb_vertices = nb_cells<Vertex>(m);
	Eigen::SparseMatrix<Scalar, Eigen::ColMajor> COTAN(nb_vertices, nb_vertices);
	std::vector<Eigen::Triplet<Scalar>> COTANcoeffs;
	COTANcoeffs.reserve(nb_vertices * 10);
	foreach_cell(m, [&](Edge e) -> bool {
		Scalar w = value<Scalar>(m, edge_cotan_weight, e);
		auto vertices = incident_vertices(m, e);
		uint32 vidx1 = value<uint32>(m, vertex_index, vertices[0]);
		uint32 vidx2 = value<uint32>(m, vertex_index, vertices[1]);
		COTANcoeffs.push_back(Eigen::Triplet<Scalar>(int(vidx1), int(vidx2), w));
		COTANcoeffs.push_back(Eigen::Triplet<Scalar>(int(vidx2), int(vidx1), w));
		COTANcoeffs.push_back(Eigen::Triplet<Scalar>(int(vidx1), int(vidx1), -w));
		COTANcoeffs.push_back(Eigen::Triplet<Scalar>(int(vidx2), int(vidx2), -w));
		return true;
	});
	COTAN.setFromTriplets(COTANcoeffs.begin(), COTANcoeffs.end());

	return COTAN;
}

template <typename MESH>
Eigen::SparseMatrix<Scalar, Eigen::ColMajor> cotan_operator_matrix(
	MESH& m, const typename mesh_traits<MESH>::template Attribute<uint32>* vertex_index,
	const typename mesh_traits<MESH>::template Attribute<Vec3>* vertex_position)
{
	static_assert(mesh_traits<MESH>::dimension == 2, "MESH dimension should be 2");

	using Edge = typename mesh_traits<MESH>::Edge;

	auto edge_cotan_weight = add_attribute<Scalar, Edge>(m, "__edge_cotan_weight");
	compute_edge_cotan_weight(m, vertex_position, edge_cotan_weight.get());
	Eigen::SparseMatrix<Scalar, Eigen::ColMajor> COTAN =
		cotan_operator_matrix(m, vertex_index, edge_cotan_weight.get());
	remove_attribute<Edge>(m, edge_cotan_weight);
	return COTAN;
}

template <typename MESH>
Eigen::SparseMatrix<Scalar, Eigen::ColMajor> cotan_laplacian_matrix(
	MESH& m, const typename mesh_traits<MESH>::template Attribute<uint32>* vertex_index,
	const typename mesh_traits<MESH>::template Attribute<Scalar>* vertex_area,
	const typename mesh_traits<MESH>::template Attribute<Scalar>* edge_cotan_weight)
{
	static_assert(mesh_traits<MESH>::dimension == 2, "MESH dimension should be 2");

	using Vertex = typename mesh_traits<MESH>::Vertex;

	Eigen::SparseMatrix<Scalar, Eigen::ColMajor> LAPL = cotan_operator_matrix(m, vertex_index, edge_cotan_weight);

	Eigen::VectorXd A(LAPL.rows());
	parallel_foreach_cell(m, [&](Vertex v) -> bool {
		uint32 vidx = value<uint32>(m, vertex_index, v);
		A(vidx) = value<Scalar>(m, vertex_area, v);
		return true;
	});

	return A.asDiagonal().inverse() * LAPL;
}

template <typename MESH>
Eigen::SparseMatrix<Scalar, Eigen::ColMajor> cotan_laplacian_matrix(
	MESH& m, const typename mesh_traits<MESH>::template Attribute<uint32>* vertex_index,
	const typename mesh_traits<MESH>::template Attribute<Vec3>* vertex_position,
	const typename mesh_traits<MESH>::template Attribute<Scalar>* vertex_area)
{
	static_assert(mesh_traits<MESH>::dimension == 2, "MESH dimension should be 2");

	using Edge = typename mesh_traits<MESH>::Edge;

	auto edge_cotan_weight = add_attribute<Scalar, Edge>(m, "__edge_cotan_weight");
	compute_edge_cotan_weight(m, vertex_position, edge_cotan_weight.get());
	Eigen::SparseMatrix<Scalar, Eigen::ColMajor> LAPL =
		cotan_laplacian_matrix(m, vertex_index, vertex_area, edge_cotan_weight.get());
	remove_attribute<Edge>(m, edge_cotan_weight);
	return LAPL;
}

template <typename MESH>
Eigen::SparseMatrix<Scalar, Eigen::ColMajor> cotan_laplacian_matrix(
	MESH& m, const typename mesh_traits<MESH>::template Attribute<uint32>* vertex_index,
	const typename mesh_traits<MESH>::template Attribute<Vec3>* vertex_position)
{
	static_assert(mesh_traits<MESH>::dimension == 2, "MESH dimension should be 2");

	using Vertex = typename mesh_traits<MESH>::Vertex;

	auto vertex_area = add_attribute<Scalar, Vertex>(m, "__vertex_area");
	compute_area<Vertex>(m, vertex_position, vertex_area.get());
	Eigen::SparseMatrix<Scalar, Eigen::ColMajor> LAPL =
		cotan_laplacian_matrix(m, vertex_index, vertex_position, vertex_area.get());
	remove_attribute<Vertex>(m, vertex_area);
	return LAPL;
}

template <typename MESH>
Eigen::SparseMatrix<Scalar, Eigen::ColMajor> topo_laplacian_matrix(
	MESH& m, const typename mesh_traits<MESH>::template Attribute<uint32>* vertex_index)
{
	using Vertex = typename mesh_traits<MESH>::Vertex;
	using Edge = typename mesh_traits<MESH>::Edge;

	uint32 nb_vertices = nb_cells<Vertex>(m);
	Eigen::SparseMatrix<Scalar, Eigen::ColMajor> LAPL(nb_vertices, nb_vertices);
	std::vector<Eigen::Triplet<Scalar>> LAPLcoeffs;
	LAPLcoeffs.reserve(nb_vertices * 10);
	foreach_cell(m, [&](Edge e) -> bool {
		auto vertices = incident_vertices(m, e);
		uint32 vidx1 = value<uint32>(m, vertex_index, vertices[0]);
		uint32 vidx2 = value<uint32>(m, vertex_index, vertices[1]);
		LAPLcoeffs.push_back(Eigen::Triplet<Scalar>(int(vidx1), int(vidx2), 1));
		LAPLcoeffs.push_back(Eigen::Triplet<Scalar>(int(vidx2), int(vidx1), 1));
		LAPLcoeffs.push_back(Eigen::Triplet<Scalar>(int(vidx1), int(vidx1), -1));
		LAPLcoeffs.push_back(Eigen::Triplet<Scalar>(int(vidx2), int(vidx2), -1));
		return true;
	});
	LAPL.setFromTriplets(LAPLcoeffs.begin(), LAPLcoeffs.end());

	return LAPL;
}

///////////
// CMap2 //
///////////

inline Eigen::SparseMatrix<std::complex<Scalar>, Eigen::ColMajor> connection_cotan_operator_matrix(
	const CMap2& m, const CMap2::Attribute<uint32>* vertex_index,
	const CMap2::Attribute<Scalar>* halfedge_normalized_angle, const CMap2::Attribute<Vec3>* vertex_position)
{
	using Vertex = CMap2::Vertex;
	using HalfEdge = CMap2::HalfEdge;
	using Edge = CMap2::Edge;
	using Face = CMap2::Face;

	uint32 nb_vertices = nb_cells<Vertex>(m);
	Eigen::SparseMatrix<std::complex<Scalar>, Eigen::ColMajor> LAPL(nb_vertices, nb_vertices);
	std::vector<Eigen::Triplet<std::complex<Scalar>>> LAPLcoeffs;
	LAPLcoeffs.reserve(9 * nb_cells<Face>(m));

	foreach_cell(m, [&](Face f) -> bool {
		Dart d0 = f.dart_;
		Dart d1 = phi1(m, d0);
		Dart d2 = phi1(m, d1);

		Vertex vi{d0}, vj{d1}, vk{d2};

		uint32 i = value<uint32>(m, vertex_index, vi);
		uint32 j = value<uint32>(m, vertex_index, vj);
		uint32 k = value<uint32>(m, vertex_index, vk);

		const Vec3& pi = value<Vec3>(m, vertex_position, vi);
		const Vec3& pj = value<Vec3>(m, vertex_position, vj);
		const Vec3& pk = value<Vec3>(m, vertex_position, vk);

		Vec3 u_ij = pj - pi;
		Vec3 v_ik = pk - pi;
		Scalar a = u_ij.dot(v_ik) / u_ij.cross(v_ik).norm();

		Vec3 u_jk = pk - pj;
		Vec3 v_ji = pi - pj;
		Scalar b = u_jk.dot(v_ji) / u_jk.cross(v_ji).norm();

		Vec3 u_ki = pi - pk;
		Vec3 v_kj = pj - pk;
		Scalar c = u_ki.dot(v_kj) / u_ki.cross(v_kj).norm();

		Scalar angle_ij = value<Scalar>(m, halfedge_normalized_angle, HalfEdge(d0));
		Scalar angle_ji = value<Scalar>(m, halfedge_normalized_angle, HalfEdge(phi2(m, d0)));
		Scalar angle_jk = value<Scalar>(m, halfedge_normalized_angle, HalfEdge(d1));
		Scalar angle_kj = value<Scalar>(m, halfedge_normalized_angle, HalfEdge(phi2(m, d1)));
		Scalar angle_ki = value<Scalar>(m, halfedge_normalized_angle, HalfEdge(d2));
		Scalar angle_ik = value<Scalar>(m, halfedge_normalized_angle, HalfEdge(phi2(m, d2)));

		std::complex<Scalar> r_ij = std::polar(Scalar(1.0), (angle_ji + M_PI) - angle_ij);
		std::complex<Scalar> r_jk = std::polar(Scalar(1.0), (angle_kj + M_PI) - angle_jk);
		std::complex<Scalar> r_ik = std::polar(Scalar(1.0), (angle_ki + M_PI) - angle_ik);

		LAPLcoeffs.emplace_back(i, i, std::complex<Scalar>(-(b + c) / 2, 0));
		LAPLcoeffs.emplace_back(j, j, std::complex<Scalar>(-(c + a) / 2, 0));
		LAPLcoeffs.emplace_back(k, k, std::complex<Scalar>(-(a + b) / 2, 0));

		LAPLcoeffs.emplace_back(i, j, (c / 2) * r_ij);
		LAPLcoeffs.emplace_back(i, k, (b / 2) * r_ik);
		LAPLcoeffs.emplace_back(j, i, (c / 2) * std::conj(r_ij));
		LAPLcoeffs.emplace_back(j, k, (a / 2) * r_jk);
		LAPLcoeffs.emplace_back(k, i, (b / 2) * std::conj(r_ik));
		LAPLcoeffs.emplace_back(k, j, (a / 2) * std::conj(r_jk));

		return true;
	});

	LAPL.setFromTriplets(LAPLcoeffs.begin(), LAPLcoeffs.end());
	return LAPL;
}

} // namespace geometry

} // namespace cgogn

#endif // CGOGN_GEOMETRY_ALGOS_LAPLACIAN_H_
