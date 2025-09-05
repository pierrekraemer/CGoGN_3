/*******************************************************************************
 * CGoGN                                                                        *
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

#ifndef CGOGN_MODULE_SURFACE_VECTOR_HEAT_METHOD_H_
#define CGOGN_MODULE_SURFACE_VECTOR_HEAT_METHOD_H_

#include <cgogn/ui/app.h>
#include <cgogn/ui/module.h>

#include <cgogn/core/ui_modules/mesh_provider.h>

#include <cgogn/geometry/algos/angle.h>
#include <cgogn/geometry/algos/parallel_transport.h>

// #include <chrono>
// #include <cmath>
// #include <iostream>
// #include <limits>
// #include <queue>
// #include <unordered_set>
// #include <utility>

namespace cgogn
{

namespace ui
{

using Vec3 = geometry::Vec3;
using Scalar = geometry::Scalar;
using Complex = std::complex<Scalar>;

template <typename MESH>
class SurfaceVectorHeatMethod : public Module
{
	static_assert(mesh_traits<MESH>::dimension == 2,
				  "SurfaceVectorHeatMethod can only be used with meshes of dimension 2");

	template <typename T>
	using Attribute = typename mesh_traits<MESH>::template Attribute<T>;

	using HalfEdge = typename mesh_traits<MESH>::HalfEdge;
	using Vertex = typename mesh_traits<MESH>::Vertex;
	using Edge = typename mesh_traits<MESH>::Edge;
	using Face = typename mesh_traits<MESH>::Face;

public:
	SurfaceVectorHeatMethod(const App& app)
		: Module(app, "SurfaceVectorHeatMethod (" + std::string{mesh_traits<MESH>::name} + ")")
	{
	}

	~SurfaceVectorHeatMethod()
	{
	}

	void diffuse_tangent_vector(MESH& m, const Attribute<Vec3>* vertex_position, const Attribute<Vec3>* vertex_normal,
								const CellsSet<MESH, Vertex>* source_vertices,
								Attribute<Complex>* vertex_tangent_vector, double t_multiplier)
	{
		// choose a reference (half)edge for each vertex
		auto vertex_ref_he = get_or_add_attribute<HalfEdge, Vertex>(m, "ref_he");
		parallel_foreach_cell(m, [&](Vertex v) -> bool {
			std::vector<HalfEdge> halfedges = incident_halfedges(m, v);
			value<HalfEdge>(m, vertex_ref_he, v) = halfedges[0]; // choose the first halfedge as reference
			return true;
		});

		// compute normalized angles around each vertex
		auto halfedge_normalized_angle = get_or_add_attribute<Scalar, HalfEdge>(m, "normalized_angle");
		geometry::compute_vertex_normalized_angles(m, vertex_position, vertex_ref_he.get(),
												   halfedge_normalized_angle.get());

		// fix source vertices direction to that of their reference halfedge
		source_vertices->foreach_cell(
			[&](Vertex v) { value<Complex>(m, vertex_tangent_vector, v) = std::polar(1.0, 0.0); });

		geometry::compute_vector_heat_transport(m, vertex_position, halfedge_normalized_angle.get(), source_vertices,
												vertex_tangent_vector, t_multiplier);

		auto vertex_tangent_vector_3 = get_or_add_attribute<Vec3, Vertex>(m, "tangent_vector_vis");

		compute_vector_from_local_direction(m, vertex_position, vertex_normal, vertex_ref_he.get(),
											vertex_tangent_vector, vertex_tangent_vector_3.get());

		mesh_provider_->emit_attribute_changed(m, vertex_tangent_vector_3.get());
	}

protected:
	void init() override
	{
		mesh_provider_ = static_cast<ui::MeshProvider<MESH>*>(
			app_.module("MeshProvider (" + std::string{mesh_traits<MESH>::name} + ")"));
	}

	void compute_vector_from_local_direction(const MESH& m, const Attribute<Vec3>* vertex_position,
											 const Attribute<Vec3>* vertex_normal,
											 const Attribute<HalfEdge>* vertex_ref_he,
											 const Attribute<Complex>* vertex_tangent_vector,
											 Attribute<Vec3>* vertex_tangent_vector_3)
	{
		parallel_foreach_cell(m, [&](Vertex v) -> bool {
			const Complex z = value<Complex>(m, vertex_tangent_vector, v);
			Scalar r = std::abs(z);
			Scalar th = std::atan2(z.imag(), z.real());

			const Vec3& p = value<Vec3>(m, vertex_position, v);

			Dart d_ref = value<HalfEdge>(m, vertex_ref_he, v).dart_;
			const Vec3& p_next = value<Vec3>(m, vertex_position, Vertex(phi1(m, d_ref)));
			Vec3 e1 = (p_next - p).normalized();

			Vec3 n = value<Vec3>(m, vertex_normal, v).normalized();
			Vec3 e2 = e1.cross(n).normalized();
			Vec3 dir = r * (std::cos(th) * e1 + std::sin(th) * e2);

			value<Vec3>(m, vertex_tangent_vector_3, v) = dir;

			return true;
		});
	}

	void left_panel() override
	{
		imgui_mesh_selector(mesh_provider_, selected_mesh_, "Surface", [&](MESH& m) {
			selected_mesh_ = &m;
			selected_vertex_position_.reset();
			mesh_provider_->mesh_data(m).outlined_until_ = App::frame_time_ + 1.0;
		});

		if (selected_mesh_)
		{
			MeshData<MESH>& md = mesh_provider_->mesh_data(*selected_mesh_);

			imgui_combo_attribute<Vertex, Vec3>(
				*selected_mesh_, selected_vertex_position_, "Vertex Position",
				[&](const std::shared_ptr<Attribute<Vec3>>& attribute) { selected_vertex_position_ = attribute; });

			imgui_combo_attribute<Vertex, Vec3>(
				*selected_mesh_, selected_vertex_normal_, "Vertex Normal",
				[&](const std::shared_ptr<Attribute<Vec3>>& attribute) { selected_vertex_normal_ = attribute; });

			imgui_combo_attribute<Vertex, Complex>(*selected_mesh_, selected_vertex_tangent_vector_, "Vector field",
												   [&](const std::shared_ptr<Attribute<Complex>>& attribute) {
													   selected_vertex_tangent_vector_ = attribute;
												   });

			imgui_combo_cells_set(md, selected_vertices_set_, "Source vertices",
								  [&](CellsSet<MESH, Vertex>* cs) { selected_vertices_set_ = cs; });

			if (selected_vertex_position_ && selected_vertex_normal_ && selected_vertices_set_)
			{
				static double t_multiplier = 1.0;
				ImGui::InputDouble("Scalar t_multiplier", &t_multiplier, 0.01f, 100.0f, "%.3f");

				if (ImGui::Button("Diffuse tangent vector"))
				{
					if (!selected_vertex_tangent_vector_)
						selected_vertex_tangent_vector_ =
							get_or_add_attribute<Complex, Vertex>(*selected_mesh_, "tangent vector");
					diffuse_tangent_vector(*selected_mesh_, selected_vertex_position_.get(),
										   selected_vertex_normal_.get(), selected_vertices_set_,
										   selected_vertex_tangent_vector_.get(), t_multiplier);
				}
			}
		}
	}

private:
	MESH* selected_mesh_ = nullptr;

	std::shared_ptr<Attribute<Vec3>> selected_vertex_position_ = nullptr;
	std::shared_ptr<Attribute<Vec3>> selected_vertex_normal_ = nullptr;
	std::shared_ptr<Attribute<Complex>> selected_vertex_tangent_vector_ = nullptr;
	CellsSet<MESH, Vertex>* selected_vertices_set_ = nullptr;

	MeshProvider<MESH>* mesh_provider_ = nullptr;

	std::shared_ptr<Attribute<Scalar>> geodesic_distance_vertex_ = nullptr;
};

} // namespace ui

} // namespace cgogn

#endif // CGOGN_MODULE_SURFACE_VECTOR_HEAT_METHOD_H_
