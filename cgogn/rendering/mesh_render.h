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

#ifndef CGOGN_RENDERING_MESH_RENDER_H_
#define CGOGN_RENDERING_MESH_RENDER_H_

#include <cgogn/rendering/cgogn_rendering_export.h>

#include <cgogn/core/functions/attributes.h>
#include <cgogn/core/functions/traversals/global.h>
#include <cgogn/core/functions/traversals/vertex.h>
#include <cgogn/core/functions/traversals/edge.h>
#include <cgogn/core/functions/traversals/face.h>

#include <cgogn/core/utils/numerics.h>

 #include <cgogn/geometry/algos/ear_triangulation.h>

#include <memory>

#include <cgogn/rendering/drawer.h>
#include <cgogn/rendering/ebo.h>
#include <cgogn/rendering/vbo.h>

namespace cgogn
{

namespace rendering
{

enum DrawingType : uint32
{
	POINTS = 0,
	LINES,
	TRIANGLES,

	VOLUMES_FACES,
	VOLUMES_EDGES,
	VOLUMES_VERTICES,

	INDEX_EDGES,
	INDEX_FACES,
	INDEX_VOLUMES,

	SIZE_BUFFER,
	POINTS_TB,
	LINES_TB,
	TRIANGLES_TB,
	VOLUMES_FACES_TB,
	VOLUMES_EDGES_TB,
	VOLUMES_VERTICES_TB,
	INDEX_EDGES_TB,
	INDEX_FACES_TB,
	INDEX_VOLUMES_TB
};

static std::vector<std::string> primitives_names = {
"POINTS",
"LINES",
"TRIANGLES",
"VOLUMES_FACES",
"VOLUMES_EDGES",
"VOLUMES_VERTICES",
"INDEX_EDGES",
"INDEX_FACES",
"INDEX_VOLUMES",
"SIZE_BUFFER",
"POINTS_TB",
"LINES_TB",
"TRIANGLES_TB",
"VOLUMES_FACES_TB",
"VOLUMES_EDGES_TB",
"VOLUMES_VERTICES_TB",
"INDEX_EDGES_TB",
"INDEX_FACES_TB",
"INDEX_VOLUMES_TB"
};


class CGOGN_RENDERING_EXPORT MeshRender
{
protected:
	std::array<std::unique_ptr<EBO>, SIZE_BUFFER> indices_buffers_;
	std::array<bool, SIZE_BUFFER> indices_buffers_uptodate_;

public:
	MeshRender();
	~MeshRender();
	CGOGN_NOT_COPYABLE_NOR_MOVABLE(MeshRender);

	inline bool is_primitive_uptodate(DrawingType prim)
	{
		return indices_buffers_uptodate_[prim];
	}

	inline void set_primitive_dirty(DrawingType prim)
	{
		indices_buffers_uptodate_[prim] = false;
	}

protected:
	template <typename MESH>
	inline void init_points(const MESH& m, std::vector<uint32>& table_indices)
	{
		using Vertex = typename mesh_traits<MESH>::Vertex;
		foreach_cell(m, [&](Vertex v) -> bool {
			table_indices.push_back(index_of(m, v));
			return true;
		});
	}


	template <typename MESH>
	inline void init_lines(const MESH& m, std::vector<uint32>& table_indices, std::vector<uint32>& table_emb_edge)
	{
		if constexpr (mesh_traits<MESH>::dimension > 0)
		{
			using Vertex = typename mesh_traits<MESH>::Vertex;
			using Edge = typename mesh_traits<MESH>::Edge;
			uint32 i_e;
			foreach_cell(m, [&](Edge e) -> bool
			{
				foreach_incident_vertex(m, e, [&](Vertex v) -> bool
				{
					table_indices.push_back(index_of(m, v));
					return true;
				});
				table_emb_edge.push_back(is_indexed<Edge>(m) ? index_of(m,e) :i_e);
				++i_e;
				return true;
			});
		}
	}


	template <typename MESH>
	inline void init_triangles(const MESH& m, std::vector<uint32>& table_indices, std::vector<uint32>& table_emb_face)
	{
		if constexpr (mesh_traits<MESH>::dimension > 1)
		{
			using Vertex = typename mesh_traits<MESH>::Vertex;
			using Face = typename mesh_traits<MESH>::Face;
			std::vector<Vertex> vertices;
			vertices.reserve(32u);
			bool emb = is_indexed<Face>(m);
			uint32 i_f=0;
			foreach_cell(m, [&](Face f) -> bool
			{
				vertices.clear();
				incident_vertices(m, f, vertices);
				for (uint32 i = 1; i < vertices.size() - 1; ++i)
				{
					table_indices.push_back(index_of(m, vertices[0]));
					table_indices.push_back(index_of(m, vertices[i]));
					table_indices.push_back(index_of(m, vertices[i + 1]));
				}
				table_emb_face.push_back( emb ? index_of(m,f) : i_f++);
				return true;
			});
		}
	}


	template <typename MESH>
	inline void init_ear_triangles(const MESH& m, std::vector<uint32>& table_indices, std::vector<uint32>& table_emb_face,
				   const typename mesh_traits<MESH>::template Attribute<geometry::Vec3>* position)
	{
		if constexpr (mesh_traits<MESH>::dimension > 1)
		{
			using Vertex = typename mesh_traits<MESH>::Vertex;
			using Face = typename mesh_traits<MESH>::Face;
			std::vector<Vertex> vertices;
			vertices.reserve(32u);
			uint32 i_f=0;
			bool emb = is_indexed<Face>(m);
			foreach_cell(m, [&](Face f) -> bool
			{
				if (codegree(m,f)==3)
				{
					vertices.clear();
					incident_vertices(m, f, vertices);

					table_indices.push_back(index_of(m, vertices[0]));
					table_indices.push_back(index_of(m, vertices[1]));
					table_indices.push_back(index_of(m, vertices[2]));
					table_emb_face.push_back(emb ? index_of(m, f) : i_f);
				}
				else
				{
					if (emb)
					{
						cgogn::geometry::append_ear_triangulation(m, f, position, table_indices,
						  [&] ()
						  {
							  table_emb_face.push_back(index_of(m, f));
						  });
					}
					else
					{
						cgogn::geometry::append_ear_triangulation(m, f, position, table_indices,
						  [&] ()
						  {
							  table_emb_face.push_back(i_f);
						  });
					}

				}
				i_f++;
				return true;
			});
		}
	}


//	template <bool EMB, typename MESH>
//	inline void init_faces_centers(const MESH& m, std::vector<uint32>& table_indices, std::vector<uint32>& table_emb_edge)
//	{
//		if constexpr (mesh_traits<MESH>::dimension > 0)
//		{
//			using Vertex = typename mesh_traits<MESH>::Vertex;
//			using Face = typename mesh_traits<MESH>::Edge;
//			uint32 last = 0;
//			bool emb = is_indexed<Face>(m);
//			foreach_cell(m, [&](Face f) -> bool
//			{
//				foreach_incident_vertex(m, f, [&](Vertex v) -> bool
//				{
//					table_indices.push_back(index_of(m, v));
//					table_indices.push_back(emb ? index_of(m,f) : last);
//					return true;
//				});
//				return true;
//				++last;
//			});
//		}
//	}

	template <typename MESH>
	inline void init_volumes(const MESH& m,
							 std::vector<uint32>& table_indices_f,
							 std::vector<uint32>& table_indices_e,
							 std::vector<uint32>& table_indices_v,
							 std::vector<uint32>& table_emb_vol,
			const typename mesh_traits<MESH>::template Attribute<geometry::Vec3>* position)
	{

		if constexpr (mesh_traits<MESH>::dimension >= 2)
		{
			using Vertex = typename mesh_traits<MESH>::Vertex;
			using Edge = typename mesh_traits<MESH>::Edge;
			using Face = typename mesh_traits<MESH>::Face;
			using Volume = typename mesh_traits<MESH>::Volume;

			std::vector<Vertex> vertices;
			vertices.reserve(256);
			uint32 i_vol=0;
			bool emb = is_indexed<Volume>(m);
			foreach_cell(m, [&](Volume vol) -> bool
			{
				foreach_incident_face(m, vol, [&](Face f)-> bool
				{
					if (codegree(m,f)==3)
					{
						vertices.clear();
						incident_vertices(m, f, vertices);
						table_indices_f.push_back(index_of(m, vertices[0]));
						table_indices_f.push_back(index_of(m, vertices[1]));
						table_indices_f.push_back(index_of(m, vertices[2]));
						table_indices_f.push_back(emb ? index_of(m, vol) : i_vol);
					}
					else
					{
						if (emb)
						{
							cgogn::geometry::append_ear_triangulation(m, f, position, table_indices_f,
								[&] () { table_indices_f.push_back(index_of(m, vol)); });
						}
						else
						{
							cgogn::geometry::append_ear_triangulation(m, f, position, table_indices_f,
								[&] () { table_indices_f.push_back(i_vol); });
						}
					}
					return true;
				});

				foreach_incident_edge(m, vol, [&](Edge e)-> bool
				{
					vertices.clear();
					incident_vertices(m, e, vertices);

					table_indices_e.push_back(index_of(m, vertices[0]));
					table_indices_e.push_back(index_of(m, vertices[1]));
					table_indices_e.push_back(emb ? index_of(m, vol) : i_vol);
					return true;
				});

				foreach_incident_vertex(m, vol, [&](Vertex v)-> bool
				{
					table_indices_v.push_back(index_of(m, v));
					table_indices_v.push_back(emb ? index_of(m, vol) : i_vol);
					return true;
				});

				table_emb_vol.push_back(emb ? index_of(m, vol) : i_vol++);

				return true;
			});
		}
	}

public:

	template <typename MESH>
	inline void init_primitives(const MESH& m, DrawingType prim,
								const typename mesh_traits<MESH>::template Attribute<geometry::Vec3>* position = nullptr)
	{
		if (prim>=SIZE_BUFFER)
			prim = DrawingType(prim + POINTS - POINTS_TB);

		auto func_update_ebo = [&] (DrawingType pr, const std::vector<uint32>& table ) -> void
		{
			indices_buffers_uptodate_[pr] = true;
			if (!table.empty())
			{
				if (!indices_buffers_[pr]->is_created())
					indices_buffers_[pr]->create();
				indices_buffers_[pr]->allocate(table.data(), table.size());
				indices_buffers_[pr]->set_name("EBO_"+primitives_names[pr]);
			}
		};

		auto start_timer = std::chrono::high_resolution_clock::now();

		std::vector<uint32> table_indices;
		table_indices.reserve(1024u);
		std::vector<uint32> table_indices_emb;
		table_indices_emb.reserve(1024u);

		std::vector<uint32> table_indices_e;
		table_indices_emb.reserve(1024u);
		std::vector<uint32> table_indices_v;
		table_indices_v.reserve(1024u);

		switch (prim)
		{
		case POINTS:
			init_points(m, table_indices);
			func_update_ebo(POINTS,table_indices);
			break;
		case LINES:
		case INDEX_EDGES:
			init_lines(m, table_indices,table_indices_emb);
			func_update_ebo(LINES,table_indices);
			func_update_ebo(INDEX_EDGES,table_indices_emb);
			break;
		case TRIANGLES:
		case INDEX_FACES:
			if (position == nullptr)
				init_triangles(m, table_indices, table_indices_emb);
			else
				init_ear_triangles(m, table_indices, table_indices_emb, position);
			func_update_ebo(TRIANGLES,table_indices);
			func_update_ebo(INDEX_FACES,table_indices_emb);
			break;

		case VOLUMES_VERTICES:
		case VOLUMES_EDGES:
		case VOLUMES_FACES:
		case INDEX_VOLUMES:
			init_volumes(m, table_indices, table_indices_e,table_indices_v, table_indices_emb, position);
			func_update_ebo(VOLUMES_FACES,table_indices);
			func_update_ebo(VOLUMES_EDGES,table_indices_e);
			func_update_ebo(VOLUMES_VERTICES,table_indices_v);
			func_update_ebo(INDEX_VOLUMES, table_indices_emb);
			break;
		default:
			break;
		}

		auto end_timer = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed_seconds = end_timer-start_timer;
		std::cout << "init primitive "<<prim<< " in "<< elapsed_seconds.count() << std::endl;

	}

	void draw(DrawingType prim, GLint binding_point=10);

	inline void bind_ebo_tb(DrawingType prim, GLint binding_point)
	{
		if (prim >= SIZE_BUFFER)
			indices_buffers_[prim - SIZE_BUFFER - 1]->bind_tb(binding_point);
		else
			indices_buffers_[prim]->bind_tb(binding_point);
	}



};

} // namespace rendering

} // namespace cgogn

#endif // CGOGN_RENDERING_MESH_RENDER_H_
