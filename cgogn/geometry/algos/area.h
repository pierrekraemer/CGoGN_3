/*******************************************************************************
* CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional Maps  *
* Copyright (C) 2015, IGG Group, ICube, University of Strasbourg, France       *
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

#ifndef CGOGN_GEOMETRY_ALGOS_AREA_H_
#define CGOGN_GEOMETRY_ALGOS_AREA_H_

#include <cgogn/core/types/mesh_traits.h>

#include <cgogn/core/functions/traversals/face.h>
#include <cgogn/core/functions/mesh_info.h>
#include <cgogn/core/functions/attributes.h>

#include <cgogn/geometry/functions/area.h>
#include <cgogn/geometry/algos/centroid.h>
#include <cgogn/geometry/types/vector_traits.h>

namespace cgogn
{

namespace geometry
{

template <typename MESH>
Scalar
convex_area(
	const MESH& m,
	typename mesh_traits<MESH>::Face f,
	const typename mesh_traits<MESH>::template Attribute<Vec3>* vertex_position
)
{
	using Vertex = typename mesh_traits<MESH>::Vertex;
    if (codegree(m, f) == 3)
    {
        std::vector<Vertex> vertices = incident_vertices(m, f);
		return area(
            value<Vec3>(m, vertex_position, vertices[0]),
            value<Vec3>(m, vertex_position, vertices[1]),
            value<Vec3>(m, vertex_position, vertices[2])
        );
    }
	else
	{
		Scalar face_area{0};
		Vec3 center = centroid<Vec3>(m, f, vertex_position);
        std::vector<Vertex> vertices = incident_vertices(m, f);
        for (uint32 i = 0, size = vertices.size(); i < size; ++i)
		{
			face_area += area(
                center,
                value<Vec3>(m, vertex_position, vertices[i]),
                value<Vec3>(m, vertex_position, vertices[(i + 1) % size])
            );
		}
		return face_area;
	}
}

template <typename MESH>
Scalar
area(
	const MESH& m,
	typename mesh_traits<MESH>::Face f,
	const typename mesh_traits<MESH>::template Attribute<Vec3>* vertex_position
)
{
    return convex_area(m, f, vertex_position);
}

template <typename MESH>
Scalar
area(
	const MESH& m,
	const typename mesh_traits<MESH>::template Attribute<Vec3>* vertex_position
)
{
	using Face = typename mesh_traits<MESH>::Face;

	// std::vector<Scalar> area_per_thread(thread_pool()->nb_workers(), 0);
	// std::vector<uint32> nb_faces_per_thread(thread_pool()->nb_workers(), 0);

    // parallel_foreach_cell(m, [&] (Face f) -> bool
    // {
	// 	uint32 thread_index = current_thread_index();
	// 	area_per_thread[thread_index] += area(m, f, vertex_position);
	// 	++nb_faces_per_thread[thread_index];
    //     return true;
    // });

    // Scalar area_sum = 0;
	// uint32 nbf = 0;
	// for (Scalar a : area_per_thread) area_sum += a;
	// for (uint32 n : nb_faces_per_thread) nbf += n;

	// return area_sum / Scalar(nbf);

    Scalar area_sum = 0;
	uint32 nbf = 0;

    foreach_cell(m, [&] (Face f) -> bool
    {
		area_sum += area(m, f, vertex_position);
		++nbf;
        return true;
    });

	return area_sum / Scalar(nbf);
}

} // namespace geometry

} // namespace cgogn

#endif // CGOGN_GEOMETRY_ALGOS_AREA_H_
