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

#ifndef CGOGN_CORE_FUNCTIONS_ATTRIBUTES_H_
#define CGOGN_CORE_FUNCTIONS_ATTRIBUTES_H_

#include <cgogn/core/types/mesh_traits.h>
#include <cgogn/core/types/cmap/cmap_ops.h>

#include <cgogn/core/utils/tuples.h>

#include <string>

namespace cgogn
{

/*****************************************************************************/

// template <typename T, typename CELL, typename MESH>
// typename mesh_traits<MESH>::template Attribute<T>* add_attribute(MESH& m, const std::string& name);

/*****************************************************************************/

//////////////
// CMapBase //
//////////////

template <typename T, typename CELL, typename MESH,
		  typename = typename std::enable_if<std::is_base_of<CMapBase, MESH>::value>::type>
typename mesh_traits<MESH>::template Attribute<T>*
add_attribute(MESH& m, const std::string& name)
{
	static_assert(is_in_tuple<CELL, typename mesh_traits<MESH>::Cells>::value, "CELL not supported in this MESH");
	if (!m.template is_embedded<CELL>())
	{
		m.template create_embedding<CELL>();
		create_embeddings<CELL>(m);
	}
	return m.attribute_containers_[CELL::ORBIT].template add_attribute<T>(name);
}

/*****************************************************************************/

// template <typename T, typename CELL, typename MESH>
// typename mesh_traits<MESH>::template Attribute<T>* get_attribute(MESH& m, const std::string& name);

/*****************************************************************************/

//////////////
// CMapBase //
//////////////

template <typename T, typename CELL, typename MESH,
		  typename = typename std::enable_if<std::is_base_of<CMapBase, MESH>::value>::type>
typename mesh_traits<MESH>::template Attribute<T>*
get_attribute(const MESH& m, const std::string& name)
{
	static_assert(is_in_tuple<CELL, typename mesh_traits<MESH>::Cells>::value, "CELL not supported in this MESH");
	return m.attribute_containers_[CELL::ORBIT].template get_attribute<T>(name);
}

/*****************************************************************************/

// template <typename CELL, typename MESH>
// uint32 index_of(MESH& m, CELL c);

/*****************************************************************************/

//////////////
// CMapBase //
//////////////

template <typename CELL, typename MESH,
		  typename std::enable_if<std::is_base_of<CMapBase, MESH>::value>::type* = nullptr>
uint32
index_of(const MESH& m, CELL c)
{
	static_assert(is_in_tuple<CELL, typename mesh_traits<MESH>::Cells>::value, "CELL not supported in this MESH");
	return m.embedding(c);
}

//////////////
// MESHVIEW //
//////////////

template <typename CELL, typename MESH,
		  typename std::enable_if<is_mesh_view<MESH>::value>::type* = nullptr>
uint32
index_of(const MESH& m, CELL c)
{
	static_assert(is_in_tuple<CELL, typename mesh_traits<MESH>::Cells>::value, "CELL not supported in this MESH");
	return index_of(m.mesh(), c);
}

/*****************************************************************************/

// template <typename T, typename CELL, typename MESH>
// T& value(MESH& m, typename mesh_traits<MESH>::template Attribute<T>* attribute, CELL c);

/*****************************************************************************/

/////////////
// GENERIC //
/////////////

template <typename T, typename CELL, typename MESH>
const T&
value(const MESH& m, const typename mesh_traits<MESH>::template Attribute<T>* attribute, CELL c)
{
	static_assert(is_in_tuple<CELL, typename mesh_traits<MESH>::Cells>::value, "CELL not supported in this MESH");
	return (*attribute)[index_of(m, c)];
}

template <typename T, typename CELL, typename MESH>
T&
value(const MESH& m, typename mesh_traits<MESH>::template Attribute<T>* attribute, CELL c)
{
	static_assert(is_in_tuple<CELL, typename mesh_traits<MESH>::Cells>::value, "CELL not supported in this MESH");
	return (*attribute)[index_of(m, c)];
}

} // namespace cgogn

#endif // CGOGN_CORE_FUNCTIONS_ATTRIBUTES_H_
