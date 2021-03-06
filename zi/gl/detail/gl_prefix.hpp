//
// Copyright (C) 2010  Aleksandar Zlateski <zlateski@mit.edu>
// ----------------------------------------------------------
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#ifndef ZI_GL_DETAIL_GL_PREFIX_HPP
#define ZI_GL_DETAIL_GL_PREFIX_HPP 1

#include <zi/gl/detail/config.hpp>

#if defined( macintosh ) && PRAGMA_IMPORT_SUPPORTED
#  pragma import on
#endif

#define ZI_GLAPI extern "C"

#define ZI_GLAPI_ENTRY

#endif
