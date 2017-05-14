/// \file CAPrimaryVertexContext.h
/// \brief 
///
/// \author Iacopo Colonnelli, Politecnico di Torino
///
/// \copyright Copyright (C) 2017  Iacopo Colonnelli. \n\n
///   This program is free software: you can redistribute it and/or modify
///   it under the terms of the GNU General Public License as published by
///   the Free Software Foundation, either version 3 of the License, or
///   (at your option) any later version. \n\n
///   This program is distributed in the hope that it will be useful,
///   but WITHOUT ANY WARRANTY; without even the implied warranty of
///   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
///   GNU General Public License for more details. \n\n
///   You should have received a copy of the GNU General Public License
///   along with this program.  If not, see <http://www.gnu.org/licenses/>.
///////////////////////////////////////////////////////////////////////////////
#ifndef TRACKINGITSU_INCLUDE_CATRACKERCONTEXT_H_
#define TRACKINGITSU_INCLUDE_CATRACKERCONTEXT_H_

#include <array>
#include <vector>

#include "CACell.h"
#include "CAConstants.h"
#include "CADefinitions.h"
#include "CAEvent.h"
#include "CAIndexTable.h"
#include "CARoad.h"
#include "CATracklet.h"

#if defined(TRACKINGITSU_GPU_MODE)
# include "CAGPUArray.h"
#endif

struct CAPrimaryVertexContext
    final
    {
      explicit CAPrimaryVertexContext(const CAEvent&, const int);

      CAPrimaryVertexContext(const CAPrimaryVertexContext&) = delete;
      CAPrimaryVertexContext &operator=(const CAPrimaryVertexContext&) = delete;

      const int primaryVertexIndex;

      std::array<std::vector<CACluster>, CAConstants::ITS::LayersNumber> clusters;
      std::array<CAIndexTable, CAConstants::ITS::TrackletsPerRoad> indexTables;
      std::array<std::vector<CATracklet>, CAConstants::ITS::TrackletsPerRoad> tracklets;
      std::array<std::vector<int>, CAConstants::ITS::CellsPerRoad> trackletsLookupTable;
      std::array<std::vector<CACell>, CAConstants::ITS::CellsPerRoad> cells;
      std::array<std::vector<int>, CAConstants::ITS::CellsPerRoad - 1> cellsLookupTable;
      std::vector<CARoad> roads;

#if defined(TRACKINGITSU_GPU_MODE)
      std::array<CAGPUArray<CACluster>, CAConstants::ITS::LayersNumber> dClusters;
      CAGPUArray<CAIndexTable> dIndexTables;
      std::array<CAGPUArray<CATracklet>, CAConstants::ITS::TrackletsPerRoad> dTracklets;
#endif

  };

#endif /* TRACKINGITSU_INCLUDE_CATRACKERCONTEXT_H_ */
