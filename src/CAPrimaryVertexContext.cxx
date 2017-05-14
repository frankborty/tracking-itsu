/// \file CAPrimaryVertexContext.cxx
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

#include "CAPrimaryVertexContext.h"

#include <algorithm>
#include <cmath>

#include "CAConstants.h"
#include "CAEvent.h"
#include "CALayer.h"

CAPrimaryVertexContext::CAPrimaryVertexContext(const CAEvent& event, const int primaryVertexIndex)
    : primaryVertexIndex { primaryVertexIndex }
{
  for (int iLayer{ 0 }; iLayer < CAConstants::ITS::LayersNumber; ++iLayer) {

    const CALayer& currentLayer { event.getLayer(iLayer) };
    const int clustersNum { currentLayer.getClustersSize() };
    clusters[iLayer].reserve(clustersNum);

    for (int iCluster { 0 }; iCluster < clustersNum; ++iCluster) {

      const CACluster& currentCluster { currentLayer.getCluster(iCluster) };
      clusters[iLayer].emplace_back(iLayer, event.getPrimaryVertex(primaryVertexIndex), currentCluster);
    }

    std::sort(clusters[iLayer].begin(), clusters[iLayer].end(), [](CACluster& cluster1, CACluster& cluster2) {
      return cluster1.indexTableBinIndex < cluster2.indexTableBinIndex;
    });

#if defined(TRACKINGITSU_GPU_MODE)
    dClusters[iLayer] = CAGPUArray<CACluster> {&clusters[iLayer][0], static_cast<int>(clusters[iLayer].size())};
#endif

    if (iLayer > 0) {

      indexTables[iLayer - 1] = CAIndexTable(iLayer, clusters[iLayer]);
    }

    if (iLayer < CAConstants::ITS::TrackletsPerRoad) {

      tracklets[iLayer].reserve(
          std::ceil(
              (CAConstants::Memory::TrackletsMemoryCoefficients[iLayer] * clustersNum)
                  * event.getLayer(iLayer + 1).getClustersSize()));

#if defined(TRACKINGITSU_GPU_MODE)
      dTracklets[iLayer] = CAGPUArray<CATracklet> {static_cast<int>(tracklets[iLayer].capacity())};
#endif
    }

    if (iLayer < CAConstants::ITS::CellsPerRoad) {

      cells[iLayer].reserve(
          std::ceil(
              ((CAConstants::Memory::CellsMemoryCoefficients[iLayer] * clustersNum)
                  * event.getLayer(iLayer + 1).getClustersSize()) * event.getLayer(iLayer + 2).getClustersSize()));
    }
  }

#if defined(TRACKINGITSU_GPU_MODE)
  dIndexTables = CAGPUArray<CAIndexTable> {indexTables.data(), CAConstants::ITS::TrackletsPerRoad};
#endif
}

