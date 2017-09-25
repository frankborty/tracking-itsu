/// \file Cluster.h
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

#ifndef TRACKINGITSU_INCLUDE_CACLUSTER_H_
#define TRACKINGITSU_INCLUDE_CACLUSTER_H_

#include "ITSReconstruction/CA/Definitions.h"
#include "ITSReconstruction/CA/MathUtils.h"
#include "ITSReconstruction/CA/IndexTableUtils.h"

namespace o2
{
namespace ITS
{
namespace CA
{

struct Cluster
    final
    {
      Cluster(const int, const int, const float, const float, const float, const float, const int);
      Cluster(const int, const float3&, const Cluster&);

      float xCoordinate;
      float yCoordinate;
      float zCoordinate;
      float phiCoordinate;
      float rCoordinate;
      int clusterId;
      float alphaAngle;
      int monteCarloId;
      int indexTableBinIndex;
  };

}
}
}

#endif /* TRACKINGITSU_INCLUDE_CACLUSTER_H_ */