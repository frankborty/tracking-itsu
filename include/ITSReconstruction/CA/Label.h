/// \file Label.h
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
#ifndef TRACKINGITSU_INCLUDE_LABEL_H_
#define TRACKINGITSU_INCLUDE_LABEL_H_

#include <ostream>

namespace o2
{
namespace ITS
{
namespace CA
{

struct Label
    final
    {
      Label(const int, const float, const float, const float, const int, const int);

      int monteCarloId;
      float transverseMomentum;
      float phiCoordinate;
      float pseudorapidity;
      int pdgCode;
      int numberOfClusters;

      friend std::ostream& operator<<(std::ostream&, const Label&);
  };

}
}
}

#endif /* TRACKINGITSU_INCLUDE_LABEL_H_ */