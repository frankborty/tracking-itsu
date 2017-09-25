/// \file IOUtils.cxx
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

#include "ITSReconstruction/CA/IOUtils.h"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <tuple>
#include <unordered_set>
#include <utility>

#include "ITSReconstruction/CA/Constants.h"

namespace {
constexpr int PrimaryVertexLayerId { -1 };
constexpr int EventLabelsSeparator { -1 };
}

namespace o2
{
namespace ITS
{
namespace CA
{

std::vector<Event> IOUtils::loadEventData(const std::string& fileName)
{
  std::vector<Event> events { };
  std::ifstream inputStream { };
  std::string line { }, unusedVariable { };
  int layerId { }, monteCarlo { };
  int clusterId { EventLabelsSeparator };
  float xCoordinate { }, yCoordinate { }, zCoordinate { }, alphaAngle { };

  inputStream.open(fileName);

  while (std::getline(inputStream, line)) {

    std::istringstream inputStringStream(line);

    if (inputStringStream >> layerId >> xCoordinate >> yCoordinate >> zCoordinate) {

      if (layerId == PrimaryVertexLayerId) {

        if (clusterId != 0) {

          events.emplace_back(events.size());
        }

        events.back().addPrimaryVertex(xCoordinate, yCoordinate, zCoordinate);
        clusterId = 0;

      } else {

        if (inputStringStream >> unusedVariable >> unusedVariable >> unusedVariable >> alphaAngle >> monteCarlo) {

          events.back().pushClusterToLayer(layerId, clusterId, xCoordinate, yCoordinate, zCoordinate, alphaAngle,
              monteCarlo);
          ++clusterId;
        }
      }
    }
  }

  return events;
}

std::vector<std::unordered_map<int, Label>> IOUtils::loadLabels(const int eventsNum, const std::string& fileName)
{
  std::vector<std::unordered_map<int, Label>> labelsMap { };
  std::unordered_map<int, Label> currentEventLabelsMap { };
  std::ifstream inputStream { };
  std::string line { };
  int monteCarloId { }, pdgCode { }, numberOfClusters { };
  float transverseMomentum { }, phiCoordinate { }, pseudorapidity { };

  labelsMap.reserve(eventsNum);

  inputStream.open(fileName);
  std::getline(inputStream, line);

  while (std::getline(inputStream, line)) {

    std::istringstream inputStringStream(line);

    if (inputStringStream >> monteCarloId) {

      if (monteCarloId == EventLabelsSeparator) {

        labelsMap.emplace_back(currentEventLabelsMap);
        currentEventLabelsMap.clear();

      } else {

        if (inputStringStream >> transverseMomentum >> phiCoordinate >> pseudorapidity >> pdgCode >> numberOfClusters) {

          if (std::abs(pdgCode) == Constants::PDGCodes::PionCode && numberOfClusters == 7) {

            currentEventLabelsMap.emplace(std::piecewise_construct, std::forward_as_tuple(monteCarloId),
                std::forward_as_tuple(monteCarloId, transverseMomentum, phiCoordinate, pseudorapidity, pdgCode,
                    numberOfClusters));
          }
        }
      }
    }
  }

  labelsMap.emplace_back(currentEventLabelsMap);

  return labelsMap;
}

void IOUtils::writeRoadsReport(std::ofstream& correctRoadsOutputStream, std::ofstream& duplicateRoadsOutputStream,
    std::ofstream& fakeRoadsOutputStream, const std::vector<std::vector<Road>>& roads,
    const std::unordered_map<int, Label>& labelsMap)
{
  const int numVertices { static_cast<int>(roads.size()) };
  std::unordered_set<int> foundMonteCarloIds { };

  correctRoadsOutputStream << EventLabelsSeparator << std::endl;
  fakeRoadsOutputStream << EventLabelsSeparator << std::endl;

  for (int iVertex { 0 }; iVertex < numVertices; ++iVertex) {

    const std::vector<Road>& currentVertexRoads { roads[iVertex] };
    const int numRoads { static_cast<int>(currentVertexRoads.size()) };

    for (int iRoad { 0 }; iRoad < numRoads; ++iRoad) {

      const Road& currentRoad { currentVertexRoads[iRoad] };
      const int currentRoadLabel { currentRoad.getLabel() };

      if (!labelsMap.count(currentRoadLabel)) {

        continue;
      }

      const Label& currentLabel { labelsMap.at(currentRoadLabel) };

      if (currentRoad.isFakeRoad()) {

        fakeRoadsOutputStream << currentLabel << std::endl;

      } else {

        if (foundMonteCarloIds.count(currentLabel.monteCarloId)) {

          duplicateRoadsOutputStream << currentLabel << std::endl;

        } else {

          correctRoadsOutputStream << currentLabel << std::endl;
          foundMonteCarloIds.emplace(currentLabel.monteCarloId);
        }
      }
    }
  }
}

}
}
}