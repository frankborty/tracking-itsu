#ifndef TRAKINGITSU_INCLUDE_STRUCT_GPU_PRIMARY_H_
#define TRAKINGITSU_INCLUDE_STRUCT_GPU_PRIMARY_H_

#include <array>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

#include "ITSReconstruction/CA/Definitions.h"
/*
#include "ITSReconstruction/CA/Event.h"
#include "ITSReconstruction/CA/MathUtils.h"
#include "ITSReconstruction/CA/PrimaryVertexContext.h"
#include "ITSReconstruction/CA/Road.h"
*/
#include "ITSReconstruction/CA/Constants.h"
#include "CL/cl.hpp"


	typedef struct __attribute__ ((packed)) float3Struct{
		cl_float x;
		cl_float y;
		cl_float z;
	}Float3Struct;

	typedef struct{
		float xCoordinate;
		float yCoordinate;
		float zCoordinate;
		float phiCoordinate;
		float rCoordinate;
		int clusterId;
		float alphaAngle;
		int monteCarloId;
		int indexTableBinIndex;
	}ClusterStruct;

	typedef struct{
		const int mFirstClusterIndex;
		const int mSecondClusterIndex;
		const int mThirdClusterIndex;
		const int mFirstTrackletIndex;
		const int mSecondTrackletIndex;
		Float3Struct mNormalVectorCoordinates;
		const float mCurvature;
	int mLevel;
	}CellStruct;

	typedef struct{
		const int firstClusterIndex;
		const int secondClusterIndex;
		const float tanLambda;
		const float phiCoordinate;
	}TrackletStruct;

	typedef struct{
		const int firstClusterIndex;
		const int secondClusterIndex;
		const float tanLambda;
		const float phiCoordinate;
	}RoadsStruct;

	typedef struct{
		void * srPunt;
		int size;
	}VectStruct;

	typedef struct {
		Float3Struct mPrimaryVertex;
		cl::Buffer bPrimaryVertex;

		int ClusterSize;
		VectStruct mClusters[o2::ITS::CA::Constants::ITS::LayersNumber];
		cl::Buffer bClusters[o2::ITS::CA::Constants::ITS::LayersNumber];
		cl::Buffer bClustersSize;

		int CellsSize;
		VectStruct mCells[o2::ITS::CA::Constants::ITS::CellsPerRoad];
		cl::Buffer bCells[o2::ITS::CA::Constants::ITS::CellsPerRoad];
		cl::Buffer bCellsSize;

		int CellsLookupTableSize;
		VectStruct mCellsLookupTable[o2::ITS::CA::Constants::ITS::CellsPerRoad - 1];
		cl::Buffer bCellsLookupTable[o2::ITS::CA::Constants::ITS::CellsPerRoad - 1];
		cl::Buffer bCellsLookupTableSize;

		int IndexTableSize;
		VectStruct mIndexTable[o2::ITS::CA::Constants::IndexTable::ZBins * o2::ITS::CA::Constants::IndexTable::PhiBins + 1];
		cl::Buffer bIndexTable[o2::ITS::CA::Constants::IndexTable::ZBins * o2::ITS::CA::Constants::IndexTable::PhiBins + 1];
		cl::Buffer bIndexTableSize;



		int TrackeltsSize;
		VectStruct mTracklets[o2::ITS::CA::Constants::ITS::TrackletsPerRoad];
		cl::Buffer bTracklets[o2::ITS::CA::Constants::ITS::TrackletsPerRoad];
		cl::Buffer bTrackletsSize;


		int TrackletLookupTableSize;
		VectStruct mTrackletLookupTable [o2::ITS::CA::Constants::ITS::CellsPerRoad];
		cl::Buffer bTrackletLookupTable [o2::ITS::CA::Constants::ITS::CellsPerRoad];
		cl::Buffer bTrackletLookupTableSize;
		/*int IndexTableX;
		int IndexTavleY;
		int **mIndexTable;
		int CellsNeighX;
		int CellsNeighY;
		int *mCellsNeigh;
		int RoadsSize;
		VectStruct mRoads;*/
	}PrimaryVertexContestStruct;




#endif /* TRAKINGITSU_INCLUDE_STRUCT_GPU_PRIMARY_H_ */
