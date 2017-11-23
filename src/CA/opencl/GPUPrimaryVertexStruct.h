#include "ITSReconstruction/CA/Constants.h"


	typedef struct{
		float x;
		float y;
		float z;
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
		Float3Struct *mPrimaryVertex;
		int ClusterSize;
		VectStruct mClusters[o2::ITS::CA::Constants::ITS::LayersNumber];
		int CellsSize;
		VectStruct mCells[o2::ITS::CA::Constants::ITS::CellsPerRoad];
		int CellsLookupTableSize;
		VectStruct mCellsLookupTable[o2::ITS::CA::Constants::ITS::CellsPerRoad - 1];
		int IndexTableSize;
		VectStruct mIndexTable[o2::ITS::CA::Constants::IndexTable::ZBins * o2::ITS::CA::Constants::IndexTable::PhiBins + 1];
		int TrackeltsSize;
		VectStruct mTracklets[o2::ITS::CA::Constants::ITS::TrackletsPerRoad];
		int TrackletLookupTableSize;
		VectStruct mTrackletLookupTable[o2::ITS::CA::Constants::ITS::CellsPerRoad];
		/*int IndexTableX;
		int IndexTavleY;
		int **mIndexTable;
		int CellsNeighX;
		int CellsNeighY;
		int *mCellsNeigh;
		int RoadsSize;
		VectStruct mRoads;*/
	}PrimaryVertexContestStruct;
