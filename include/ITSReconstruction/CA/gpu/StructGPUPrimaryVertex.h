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

	typedef struct __attribute__ ((packed)) int3Struct{
		cl_int x;
		cl_int y;
		cl_int z;
	}Int3Struct;

	typedef struct __attribute__ ((packed)) float3Struct{
		cl_float x;
		cl_float y;
		cl_float z;
	}Float3Struct;

	typedef struct __attribute__ ((packed)) clusterStruct{
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

	typedef struct __attribute__ ((packed)) cellStruct{
		const int mFirstClusterIndex;
		const int mSecondClusterIndex;
		const int mThirdClusterIndex;
		const int mFirstTrackletIndex;
		const int mSecondTrackletIndex;
		Float3Struct mNormalVectorCoordinates;
		const float mCurvature;
		int mLevel;
	}CellStruct;

	typedef struct __attribute__ ((packed)) trackletStruct{
		int firstClusterIndex;
		int secondClusterIndex;
		float tanLambda;
		float phiCoordinate;
	}TrackletStruct;










#endif /* TRAKINGITSU_INCLUDE_STRUCT_GPU_PRIMARY_H_ */
