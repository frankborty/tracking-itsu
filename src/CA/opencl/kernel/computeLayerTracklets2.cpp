//============================================================================
// Name        : KernelTest.cpp
// Author      : frank
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================



//__constant float TrackletMaxDeltaZThreshold[6]= { 0.1f, 0.1f, 0.3f, 0.3f, 0.3f, 0.3f }; //default
__constant float TrackletMaxDeltaZThreshold[6]= { 100.1f, 100.1f, 100.3f, 100.3f, 100.3f, 100.3f };
__constant int ZBins=20;
__constant int PhiBins=20;
__constant float Pi=3.14159265359f;
__constant float TwoPi=2.0f * 3.14159265359f ;
__constant int UnusedIndex=-1 ;
__constant float CellMaxDeltaPhiThreshold=0.14f;
//__constant float PhiCoordinateCut=0.3f;	//default
__constant float PhiCoordinateCut= 8.0f*3.14159265359f/4.0f;	//default

__constant float ZCoordinateCut=0.5f;	//default
__constant float InversePhiBinSize=20 / (2.0f * 3.14159265359f) ;
__constant float LayersZCoordinate[7]={16.333f, 16.333f, 16.333f, 42.140f, 42.140f, 73.745f, 73.745f};
__constant float LayersRCoordinate[7]={2.33959f, 3.14076f, 3.91924f, 19.6213f, 24.5597f, 34.388f, 39.3329f};
__constant float InverseZBinSize[7]=  {0.5f * 20 / 16.333f, 0.5f * 20 / 16.333f, 0.5f * 20 / 16.333f,0.5f * 20 / 42.140f, 0.5f * 20 / 42.140f, 0.5f * 20 / 73.745f, 0.5f * 20 / 73.745f };


	typedef struct{
		int x;
		int y;
		int z;
		int w;
	}Int4Struct;

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
		int mFirstClusterIndex;
		int mSecondClusterIndex;
		int mThirdClusterIndex;
		int mFirstTrackletIndex;
		int mSecondTrackletIndex;
		Float3Struct mNormalVectorCoordinates;
		float mCurvature;
		int mLevel;
	}CellStruct;

	typedef struct{
		int firstClusterIndex;
		int secondClusterIndex;
		float tanLambda;
		float phiCoordinate;
	}TrackletStruct;

	typedef struct{
		int firstClusterIndex;
		int secondClusterIndex;
		float tanLambda;
		float phiCoordinate;
	}RoadsStruct;

	typedef struct{
		void * srPunt;
		int size;
	}VectStruct;


int myMin(int a,int b){
	if(a<b)
		return a;
	return b;
}

int myMax(int a,int b){
	if(a>b)
		return a;
	return b;
}

double myAbs(double iNum){
	if(iNum<0)
		return -1*iNum;
	return iNum;
}



int getZBinIndex(int layerIndex, float zCoordinate){
	return (zCoordinate + LayersZCoordinate[layerIndex])* InverseZBinSize[layerIndex];
}

int getPhiBinIndex(float currentPhi)
{
  return (currentPhi * InversePhiBinSize);
}

float getNormalizedPhiCoordinate(float phiCoordinate)
{
  return (phiCoordinate < 0) ? phiCoordinate + TwoPi :
         (phiCoordinate > TwoPi) ? phiCoordinate - TwoPi : phiCoordinate;
}


Int4Struct getBinsRect(__global ClusterStruct* currentCluster, int layerIndex,float directionZIntersection)
{
	const float zRangeMin = directionZIntersection - 2 * ZCoordinateCut;
	const float phiRangeMin = currentCluster->phiCoordinate - PhiCoordinateCut;
	const float zRangeMax = directionZIntersection + 2 * ZCoordinateCut;
	const float phiRangeMax = currentCluster->phiCoordinate + PhiCoordinateCut;
	Int4Struct binRect;
	binRect.x=0;
	binRect.y=0;
	binRect.z=0;
	binRect.w=0;

	if (zRangeMax < -LayersZCoordinate[layerIndex + 1]|| zRangeMin > LayersZCoordinate[layerIndex + 1] || zRangeMin > zRangeMax) {
		return binRect;
	}

	binRect.x=myMax(0, getZBinIndex(layerIndex + 1, zRangeMin));
	binRect.y=getPhiBinIndex(getNormalizedPhiCoordinate(phiRangeMin));
	binRect.z=myMin(ZBins - 1, getZBinIndex(layerIndex + 1, zRangeMax));
	binRect.w=getPhiBinIndex(getNormalizedPhiCoordinate(phiRangeMax));
	return binRect;

}


int getBinIndex(int zIndex,int phiIndex)
{
	return myMin(phiIndex * PhiBins + zIndex,ZBins * PhiBins);
}






__kernel void computeLayerTracklets(
					__global Float3Struct* primaryVertex,	//0
					__global ClusterStruct* currentLayerClusters, //1
					__global ClusterStruct* nextLayerClusters, //2
					__global int * currentLayerIndexTable, //3
					__global int * currentLayerTrackletsLookupTable, //4
					__global int * previousLayerTrackletsLookupTable, //5
					__global TrackletStruct* currentLayerTracklets, //6
					__global int * iTrackletFound, //7
					__global int * iParamNextLayerSize, //8
					__global int * iCurrentLayer, //9
					__global float * iPhiThreshold, //10
					__global float * iZThreshold) //11
					
{

	int iCluster=get_global_id(0); //lo trovo con l'id
	int iLayer=*iCurrentLayer;
	int iNextLayerSize=*iParamNextLayerSize;
	int iCurrentTrackletPosition=iCluster*iNextLayerSize/10;
	int iLocalTrackletNum=0;
	float myPhithr=*iPhiThreshold;
	float myZThr=*iZThreshold;
	if(iLayer==0)
	//	printf("Current cluster id=%d\n",iCluster);
	//	printf("My phiThr.=%.2f		-	 My zThr.=%.2f\n",myPhithr,myZThr);
		
		
	__global ClusterStruct *currentCluster=&currentLayerClusters[iCluster];

	float tanLambda=(currentCluster->zCoordinate-primaryVertex->z)/currentCluster->rCoordinate;

	float directionZIntersection= tanLambda*(LayersRCoordinate[iLayer+1]-currentCluster->rCoordinate)+currentCluster->zCoordinate;


	Int4Struct selectedBinsRect=getBinsRect(currentCluster,iLayer,directionZIntersection);

	if (selectedBinsRect.x == 0 && selectedBinsRect.y == 0 && selectedBinsRect.z == 0 && selectedBinsRect.w == 0) {
		//iTrackletFound[iCluster]=-1;
		return;
	}

	int phiBinsNum=selectedBinsRect.w-selectedBinsRect.y+1;


	if (phiBinsNum < 0) {
		phiBinsNum+=PhiBins;
	}
	
	int iPhiBin,iPhiCount;
	for(iPhiBin=selectedBinsRect.y,iPhiCount=0;iPhiCount<PhiBins;iPhiBin = ++iPhiBin == PhiBins ? 0 : iPhiBin, iPhiCount++){


		int firstBinIndex=getBinIndex(selectedBinsRect.x, iPhiBin);
		int maxBinIndex=firstBinIndex + selectedBinsRect.z - selectedBinsRect.x + 1;
		int firstRowClusterIndex = currentLayerIndexTable[firstBinIndex];
		int maxRowClusterIndex = currentLayerIndexTable[maxBinIndex];

		int iNextLayerCluster;
		for (iNextLayerCluster=firstRowClusterIndex; iNextLayerCluster <= maxRowClusterIndex;++iNextLayerCluster) {

			__global ClusterStruct *nextCluster=&nextLayerClusters[iNextLayerCluster];
			float deltaZ=myAbs(tanLambda*(nextCluster->rCoordinate-currentCluster->rCoordinate)+currentCluster->zCoordinate - nextCluster->zCoordinate);
			float deltaPhi=myAbs(currentCluster->phiCoordinate - nextCluster->phiCoordinate);

			int iTrackletLookupTableIndex=previousLayerTrackletsLookupTable[iCluster];

			if(deltaZ<TrackletMaxDeltaZThreshold[iLayer] && currentCluster->monteCarloId==nextCluster->monteCarloId && (deltaPhi<PhiCoordinateCut || myAbs(deltaPhi-TwoPi)<PhiCoordinateCut)){

				__global TrackletStruct* tracklet=&currentLayerTracklets[iCurrentTrackletPosition];
					//printf("iCurrentTrackletPosition=%d\n",iCurrentTrackletPosition);

				tracklet->firstClusterIndex=iCluster;
				tracklet->secondClusterIndex=iNextLayerCluster;
				tracklet->tanLambda=(currentCluster->zCoordinate - nextCluster->zCoordinate) / (currentCluster->rCoordinate - nextCluster->rCoordinate);
				tracklet->phiCoordinate= atan2(currentCluster->yCoordinate - nextCluster->yCoordinate, currentCluster->xCoordinate - nextCluster->xCoordinate);


					iCurrentTrackletPosition++;
				//}
				iLocalTrackletNum++;
			}
		}
	}
	iTrackletFound[iCluster]=iLocalTrackletNum;
	return;
}
