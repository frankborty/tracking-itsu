__constant float CellMaxDeltaZThreshold[5]= { 0.2f, 0.4f, 0.5f, 0.6f, 3.0f } ;
__constant float TrackletMaxDeltaZThreshold[6]= { 0.1f, 0.1f, 0.3f, 0.3f, 0.3f, 0.3f }; //default
//__constant float TrackletMaxDeltaZThreshold[6]= { 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f };
__constant int ZBins=20;
__constant int PhiBins=20;
__constant float Pi=3.14159265359f;
__constant float TwoPi=2.0f * 3.14159265359f ;
__constant int UnusedIndex=-1 ;
__constant float CellMaxDeltaPhiThreshold=0.14f;
__constant float CellMaxDeltaTanLambdaThreshold=0.025f;
__constant float PhiCoordinateCut=0.3f;	//default
//__constant float PhiCoordinateCut=PHICUT;	//default

__constant float FloatMinThreshold = 1e-20f ;
__constant float ZCoordinateCut=0.5f;	//default
__constant float InversePhiBinSize=20 / (2.0f * 3.14159265359f) ;
__constant float LayersZCoordinate[7]={16.333f, 16.333f, 16.333f, 42.140f, 42.140f, 73.745f, 73.745f};
__constant float LayersRCoordinate[7]={2.33959f, 3.14076f, 3.91924f, 19.6213f, 24.5597f, 34.388f, 39.3329f};
__constant float InverseZBinSize[7]=  {0.5f * 20 / 16.333f, 0.5f * 20 / 16.333f, 0.5f * 20 / 16.333f,0.5f * 20 / 42.140f, 0.5f * 20 / 42.140f, 0.5f * 20 / 73.745f, 0.5f * 20 / 73.745f };
__constant float CellMaxDistanceOfClosestApproachThreshold[5]= { 0.05f, 0.04f, 0.05f, 0.2f, 0.4f } ;


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
			float x;
			float y;
	}Float2Struct;

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

Float3Struct crossProduct(const Float3Struct* firstVector,  const Float3Struct* secondVector)
{
	Float3Struct result;
	result.x=(firstVector->y * secondVector->z) - (firstVector->z * secondVector->y);
	result.y=(firstVector->z * secondVector->x) - (firstVector->x * secondVector->z);
    result.z=(firstVector->x * secondVector->y) - (firstVector->y * secondVector->x);
    return result;
}

__kernel void openClScan(__global int *in, __global int *out)
{
	int in_data;
	int i = get_global_id(0);
	const int numberOfClusterForCurrentLayer=get_global_size(0);

	in_data = in[i];
	out[i] = work_group_scan_exclusive_add(in_data);

}



//__device__ void computeLayerCells(PrimaryVertexContext& primaryVertexContext, const int layerIndex,
//    Vector<Cell>& cellsVector)
void computeLayerCells(
		Float3Struct* fPrimaryVertex,
		int *iCurrentLayer,
		int * iLayerTrackletSize, //store the number of tracklet found for each layer
		TrackletStruct* currentLayerTracklets,
		TrackletStruct* nextLayerTracklets,
		TrackletStruct* next2LayerTracklets,
		int* currentLayerTrackletsLookupTable,
		int * iCellsPerTrackletPreviousLayer //8
)



{
 // const int currentTrackletIndex = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
	const int currentTrackletIndex=1;/*get_global_id(0);*/
	const Float3Struct primaryVertex = *primaryVertex;
	int iLayer=*iCurrentLayer;
	int trackletCellsNum = 0;

	if (currentTrackletIndex < iLayerTrackletSize[iLayer]) {
		//const Tracklet& currentTracklet { primaryVertexContext.getTracklets()[layerIndex][currentTrackletIndex] };
		const TrackletStruct* currentTracklet=&currentLayerTracklets[currentTrackletIndex];

		//const int nextLayerClusterIndex { currentTracklet.secondClusterIndex };
		const int nextLayerClusterIndex=currentTracklet->secondClusterIndex;

		//const int nextLayerFirstTrackletIndex {	primaryVertexContext.getTrackletsLookupTable()[layerIndex][nextLayerClusterIndex]};
		const int nextLayerFirstTrackletIndex=currentLayerTrackletsLookupTable[nextLayerClusterIndex];

		//const int nextLayerTrackletsNum { static_cast<int>(primaryVertexContext.getTracklets()[layerIndex + 1].size()) };
		const int nextLayerTrackletsNum=iLayerTrackletSize[iLayer + 1];

		//if (primaryVertexContext.getTracklets()[layerIndex + 1][nextLayerFirstTrackletIndex].firstClusterIndex == nextLayerClusterIndex) {
		TrackletStruct* nextLayerFirstTracklet=&nextLayerTracklets[nextLayerFirstTrackletIndex];
		if (nextLayerFirstTracklet->firstClusterIndex == nextLayerClusterIndex) {
			//const Cluster& firstCellCluster {primaryVertexContext.getClusters()[layerIndex][currentTracklet.firstClusterIndex] };
			const ClusterStruct* firstCellCluster=&currentLayerTracklets[currentTracklet->firstClusterIndex] ;

			//const Cluster& secondCellCluster {primaryVertexContext.getClusters()[layerIndex + 1][currentTracklet.secondClusterIndex] };
			const ClusterStruct* secondCellCluster=&nextLayerTracklets[currentTracklet->secondClusterIndex];

			//const float firstCellClusterQuadraticRCoordinate { firstCellCluster.rCoordinate * firstCellCluster.rCoordinate };
			const float firstCellClusterQuadraticRCoordinate=firstCellCluster->rCoordinate * firstCellCluster->rCoordinate;

			//const float secondCellClusterQuadraticRCoordinate { secondCellCluster.rCoordinate * secondCellCluster.rCoordinate };
			const float secondCellClusterQuadraticRCoordinate=secondCellCluster->rCoordinate * secondCellCluster->rCoordinate;


			//const float3 firstDeltaVector { secondCellCluster.xCoordinate - firstCellCluster.xCoordinate,
			//secondCellCluster.yCoordinate - firstCellCluster.yCoordinate, secondCellClusterQuadraticRCoordinate
			//- firstCellClusterQuadraticRCoordinate };
			Float3Struct firstDeltaVector;
			firstDeltaVector.x=secondCellCluster->xCoordinate - firstCellCluster->xCoordinate;
			firstDeltaVector.y=secondCellCluster->yCoordinate - firstCellCluster->yCoordinate;
			firstDeltaVector.z=secondCellClusterQuadraticRCoordinate- firstCellClusterQuadraticRCoordinate;

			//for (int iNextLayerTracklet { nextLayerFirstTrackletIndex };
			//		iNextLayerTracklet < nextLayerTrackletsNum	&& primaryVertexContext.getTracklets()[layerIndex + 1][iNextLayerTracklet].firstClusterIndex== nextLayerClusterIndex; ++iNextLayerTracklet)
			//{
			for (int iNextLayerTracklet=nextLayerFirstTrackletIndex ;
					iNextLayerTracklet < nextLayerTrackletsNum	&& currentLayerTracklets[iNextLayerTracklet].firstClusterIndex== nextLayerClusterIndex;
					++iNextLayerTracklet){

				//const Tracklet& nextTracklet { primaryVertexContext.getTracklets()[layerIndex + 1][iNextLayerTracklet] };
				const TrackletStruct* nextTracklet=nextLayerTracklets[iNextLayerTracklet];

				//const float deltaTanLambda { MATH_ABS(currentTracklet.tanLambda - nextTracklet.tanLambda) };
				const float deltaTanLambda=myAbs(currentTracklet.tanLambda - nextTracklet.tanLambda);

				const float deltaPhi=myAbs(currentTracklet.phiCoordinate - nextTracklet.phiCoordinate);

				//if (deltaTanLambda < Constants::Thresholds::CellMaxDeltaTanLambdaThreshold && (deltaPhi < Constants::Thresholds::CellMaxDeltaPhiThreshold
				//|| MATH_ABS(deltaPhi - Constants::Math::TwoPi) < Constants::Thresholds::CellMaxDeltaPhiThreshold)) {
				if (deltaTanLambda < CellMaxDeltaTanLambdaThreshold && (deltaPhi < CellMaxDeltaPhiThreshold
					|| myAbs(deltaPhi - TwoPi) < CellMaxDeltaPhiThreshold)) {

					//const float averageTanLambda { 0.5f * (currentTracklet.tanLambda + nextTracklet.tanLambda) };
					const float averageTanLambda= 0.5f * (currentTracklet->tanLambda + nextTracklet->tanLambda) ;

					//const float directionZIntersection { -averageTanLambda * firstCellCluster.rCoordinate+ firstCellCluster.zCoordinate };
					const float directionZIntersection=-averageTanLambda * firstCellCluster->rCoordinate+ firstCellCluster->zCoordinate ;

					//const float deltaZ { MATH_ABS(directionZIntersection - primaryVertex.z) };
					const float deltaZ=myAbs(directionZIntersection - primaryVertex.z) ;

					//if (deltaZ < Constants::Thresholds::CellMaxDeltaZThreshold()[layerIndex]) {
					if (deltaZ < CellMaxDeltaZThreshold[iLayer]) {

						//const Cluster& thirdCellCluster {primaryVertexContext.getClusters()[layerIndex + 2][nextTracklet.secondClusterIndex] };
						const ClusterStruct* thirdCellCluster=next2LayerTracklets[nextTracklet->secondClusterIndex];

						const float thirdCellClusterQuadraticRCoordinate=thirdCellCluster->rCoordinate	* thirdCellCluster->rCoordinate ;

						//const float3 secondDeltaVector { thirdCellCluster.xCoordinate - firstCellCluster.xCoordinate,thirdCellCluster.yCoordinate - firstCellCluster.yCoordinate, thirdCellClusterQuadraticRCoordinate- firstCellClusterQuadraticRCoordinate };
						const Float3Struct secondDeltaVector;
						secondDeltaVector.x=thirdCellCluster->xCoordinate - firstCellCluster->xCoordinate;
						secondDeltaVector.y=thirdCellCluster->yCoordinate - firstCellCluster->yCoordinate;
						secondDeltaVector.z=thirdCellClusterQuadraticRCoordinate- firstCellClusterQuadraticRCoordinate;

						//float3 cellPlaneNormalVector { MathUtils::crossProduct(firstDeltaVector, secondDeltaVector) };
						Float3Struct cellPlaneNormalVector=crossProduct(&firstDeltaVector, &secondDeltaVector);

						//const float vectorNorm { std::sqrt(
						//cellPlaneNormalVector.x * cellPlaneNormalVector.x + cellPlaneNormalVector.y * cellPlaneNormalVector.y
						//+ cellPlaneNormalVector.z * cellPlaneNormalVector.z) };
						const float vectorNorm=sqrt(cellPlaneNormalVector.x * cellPlaneNormalVector.x + cellPlaneNormalVector.y * cellPlaneNormalVector.y
								+ cellPlaneNormalVector.z * cellPlaneNormalVector.z);



						//if (!(vectorNorm < Constants::Math::FloatMinThreshold
						//|| MATH_ABS(cellPlaneNormalVector.z) < Constants::Math::FloatMinThreshold)) {
						if (!(vectorNorm < FloatMinThreshold || myAbs(cellPlaneNormalVector.z) < FloatMinThreshold)) {
							const float inverseVectorNorm = 1.0f / vectorNorm ;

							//const float3 normalizedPlaneVector { cellPlaneNormalVector.x * inverseVectorNorm, cellPlaneNormalVector.y
							//* inverseVectorNorm, cellPlaneNormalVector.z * inverseVectorNorm };

							const Float3Struct normalizedPlaneVector = {cellPlaneNormalVector.x * inverseVectorNorm, cellPlaneNormalVector.y
								*inverseVectorNorm, cellPlaneNormalVector.z * inverseVectorNorm };

							//const float planeDistance { -normalizedPlaneVector.x * (secondCellCluster.xCoordinate - primaryVertex.x)
							//- (normalizedPlaneVector.y * secondCellCluster.yCoordinate - primaryVertex.y)
							//- normalizedPlaneVector.z * secondCellClusterQuadraticRCoordinate };
							const float planeDistance = -normalizedPlaneVector.x * (secondCellCluster->xCoordinate - primaryVertex.x)
								- (normalizedPlaneVector.y * secondCellCluster->yCoordinate - primaryVertex.y)
								- normalizedPlaneVector.z * secondCellClusterQuadraticRCoordinate ;



							//const float normalizedPlaneVectorQuadraticZCoordinate { normalizedPlaneVector.z * normalizedPlaneVector.z };
							const float normalizedPlaneVectorQuadraticZCoordinate = normalizedPlaneVector.z * normalizedPlaneVector.z ;


							//const float cellTrajectoryRadius { MATH_SQRT(
							//(1.0f - normalizedPlaneVectorQuadraticZCoordinate - 4.0f * planeDistance * normalizedPlaneVector.z)
							/// (4.0f * normalizedPlaneVectorQuadraticZCoordinate)) };

							const float cellTrajectoryRadius = sqrt(
								(1.0f - normalizedPlaneVectorQuadraticZCoordinate - 4.0f * planeDistance * normalizedPlaneVector.z)
								/ (4.0f * normalizedPlaneVectorQuadraticZCoordinate)) ;



							//const float2 circleCenter { -0.5f * normalizedPlaneVector.x / normalizedPlaneVector.z, -0.5f
							//* normalizedPlaneVector.y / normalizedPlaneVector.z };
							const Float2Struct circleCenter ={ -0.5f * normalizedPlaneVector.x / normalizedPlaneVector.z, -0.5f
								* normalizedPlaneVector.y / normalizedPlaneVector.z };


							//const float distanceOfClosestApproach { MATH_ABS(
							//cellTrajectoryRadius - MATH_SQRT(circleCenter.x * circleCenter.x + circleCenter.y * circleCenter.y)) };
							const float distanceOfClosestApproach = myAbs(
								cellTrajectoryRadius - sqrt(circleCenter.x * circleCenter.x + circleCenter.y * circleCenter.y)) ;


							if (distanceOfClosestApproach	<= CellMaxDistanceOfClosestApproachThreshold[iLayer]) {
								//new cell found
								++trackletCellsNum;
							}
						}
					}
				}
			}

			//if (iLayer > 0) {
			//	primaryVertexContext.getCellsPerTrackletTable()[layerIndex - 1][currentTrackletIndex] = trackletCellsNum;
			//}
			if(trackletCellsNum>0) {
			// 	printf("iLayer = %d,\tclusterTrackletsNum = %d,\tcurrentClusterIndex = %d\n",iLayer,clusterTrackletsNum,currentClusterIndex);
				iCellsPerTrackletPreviousLayer[currentTrackletIndex] = trackletCellsNum;
			}
			else{
				iCellsPerTrackletPreviousLayer[currentTrackletIndex] = 0;
			}


		}
	}
}




