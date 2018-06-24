// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \file PrimaryVertexContext.cxx
/// \brief
///

#ifndef _TRAKINGITSU_INCLUDE_PRIMARY_VERTEX_CONTEXT_H_
#define _TRAKINGITSU_INCLUDE_PRIMARY_VERTEX_CONTEXT_H_


#include <sstream>
#include <iostream>
#include <algorithm>
#include "ITSReconstruction/CA/Definitions.h"
#include "ITSReconstruction/CA/gpu/PrimaryVertexContext.h"
#include "ITSReconstruction/CA/gpu/Context.h"

namespace o2
{
namespace ITS
{
namespace CA
{
namespace GPU
{

bool myClusterComparator(const ClusterStruct &lhs, const ClusterStruct &rhs)
{
        return lhs.indexTableBinIndex < rhs.indexTableBinIndex;
}


PrimaryVertexContext::PrimaryVertexContext()
{
  // Nothing to do
}

void  PrimaryVertexContext::sortClusters(int iLayer)
{
	std::sort(this->mClusters[iLayer],this->mClusters[iLayer]+this->iClusterSize[iLayer],myClusterComparator);
}

void PrimaryVertexContext::initialize(cl::Context oclContext)
{

	/*int iTrackletsFoundForLayer[]={0,0,0,0,0,0};
	this->bTrackletsFoundForLayer=cl::Buffer(
		oclContext,
		(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		o2::ITS::CA::Constants::ITS::TrackletsPerRoad*sizeof(int),
		(void *) iTrackletsFoundForLayer);
	 */
	/*int iCellsFoundForLayer[]={0,0,0,0,0};
	this->bCellsFoundForLayer=cl::Buffer(
		oclContext,
		(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		o2::ITS::CA::Constants::ITS::CellsPerRoad*sizeof(int),
		(void *) iCellsFoundForLayer);
		*/
}


void PrimaryVertexContext::newInitialize(
		const Event& event,
		float3 &mPrimaryVertex,
		const std::array<std::vector<Cluster>, Constants::ITS::LayersNumber>& clusters,
		std::array<std::vector<int>, Constants::ITS::CellsPerRoad>& mTrackletsLookupTable
		)
{
	int iTrackletsFoundForLayer[]={0,0,0,0,0,0};
	int iClusterSize[Constants::ITS::LayersNumber];
	u_int iClusterNum,iTrackletNum,iTrackletLookupSize,cellsLookupTableMemorySize;
	cl::Context clContext = GPU::Context::getInstance().getDeviceProperties().oclContext;
	cl::CommandQueue oclQueue=GPU::Context::getInstance().getDeviceProperties().oclQueue;
	cl::CommandQueue clQueues[Constants::ITS::LayersNumber];
	int tmpIndexTables[Constants::ITS::TrackletsPerRoad][Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins + 1];
	for(int i=0;i<Constants::ITS::LayersNumber;i++){
		clQueues[i]=GPU::Context::getInstance().getDeviceProperties().oclCommandQueues[i];
	}
	if(iInitialize==1){

		this->bPrimaryVertex= cl::Buffer(clContext,
			CL_MEM_READ_WRITE,
			sizeof(FLOAT3),
			NULL);

		this->bTrackletsFoundForLayer=cl::Buffer(
			clContext,
			(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
			o2::ITS::CA::Constants::ITS::TrackletsPerRoad * sizeof(int),
			(void *) iTrackletsFoundForLayer);

		for(int i=0;i<o2::ITS::CA::Constants::ITS::TrackletsPerRoad;i++){
			this->bLayerIndex[i]=cl::Buffer(
				clContext,
				(cl_mem_flags)CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				sizeof(int),
				(void *) &(i));
		}

		for (int iLayer { 0 }; iLayer < Constants::ITS::LayersNumber; ++iLayer){

			if(iLayer < Constants::ITS::CellsPerRoad - 1){
				this->bCellsLookupTable[iLayer]=cl::Buffer(clContext,
					CL_MEM_READ_WRITE,
					sizeof(int),
					NULL);
			}



			if(iLayer < Constants::ITS::CellsPerRoad){
				this->bTrackletsLookupTable[iLayer]=cl::Buffer(clContext,
					CL_MEM_READ_WRITE,
					sizeof(int),
					NULL);
			}

			if(iLayer > 0){
				this->bIndexTables[iLayer-1]=cl::Buffer(clContext,
					CL_MEM_READ_WRITE,
					(Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins + 1)*sizeof(int),
					NULL);
			}

			if(iLayer < Constants::ITS::TrackletsPerRoad) {
				this->bTracklets[iLayer]=cl::Buffer(clContext,
					CL_MEM_READ_WRITE,
					sizeof(Tracklet),
					NULL);
				this->iTrackletSize[iLayer]=0;
			}

			this->bClusters[iLayer]=cl::Buffer(clContext,
				CL_MEM_READ_WRITE,
				sizeof(Cluster),
				NULL);
			this->iClusterSize[iLayer]=0;

			if(iLayer < Constants::ITS::CellsPerRoad)
				this->iTrackletsLookupTableSize[iLayer]=0;

			if(iLayer < Constants::ITS::CellsPerRoad - 1)
				this->iCellsLookupTableSize[iLayer]=0;



		}


	}


	oclQueue.enqueueWriteBuffer(this->bPrimaryVertex,
			CL_FALSE,
			0,
			sizeof(FLOAT3),
			(void *) &mPrimaryVertex
			);


	for (int iLayer { 0 }; iLayer < Constants::ITS::LayersNumber; ++iLayer){

		iClusterNum=clusters[iLayer].size();
		iClusterSize[iLayer]=iClusterNum;


		if(this->iClusterSize[iLayer]<=iClusterNum){
			this->bClusters[iLayer]=cl::Buffer(clContext,
				CL_MEM_READ_WRITE,
				iClusterNum*sizeof(Cluster),
				NULL);
			this->iClusterSize[iLayer]=iClusterNum;
		}
		clQueues[iLayer].enqueueWriteBuffer(this->bClusters[iLayer],
			CL_FALSE,
			0,
			iClusterNum*sizeof(Cluster),
			(void*)clusters[iLayer].data());

		if(iLayer < Constants::ITS::TrackletsPerRoad) {
			iTrackletNum = std::ceil((Constants::Memory::TrackletsMemoryCoefficients[iLayer] * event.getLayer(iLayer).getClustersSize())
				   * event.getLayer(iLayer + 1).getClustersSize());
			if(this->iTrackletSize[iLayer]<=iTrackletNum){
				this->bTracklets[iLayer]=cl::Buffer(clContext,
					CL_MEM_READ_WRITE,
					iTrackletNum*sizeof(Tracklet),
					NULL);
				this->iTrackletSize[iLayer]=iTrackletNum;
			}
		}


		if(iLayer < Constants::ITS::CellsPerRoad) {
			iTrackletLookupSize=event.getLayer(iLayer + 1).getClustersSize();
			if(this->iTrackletsLookupTableSize[iLayer]<=iTrackletLookupSize){
				this->iTrackletsLookupTableSize[iLayer]=iTrackletLookupSize;
				this->bTrackletsLookupTable[iLayer]=cl::Buffer(clContext,
					CL_MEM_READ_WRITE,
					iTrackletLookupSize*sizeof(int),
					NULL);
			}
		}

		 if(iLayer < Constants::ITS::CellsPerRoad - 1) {
			cellsLookupTableMemorySize=std::ceil((Constants::Memory::TrackletsMemoryCoefficients[iLayer + 1] * event.getLayer(iLayer + 1).getClustersSize())
						* event.getLayer(iLayer + 2).getClustersSize());
			if(this->iCellsLookupTableSize[iLayer]<=cellsLookupTableMemorySize){
				this->iCellsLookupTableSize[iLayer]=cellsLookupTableMemorySize;
				this->bCellsLookupTable[iLayer]=cl::Buffer(clContext,
					CL_MEM_READ_WRITE,
					cellsLookupTableMemorySize*sizeof(int),
					NULL);
			}
		 }


		 if(iLayer >0){
			int previousBinIndex { 0 };
			tmpIndexTables[iLayer - 1][0] = 0;
			for (int iCluster { 0 }; iCluster < (int)iClusterNum; ++iCluster) {
				const int currentBinIndex { clusters[iLayer][iCluster].indexTableBinIndex };
				if (currentBinIndex > previousBinIndex) {
					for (int iBin { previousBinIndex + 1 }; iBin <= currentBinIndex; ++iBin) {
						tmpIndexTables[iLayer - 1][iBin] = iCluster;
					}
					previousBinIndex = currentBinIndex;
				}
			}

			for (int iBin { previousBinIndex + 1 }; iBin <= Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins;iBin++) {
				tmpIndexTables[iLayer - 1][iBin] = iClusterNum;
			}


			clQueues[iLayer].enqueueWriteBuffer(this->bIndexTables[iLayer-1],
				CL_FALSE,
				0,
				(Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins + 1)*sizeof(int),
				tmpIndexTables[iLayer-1]);
		 }

	}

	this->bClustersSize=cl::Buffer(clContext,
			CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
			Constants::ITS::LayersNumber*sizeof(int),
			iClusterSize);

	/*mGPUContext.bClustersSize=cl::Buffer(
		oclContext,
		(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		7*sizeof(int),
		(void *) mGPUContext.iClusterSize);
		*/


	/*std::array<std::array<int, Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins + 1>,
			                    Constants::ITS::TrackletsPerRoad> tmpIndexTables;

	compute::context boostContext =GPU::Context::getInstance().getBoostDeviceProperties().boostContext;
	compute::command_queue boostQueue =GPU::Context::getInstance().getBoostDeviceProperties().boostCommandQueues[0];
	compute::device device =GPU::Context::getInstance().getBoostDeviceProperties().boostDevice;
	u_int iClusterNum,iTrackletNum,iTrackletLookupSize,cellsLookupTableMemorySize;
	compute::command_queue boostQueues[Constants::ITS::LayersNumber];
	int iClusterSize[Constants::ITS::LayersNumber];
	if(iInitialize==1){


		for (int iLayer { 0 }; iLayer < Constants::ITS::LayersNumber; ++iLayer){

			this->boostClusters[iLayer]=compute::vector<Cluster>(1,boostContext);

			if(iLayer < Constants::ITS::TrackletsPerRoad) {
				this->boostTracklets[iLayer]=compute::vector<Tracklet>(1,boostContext);
			}
		}



		iInitialize=0;
	}

	boostQueue.enqueue_write_buffer((this->boostPrimaryVertex), 0, sizeof(FLOAT3), &mPrimaryVertex);

	for (int iLayer { 0 }; iLayer < Constants::ITS::LayersNumber; ++iLayer){

		if(this->boostClusters[iLayer].capacity()<=iClusterNum)
			this->boostClusters[iLayer].reserve(iClusterNum);
		//compute::copy_n(clusters[iLayer].begin(), iClusterNum, this->boostClusters[iLayer].begin(), boostQueue);
		compute::copy_async(clusters[iLayer].begin(),clusters[iLayer].begin().operator +=(iClusterNum),this->boostClusters[iLayer].begin(),boostQueues[iLayer]);

		if(iLayer < Constants::ITS::TrackletsPerRoad) {
			iTrackletNum = std::ceil((Constants::Memory::TrackletsMemoryCoefficients[iLayer] * event.getLayer(iLayer).getClustersSize())
				   * event.getLayer(iLayer + 1).getClustersSize());
			if(this->boostTracklets[iLayer].capacity()<=iTrackletNum)
				this->boostTracklets[iLayer].reserve(iTrackletNum);
		}





	}


*/

}


}
}
}
}

#endif /* _TRAKINGITSU_INCLUDE_PRIMARY_VERTEX_CONTEXT_H_ */
