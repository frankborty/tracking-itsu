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
/// \file CAGPUTracker.cu
/// \brief
///

#include <Context.h>
#include <ITSReconstruction/CA/Cell.h>
#include <ITSReconstruction/CA/Constants.h>
#include <ITSReconstruction/CA/Tracker.h>
#include <ITSReconstruction/CA/Tracklet.h>
#include "ITSReconstruction/CA/Definitions.h"
#include <StructGPUPrimaryVertex.h>
#include <Utils.h>
#include <Vector.h>
#include <stdexcept>
#include <string>

#if TRACKINGITSU_OCL_MODE
#include <CL/cl.hpp>
#include <clogs/clogs.h>
#include <clogs/scan.h>
#include "ITSReconstruction/CA/gpu/myThresholds.h"
#endif

#if TRACKINGITSU_CUDA_MODE
#include "ITSReconstruction/CA/gpu/Vector.h"
#include "ITSReconstruction/CA/gpu/Utils.h"
#endif


namespace o2
{
namespace ITS
{
namespace CA
{
namespace GPU
{



void computeLayerTracklets(PrimaryVertexContext &primaryVertexContext, const int layerIndex,
    Vector<Tracklet>& trackletsVector)
{

}

void computeLayerCells(PrimaryVertexContext& primaryVertexContext, const int layerIndex,
    Vector<Cell>& cellsVector)
{
}

void layerTrackletsKernel(PrimaryVertexContext& primaryVertexContext, const int layerIndex,
    Vector<Tracklet> trackletsVector)
{
  computeLayerTracklets(primaryVertexContext, layerIndex, trackletsVector);
}

void sortTrackletsKernel(PrimaryVertexContext& primaryVertexContext, const int layerIndex,
    Vector<Tracklet> tempTrackletArray)
{

}

void layerCellsKernel(PrimaryVertexContext& primaryVertexContext, const int layerIndex,
    Vector<Cell> cellsVector)
{
//  computeLayerCells(primaryVertexContext, layerIndex, cellsVector);
}



void sortCellsKernel(PrimaryVertexContext& primaryVertexContext, const int layerIndex,
    Vector<Cell> tempCellsArray)
{

}

} /// End of GPU namespace

template<>
void TrackerTraits<true>::computeLayerTracklets(CA::PrimaryVertexContext& primaryVertexContext)
{
	//std::cout << "OCL_Tracker:computeLayerTracklets"<< std::endl;
	cl::CommandQueue oclCommandqueues[6];
	cl::Buffer bLayerID;
	cl::Buffer bTrackletLookUpTable;
	cl::CommandQueue oclCommandQueue;
	int *firstLayerLookUpTable;
	int clustersNum;
	time_t t1,t2,t0;
	time_t tx,ty;
	int* trackletsFound;
	int workgroupSize=5*32;
	try{

		cl::Context oclContext=GPU::Context::getInstance().getDeviceProperties().oclContext;
		cl::Device oclDevice=GPU::Context::getInstance().getDeviceProperties().oclDevice;
		//cl::CommandQueue oclCommandQueue=GPU::Context::getInstance().getDeviceProperties().oclQueue;
		PrimaryVertexContestStruct pvcStruct=(PrimaryVertexContestStruct)primaryVertexContext.mPrimaryVertexStruct;
		//must be move to oclContext create
		cl::Kernel oclCountKernel=GPU::Utils::CreateKernelFromFile(oclContext,oclDevice,"./src/kernel/computeLayerTracklets.cl","countLayerTracklets");
		cl::Kernel oclComputeKernel=GPU::Utils::CreateKernelFromFile(oclContext,oclDevice,"./src/kernel/computeLayerTracklets.cl","computeLayerTracklets");

		//int warpSize=GPU::Context::getInstance().getDeviceProperties().warpSize;

		for(int i=0;i<6;i++)
			oclCommandqueues[i]=cl::CommandQueue(oclContext, oclDevice, CL_QUEUE_PROFILING_ENABLE );

		clustersNum=primaryVertexContext.getClusters()[0].size();
		firstLayerLookUpTable=(int*)malloc(clustersNum*sizeof(int));
		memset(firstLayerLookUpTable,-1,clustersNum*sizeof(int));
		bTrackletLookUpTable = cl::Buffer(
				oclContext,
				(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
				clustersNum*sizeof(int),
				(void *) &firstLayerLookUpTable[0]);



		std::string deviceName;
		oclDevice.getInfo(CL_DEVICE_NAME,&deviceName);
		std::cout<< "Device: "<<deviceName<<std::endl;
/*
		const char outputFileName[] = "LookupTable-ocl.txt";
		std::ofstream outFile;
		outFile.open((const char*)outputFileName);
*/
/*
		const char outputTrackletFileName[] = "oclTrackletsFound.txt";
		std::ofstream outFileTracklet;
		outFileTracklet.open((const char*)outputTrackletFileName);
*/		t0=t1=clock();
		for (int iLayer{ 0 }; iLayer<Constants::ITS::TrackletsPerRoad; ++iLayer) {
			tx=clock();

  			clustersNum=primaryVertexContext.getClusters()[iLayer].size();


			oclCountKernel.setArg(0, pvcStruct.bPrimaryVertex);
			oclCountKernel.setArg(1, pvcStruct.bClusters[iLayer]);
			oclCountKernel.setArg(2, pvcStruct.bClusters[iLayer+1]);
			oclCountKernel.setArg(3, pvcStruct.bIndexTable[iLayer]);
			oclCountKernel.setArg(4, pvcStruct.bLayerId[iLayer]);
			oclCountKernel.setArg(5, pvcStruct.bTrackletsFoundForLayer);
			oclCountKernel.setArg(6, pvcStruct.bLayerClustersSize);
			if(iLayer==0)
				oclCountKernel.setArg(7, bTrackletLookUpTable);
			else
				oclCountKernel.setArg(7, pvcStruct.bTrackletLookupTable[iLayer-1]);

			/*while(true){
				if((clustersNum % workgroupSize)!=0)
					workgroupSize--;
				else
					break;
			}*/
			int pseudoClusterNumber=clustersNum;
			if((clustersNum % workgroupSize)!=0){
				int mult=clustersNum/workgroupSize;
				pseudoClusterNumber=(mult+1)*workgroupSize;
			}

			oclCommandqueues[iLayer].enqueueNDRangeKernel(
					oclCountKernel,
				cl::NullRange,
				cl::NDRange(pseudoClusterNumber),
				cl::NDRange(workgroupSize));
				//cl::NullRange);




			trackletsFound = (int *) oclCommandqueues[iLayer].enqueueMapBuffer(
					pvcStruct.bTrackletsFoundForLayer,
					CL_TRUE, // block
					CL_MAP_READ,
					0,
					6*sizeof(int)
			);
			trackletsFound[iLayer]++;
			ty=clock();
			//float time = ((float) ty - (float) tx) / (CLOCKS_PER_SEC / 1000);
			//std::cout<< "\tLayer " << iLayer <<" time = "<<time<<" ms" <<"\tWG = " <<workgroupSize<<"\tclusterNr = "<<clustersNum<<"\tpseudoClusterNr = "<<pseudoClusterNumber<<std::endl;
		}
		t2 = clock();
		//float countTrack = ((float) t2 - (float) t1) / (CLOCKS_PER_SEC / 1000);
		//std::cout<< "countTrack time " << countTrack <<" ms" << std::endl;

		//scan
		//std::cout<<"scan and sort"<<std::endl;
		t1=clock();
		for (int iLayer { 0 }; iLayer<Constants::ITS::TrackletsPerRoad; ++iLayer) {
			tx=clock();
			clustersNum=primaryVertexContext.getClusters()[iLayer].size();
			if(iLayer==0){
				clogs::ScanProblem problem;
				problem.setType(clogs::TYPE_UINT);
				clogs::Scan scanner(oclContext, oclDevice, problem);
				oclCommandqueues[iLayer].finish();
				scanner.enqueue(oclCommandqueues[iLayer], bTrackletLookUpTable, bTrackletLookUpTable, clustersNum);
			}
			else{
				clogs::ScanProblem problem;
				problem.setType(clogs::TYPE_UINT);
				clogs::Scan scanner(oclContext, oclDevice, problem);
				oclCommandqueues[iLayer].finish();
				scanner.enqueue(oclCommandqueues[iLayer], pvcStruct.bTrackletLookupTable[iLayer-1], pvcStruct.bTrackletLookupTable[iLayer-1], clustersNum);
			}
			ty=clock();
			//float time = ((float) ty - (float) tx) / (CLOCKS_PER_SEC / 1000);
			//std::cout<< "\t " << iLayer <<" time = "<<time<<" ms" << std::endl;
		}
		t2 = clock();
		//float scanTrack = ((float) t2 - (float) t1) / (CLOCKS_PER_SEC / 1000);
		//std::cout<< "scanTrack time " << scanTrack <<" ms" << std::endl;

		//calcolo le tracklet
		//std::cout<<"calcolo le tracklet"<<std::endl;
		t1=clock();
		for (int iLayer{ 0 }; iLayer<Constants::ITS::TrackletsPerRoad; ++iLayer) {
			tx=clock();
			//outFileTracklet<<"\nTracklets between Layer "<<iLayer<<" and "<<iLayer+1<<"\n";
			oclCommandqueues[iLayer].finish();
			clustersNum=primaryVertexContext.getClusters()[iLayer].size();

			oclComputeKernel.setArg(0, pvcStruct.bPrimaryVertex);
			oclComputeKernel.setArg(1, pvcStruct.bClusters[iLayer]);
			oclComputeKernel.setArg(2, pvcStruct.bClusters[iLayer+1]);
			oclComputeKernel.setArg(3, pvcStruct.bIndexTable[iLayer]);
			oclComputeKernel.setArg(4, pvcStruct.bTracklets[iLayer]);
			oclComputeKernel.setArg(5, pvcStruct.bLayerId[iLayer]);
			oclComputeKernel.setArg(6, pvcStruct.bTrackletsFoundForLayer);
			oclComputeKernel.setArg(7, pvcStruct.bLayerClustersSize);
			if(iLayer==0)
				oclComputeKernel.setArg(8, bTrackletLookUpTable);
			else
				oclComputeKernel.setArg(8, pvcStruct.bTrackletLookupTable[iLayer-1]);

			int pseudoClusterNumber=clustersNum;
			if((clustersNum % workgroupSize)!=0){
				int mult=clustersNum/workgroupSize;
				pseudoClusterNumber=(mult+1)*workgroupSize;
			}

				// Do the work
			oclCommandqueues[iLayer].enqueueNDRangeKernel(
					oclComputeKernel,
				cl::NullRange,
				cl::NDRange(pseudoClusterNumber),
				cl::NDRange(workgroupSize));
				//cl::NullRange);

/*
			TrackletStruct* output = (TrackletStruct *) oclCommandqueues[iLayer].enqueueMapBuffer(
				pvcStruct.bTracklets[iLayer],
				CL_TRUE, // block
				CL_MAP_READ,
				0,
				trackletsFound[iLayer] * sizeof(TrackletStruct)
			);
*/
			/*for(int i=0;i<trackletsFound[iLayer];i++)
				outFileTracklet<<output[i].firstClusterIndex<<"\t"<<output[i].secondClusterIndex<<"\t"<<output[i].phiCoordinate<<"\t"<<output[i].tanLambda<<"\n";
*/

			ty=clock();
			//float time = ((float) ty - (float) tx) / (CLOCKS_PER_SEC / 1000);
			//std::cout<< "\tLayer " << iLayer <<" time = "<<time<<" ms" <<"\tWG = " <<workgroupSize<<std::endl;
		}
		t2 = clock();
		//float findTrack = ((float) t2 - (float) t1) / (CLOCKS_PER_SEC / 1000);
		//std::cout<< "findTrack time " << findTrack <<" ms" << std::endl;

	}
	catch(const cl::Error &err){
			std::string errString=GPU::Utils::OCLErr_code(err.err());
			std::cout<< errString << std::endl;
			throw std::runtime_error { errString };
	}
	catch(...){
		std::cout<<"Errore non opencl"<<std::endl;
		throw std::runtime_error {"ERRORE NON OPENCL"};
	}

	t2 = clock();
	//float diff = ((float) t2 - (float) t0) / (CLOCKS_PER_SEC / 1000);
	//std::cout<< "compute tracklets time " << diff <<" ms" << std::endl;
}




template<>
void TrackerTraits<true>::computeLayerCells(CA::PrimaryVertexContext& primaryVertexContext)
{
//  std::array<size_t, Constants::ITS::CellsPerRoad - 1> tempSize;
//  std::array<int, Constants::ITS::CellsPerRoad - 1> trackletsNum;
//  std::array<int, Constants::ITS::CellsPerRoad - 1> cellsNum;
//  std::array<GPU::Stream, Constants::ITS::CellsPerRoad> streamArray;
//
//  for (int iLayer { 0 }; iLayer < Constants::ITS::CellsPerRoad - 1; ++iLayer) {
//
//    tempSize[iLayer] = 0;
//    trackletsNum[iLayer] = primaryVertexContext.getDeviceTracklets()[iLayer + 1].getSizeFromDevice();
//    const int cellsNum { static_cast<int>(primaryVertexContext.getDeviceCells()[iLayer + 1].capacity()) };
//    primaryVertexContext.getTempCellArray()[iLayer].reset(cellsNum);
//
//    cub::DeviceScan::ExclusiveSum(static_cast<void *>(NULL), tempSize[iLayer],
//        primaryVertexContext.getDeviceCellsPerTrackletTable()[iLayer].get(),
//        primaryVertexContext.getDeviceCellsLookupTable()[iLayer].get(), trackletsNum[iLayer]);
//
//    primaryVertexContext.getTempTableArray()[iLayer].reset(static_cast<int>(tempSize[iLayer]));
//  }
//
//  cudaDeviceSynchronize();
//
//  for (int iLayer { 0 }; iLayer < Constants::ITS::CellsPerRoad; ++iLayer) {
//
//    const GPU::DeviceProperties& deviceProperties = GPU::Context::getInstance().getDeviceProperties();
//    const int trackletsSize = primaryVertexContext.getDeviceTracklets()[iLayer].getSizeFromDevice();
//    dim3 threadsPerBlock { GPU::Utils::Host::getBlockSize(trackletsSize) };
//    dim3 blocksGrid { GPU::Utils::Host::getBlocksGrid(threadsPerBlock, trackletsSize) };
//
//    if(iLayer == 0) {
//
//      GPU::layerCellsKernel<<< blocksGrid, threadsPerBlock, 0, streamArray[iLayer].get() >>>(primaryVertexContext.getDeviceContext(),
//          iLayer, primaryVertexContext.getDeviceCells()[iLayer].getWeakCopy());
//
//    } else {
//
//      GPU::layerCellsKernel<<< blocksGrid, threadsPerBlock, 0, streamArray[iLayer].get() >>>(primaryVertexContext.getDeviceContext(),
//          iLayer, primaryVertexContext.getTempCellArray()[iLayer - 1].getWeakCopy());
//    }
//
//    cudaError_t error = cudaGetLastError();
//
//    if (error != cudaSuccess) {
//
//      std::ostringstream errorString { };
//      errorString << "CUDA API returned error [" << cudaGetErrorString(error) << "] (code " << error << ")"
//          << std::endl;
//
//      throw std::runtime_error { errorString.str() };
//    }
//  }
//
//  cudaDeviceSynchronize();
//
//  for (int iLayer { 0 }; iLayer < Constants::ITS::CellsPerRoad - 1; ++iLayer) {
//
//    cellsNum[iLayer] = primaryVertexContext.getTempCellArray()[iLayer].getSizeFromDevice();
//    primaryVertexContext.getDeviceCells()[iLayer + 1].resize(cellsNum[iLayer]);
//
//    cub::DeviceScan::ExclusiveSum(static_cast<void *>(primaryVertexContext.getTempTableArray()[iLayer].get()), tempSize[iLayer],
//        primaryVertexContext.getDeviceCellsPerTrackletTable()[iLayer].get(),
//        primaryVertexContext.getDeviceCellsLookupTable()[iLayer].get(), trackletsNum[iLayer],
//        streamArray[iLayer + 1].get());
//
//    dim3 threadsPerBlock { GPU::Utils::Host::getBlockSize(trackletsNum[iLayer]) };
//    dim3 blocksGrid { GPU::Utils::Host::getBlocksGrid(threadsPerBlock, trackletsNum[iLayer]) };
//
//    GPU::sortCellsKernel<<< blocksGrid, threadsPerBlock, 0, streamArray[iLayer + 1].get() >>>(primaryVertexContext.getDeviceContext(),
//        iLayer + 1, primaryVertexContext.getTempCellArray()[iLayer].getWeakCopy());
//
//    cudaError_t error = cudaGetLastError();
//
//    if (error != cudaSuccess) {
//
//      std::ostringstream errorString { };
//      errorString << "CUDA API returned error [" << cudaGetErrorString(error) << "] (code " << error << ")"
//          << std::endl;
//
//      throw std::runtime_error { errorString.str() };
//    }
//  }
//
//  cudaDeviceSynchronize();
//
//  for (int iLayer { 0 }; iLayer < Constants::ITS::CellsPerRoad; ++iLayer) {
//
//    int cellsSize;
//
//    if (iLayer == 0) {
//
//      cellsSize = primaryVertexContext.getDeviceCells()[iLayer].getSizeFromDevice();
//
//    } else {
//
//      cellsSize = cellsNum[iLayer - 1];
//
//      primaryVertexContext.getDeviceCellsLookupTable()[iLayer - 1].copyIntoVector(
//          primaryVertexContext.getCellsLookupTable()[iLayer - 1], trackletsNum[iLayer - 1]);
//    }
//
//    primaryVertexContext.getDeviceCells()[iLayer].copyIntoVector(primaryVertexContext.getCells()[iLayer], cellsSize);
//  }
}

}
}
}
