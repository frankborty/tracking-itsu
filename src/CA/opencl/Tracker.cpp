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
	time_t t1,t2;
	t1=clock();
	try{

		cl::Context oclContext=GPU::Context::getInstance().getDeviceProperties().oclContext;
		cl::Device oclDevice=GPU::Context::getInstance().getDeviceProperties().oclDevice;
		//cl::CommandQueue oclCommandQueue=GPU::Context::getInstance().getDeviceProperties().oclQueue;
		PrimaryVertexContestStruct pvcStruct=(PrimaryVertexContestStruct)primaryVertexContext.mPrimaryVertexStruct;
		cl::Kernel oclKernel=GPU::Utils::CreateKernelFromFile(oclContext,oclDevice,"./src/kernel/computeLayerTracklets.cl","computeLayerTracklets");
		t1=clock();
		for(int i=0;i<6;i++)
				oclCommandqueues[i]=cl::CommandQueue(oclContext, oclDevice, 0);


		std::string deviceName;
		oclDevice.getInfo(CL_DEVICE_NAME,&deviceName);
		std::cout<< "Device: "<<deviceName<<std::endl;

	/*
		const char outputFileName[] = "LookupTable-ocl.txt";
		std::ofstream outFile;
		outFile.open((const char*)outputFileName);
	*/

		const char outputFileName[] = "oclTrackletsFound.txt";
		std::ofstream outFileTracklet;
		outFileTracklet.open((const char*)outputFileName);

		int workgroupSize=myWorkGroupSize;


		for (int iLayer { 0 }; iLayer< Constants::ITS::TrackletsPerRoad; ++iLayer) {

			outFileTracklet<<"From layer "<<iLayer<<" to "<<iLayer+1<<"\n";
			cl::CommandQueue oclCommandQueue=oclCommandqueues[iLayer];
			int clustersNum=primaryVertexContext.getClusters()[iLayer].size();
			int iNextLayerSize=primaryVertexContext.getClusters()[iLayer+1].size();

			cl::Buffer bTrackletClusterTable;
			if(iLayer>0){
				int *trackletClusterPreviousLayerTable=(int*)malloc(clustersNum*sizeof(int));
				memset(trackletClusterPreviousLayerTable,-1,clustersNum*sizeof(int));
				bTrackletClusterTable = cl::Buffer(
						oclContext,
						(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
						clustersNum*sizeof(int),
						(void *) &trackletClusterPreviousLayerTable[0]);
			}
			else{
				int fakeVector;
				bTrackletClusterTable = cl::Buffer(
						oclContext,
						(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
						sizeof(int),
						(void *) &fakeVector);
			}

			cl::Buffer bLayerID = cl::Buffer(
				oclContext,
				(cl_mem_flags)CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				sizeof(int),
				(void *) &iLayer);

			//buffer per l'atomic_add
			int iCurrentPosition=-1;
			cl::Buffer bCurrentTrackletPosition = cl::Buffer(
				oclContext,
				(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
				sizeof(int),
				(void *) &iCurrentPosition);

			cl::Buffer bCurrentLayerSize = cl::Buffer(
				oclContext,
				(cl_mem_flags)CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				sizeof(int),
				(void *) &clustersNum);

			cl::Buffer bNextLayerSize = cl::Buffer(
				oclContext,
				(cl_mem_flags)CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				sizeof(int),
				(void *) &iNextLayerSize);

			oclKernel.setArg(0, pvcStruct.bPrimaryVertex);
			oclKernel.setArg(1, pvcStruct.bClusters[iLayer]);
			oclKernel.setArg(2, pvcStruct.bClusters[iLayer+1]);
			oclKernel.setArg(3, pvcStruct.bIndexTable[iLayer]);
			oclKernel.setArg(4, pvcStruct.bTracklets[iLayer]);
			oclKernel.setArg(5, bLayerID);
			oclKernel.setArg(6, bCurrentTrackletPosition);
			oclKernel.setArg(7, bCurrentLayerSize);
			oclKernel.setArg(8, bNextLayerSize);
			oclKernel.setArg(9, bTrackletClusterTable);

			workgroupSize=128;
			while(true){
				if(pvcStruct.mClusters[iLayer].size%workgroupSize!=0)
					workgroupSize--;
				else
					break;
			}

				// Do the work

			t1=clock();
			oclCommandQueue.enqueueNDRangeKernel(
				oclKernel,
				cl::NullRange,
				cl::NDRange(clustersNum),
				cl::NDRange(workgroupSize));


			int* trackletsFound = (int *) oclCommandQueue.enqueueMapBuffer(
					bCurrentTrackletPosition,
					//bTrackletFound,
					CL_TRUE, // block
					CL_MAP_READ,
					0,
					sizeof(int)
			);
			(*trackletsFound)++;
/*
			TrackletStruct* output = (TrackletStruct *) oclCommandQueue.enqueueMapBuffer(
				pvcStruct.bTracklets[iLayer],
				CL_TRUE, // block
				CL_MAP_READ,
				0,
				(*trackletsFound) * sizeof(TrackletStruct)
			);


			if(iLayer>0){
				clogs::ScanProblem problem;
				problem.setType(clogs::TYPE_UINT);
				clogs::Scan scanner(oclContext, oclDevice, problem);

				scanner.enqueue(oclCommandQueue, bTrackletClusterTable, bTrackletClusterTable, clustersNum);

			}
*/			std::cout<<"TOTAL= "<<(*trackletsFound)<<std::endl;
	}
		//outFile<<"TOTAL= "<<totalSum<<"\n";
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
	t2=clock();
	const float diff=((float)t2-(float)t1)/(CLOCKS_PER_SEC/1000);


	std::cout << "Time = "<<diff<<std::endl;



	//std::cout<<"EXECUTION TIME = "<<diff<<std::endl;

}

//first version of computeLayerTracklet(cpu algorithm)
/*template<>
void TrackerTraits<true>::computeLayerTracklets(CA::PrimaryVertexContext& primaryVertexContext)
{
	//std::cout << "OCL_Tracker:computeLayerTracklets"<< std::endl;

try{

	int ksum=0;
	//creo il kernel
	cl::Context oclContext=GPU::Context::getInstance().getDeviceProperties().oclContext;
	cl::Device oclDevice=GPU::Context::getInstance().getDeviceProperties().oclDevice;
	//int warpSize=GPU::Context::getInstance().getDeviceProperties().warpSize;
	cl::CommandQueue oclCommandQueue=GPU::Context::getInstance().getDeviceProperties().oclQueue;
	PrimaryVertexContestStruct pvcStruct=(PrimaryVertexContestStruct)primaryVertexContext.mPrimaryVertexStruct;
	cl::Kernel oclKernel=GPU::Utils::CreateKernelFromFile(oclContext,oclDevice,"src/kernel/computeLayerTracklets.cl","computeLayerTracklets");
	int warpSize=myWorkGroupSize;
	std::string deviceName;
	oclDevice.getInfo(CL_DEVICE_NAME,&deviceName);
	//std::cout<< "Device: "<<deviceName<<std::endl;

	float deltaPhi;
	float deltaZ;
	int totalSum=0;
	deltaPhi=myPhiThreshold;
	deltaZ=myZThreshold;

	//std::string outputFileName = "trackletFound_deltaPHI" +std::to_string(deltaPhi)+"_deltaZ"+std::to_string(deltaZ)+".txt";

	const char outputFileNameLUT[] = "LookUpTableTracklet-ocl.txt";
	std::ofstream outFilePreLUT;
	outFilePreLUT.open((const char*)outputFileNameLUT);

	//const char outputFileName[] = "TrackletFound-ocl.txt";LookUpTableTracklet
	const char outputFileName[] = "NEW_WorkGroupSize-ocl.txt";
	std::ofstream outFilePre;
	outFilePre.open((const char*)outputFileName, std::ios_base::app);
	outFilePre << "DeltaPHI="<<deltaPhi<<"	WorkGroup="<<myWorkGroupSize<<"\n";
	std::cout << "DeltaPHI="<<deltaPhi<<"	WorkGroup="<<myWorkGroupSize<<std::endl;
	time_t t1,t2;
	t1=clock();
	for (int iLayer { 0 }; iLayer < Constants::ITS::TrackletsPerRoad; ++iLayer) {
		deltaZ=Constants::Thresholds::TrackletMaxDeltaZThreshold()[iLayer];
		//std::cout<< "OpenClKernel for compute trackelts between Layer "<<iLayer<<" and "<< iLayer+1<< std::endl;
		//outFilePre << "Trackelts between Layer "<<iLayer<<" and "<< iLayer+1 << "	-	deltaZ:"<<deltaZ<<	"\n";



		//dichiaro un fakeBuffer per la creazione dei tracklet tra i primi due layer
		int fakeVector[10];
		int iNextLayerSize=pvcStruct.mClusters[iLayer+1].size;
		int *iNumberTrackletFound;
		iNumberTrackletFound=(int*)malloc(pvcStruct.mClusters[iLayer].size*sizeof(int));


		//calcolo la grandezza massima del vettore di tracklet, lo creo e creo il buffer associto
		int maxTrackletNum=pvcStruct.mClusters[iLayer].size*iNextLayerSize/10;	//limito lo il numero di tracklet che posso trovare con un singolo cluste
																				//in questo modo posso creare i buffer senza problemi di memoria
																				//(dimensione ~250MB compresi i 16B della struttua ClusterStruct)


		TrackletStruct *trackletFound;
		trackletFound=(TrackletStruct*)malloc(maxTrackletNum*sizeof(TrackletStruct));
		for(int j=0;j<maxTrackletNum;j++)
			trackletFound[j].firstClusterIndex=-1;


		cl::Buffer bNumberTrackletFound = cl::Buffer(
			oclContext,
			(cl_mem_flags)CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
			pvcStruct.mClusters[iLayer].size*sizeof(int),
			(void *) iNumberTrackletFound);

		cl::Buffer bNextLayerSize = cl::Buffer(
			oclContext,
			(cl_mem_flags)CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			sizeof(int),
			(void *) &iNextLayerSize);

		cl::Buffer fakeBuff = cl::Buffer(
			oclContext,
			(cl_mem_flags)CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			10*sizeof(int),
			(void *) &fakeVector);

		cl::Buffer bLayerID = cl::Buffer(
			oclContext,
			(cl_mem_flags)CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			sizeof(int),
			(void *) &iLayer);

		//creo un buffer che tiene conto della posizione attuale nella quale inserire
		//le nuove tracklet trovate
		int iCurrentPosition=-1;
		cl::Buffer bCurrentPosition = cl::Buffer(
			oclContext,
			(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
			sizeof(int),
			(void *) &iCurrentPosition);


		oclKernel.setArg(0, pvcStruct.bPrimaryVertex);
		oclKernel.setArg(1, pvcStruct.bClusters[iLayer]);
		oclKernel.setArg(2, pvcStruct.bClusters[iLayer+1]);
		oclKernel.setArg(3, pvcStruct.bIndexTable[iLayer]);
		oclKernel.setArg(4, pvcStruct.bTrackletLookupTable[iLayer]);
		if(iLayer==0)
			oclKernel.setArg(5,fakeBuff);
		else
			oclKernel.setArg(5, pvcStruct.bTrackletLookupTable[iLayer-1]);
		oclKernel.setArg( 6, pvcStruct.bTracklets[iLayer]);//vettore nella quale vengono salvate le tracklet
		oclKernel.setArg( 7, bNumberTrackletFound);
		oclKernel.setArg( 8, bNextLayerSize);
		oclKernel.setArg( 9, bLayerID);
		oclKernel.setArg(10, bCurrentPosition);

		while(true){
			if(pvcStruct.mClusters[iLayer].size%warpSize!=0)
				warpSize--;
			else
				break;
		}

		outFilePre << "\nWorkGroup="<<myWorkGroupSize<<"\n";

			// Do the work
		oclCommandQueue.enqueueNDRangeKernel(
			oclKernel,
			cl::NullRange,
			cl::NDRange(pvcStruct.mClusters[iLayer].size),
			//cl::NullRange);
			cl::NullRange);
			//warpSize);


		int* iTrackletSize = (int *) oclCommandQueue.enqueueMapBuffer(
				bNumberTrackletFound,
				CL_TRUE, // block
				CL_MAP_READ,
				0,
				pvcStruct.mClusters[iLayer].size*sizeof(int));

		for(int j=0;j<pvcStruct.mClusters[iLayer].size;j++){
			if(iTrackletSize[j]>0){
				ksum+=iTrackletSize[j];
				iTrackletSize[j]=ksum;
			}
			outFilePreLUT<<"Layer:"<<iLayer<<"	index"<<j<<" -> "<<iTrackletSize[j]<<"\n";
		}

		TrackletStruct* output = (TrackletStruct *) oclCommandQueue.enqueueMapBuffer(
				pvcStruct.bTracklets[iLayer],
				//bTrackletFound,
				CL_TRUE, // block
				CL_MAP_READ,
				0,
				pvcStruct.mTracklets[iLayer].size * sizeof(TrackletStruct));

		int* trackletFoundNumber = (int *) oclCommandQueue.enqueueMapBuffer(
				bCurrentPosition,
				CL_TRUE, // block
				CL_MAP_READ,
				0,
				sizeof(int));

		int partialSum=*trackletFoundNumber;

		outFilePre << "Tracklets found = "<<partialSum<<"\n";
		totalSum+=partialSum;
		//Scrivi tracklets trovate su my_tracklets.txt
	//	int sum=0;
	//	std::ofstream outFilePre("my_tracklets.txt");
		//if(iLayer==0){
	//		for(int j=0;j<pvcStruct.mClusters[iLayer].size;j++){
	//			outFilePre << j << " = " << iTrackletSize[j] << "\n";
	//			sum+=iTrackletSize[j];
	//		}
		//}
//		std::cout<<"TOTAL="<<sum<<std::endl;

		}
		t2=clock();
		const float diff=((float)t2-(float)t1)/(CLOCKS_PER_SEC/1000);
		outFilePre << "Time = "<<diff<<"\n";
		outFilePre << "Total = "<<totalSum<<"\n\n";
		std::cout << "Total = "<<totalSum<<std::endl;
		std::cout << "TotalKSum = "<<ksum<<std::endl;
		outFilePreLUT << "Total = "<<ksum<<"\n";
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


	//std::cout<<"EXECUTION TIME = "<<diff<<std::endl;

}
*/

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
