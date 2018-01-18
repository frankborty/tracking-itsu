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

#if 0
#define PRINT_EXECUTION_TIME
#endif


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
//#include <CL/cl.hpp>
#include "ITSReconstruction/CA/gpu/myThresholds.h"
	#if CLOGS
	#include <clogs/clogs.h>
	#include <clogs/scan.h>
	#endif
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
	cl::CommandQueue oclCPUCommandqueues[6];
	cl::Buffer bLayerID;
	cl::Buffer bTrackletLookUpTable;
	cl::CommandQueue oclCommandQueue;
	int *firstLayerLookUpTable;
	int clustersNum;
	time_t t1;
	//time_t t2;
	//time_t tx,ty;
	int* trackletsFound;
	int workgroupSize=5*32;
	int totalTrackletsFound=0;
	try{

		cl::Context oclContext=GPU::Context::getInstance().getDeviceProperties().oclContext;
		cl::Device oclDevice=GPU::Context::getInstance().getDeviceProperties().oclDevice;
		cl::CommandQueue oclCommandQueue=GPU::Context::getInstance().getDeviceProperties().oclQueue;

		std::string deviceName;
		oclDevice.getInfo(CL_DEVICE_NAME,&deviceName);
		std::cout<< "Device: "<<deviceName<<std::endl;

		//must be move to oclContext create
/*		cl::Kernel oclCountKernel=GPU::Utils::CreateKernelFromFile(oclContext,oclDevice,"./src/kernel/computeLayerTracklets.cl","countLayerTracklets");
		cl::Kernel oclComputeKernel=GPU::Utils::CreateKernelFromFile(oclContext,oclDevice,"./src/kernel/computeLayerTracklets.cl","computeLayerTracklets");
		cl::Kernel oclTestKernel=GPU::Utils::CreateKernelFromFile(oclContext,oclDevice,"./src/kernel/computeLayerTracklets.cl","openClScan");
		//
*/		cl::Kernel oclCountKernel=GPU::Context::getInstance().getDeviceProperties().oclCountKernel;
		cl::Kernel oclComputeKernel=GPU::Context::getInstance().getDeviceProperties().oclComputeKernel;
		cl::Kernel oclTestKernel=GPU::Context::getInstance().getDeviceProperties().oclTestKernel;

		//int warpSize=GPU::Context::getInstance().getDeviceProperties().warpSize;

		for(int i=0;i<6;i++){
			oclCommandqueues[i]=cl::CommandQueue(oclContext, oclDevice, 0);
		}
		//clustersNum=primaryVertexContext.getClusters()[0].size();
		clustersNum=primaryVertexContext.openClPrimaryVertexContext.iClusterSize[0];

		firstLayerLookUpTable=(int*)malloc(clustersNum*sizeof(int));
		memset(firstLayerLookUpTable,-1,clustersNum*sizeof(int));
		bTrackletLookUpTable = cl::Buffer(
				oclContext,
				(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
				clustersNum*sizeof(int),
				(void *) &firstLayerLookUpTable[0]);



/*
		const char outputFileName[] = "../LookupTable-ocl.txt";
		std::ofstream outFileLookUp;
		outFileLookUp.open((const char*)outputFileName);

		const char outputTrackletFileName[] = "../oclTrackletsFound.txt";
		std::ofstream outFileTracklet;
		outFileTracklet.open((const char*)outputTrackletFileName);
*/

		t1=clock();
		for (int iLayer{ 0 }; iLayer<Constants::ITS::TrackletsPerRoad; ++iLayer) {
			//tx=clock();

  			clustersNum=primaryVertexContext.openClPrimaryVertexContext.iClusterSize[iLayer];

			oclCountKernel.setArg(0, primaryVertexContext.openClPrimaryVertexContext.bPrimaryVertex);
			oclCountKernel.setArg(1, primaryVertexContext.openClPrimaryVertexContext.bClusters[iLayer]);
			oclCountKernel.setArg(2, primaryVertexContext.openClPrimaryVertexContext.bClusters[iLayer+1]);
			oclCountKernel.setArg(3, primaryVertexContext.openClPrimaryVertexContext.bIndexTables[iLayer]);
			oclCountKernel.setArg(4, primaryVertexContext.openClPrimaryVertexContext.bLayerIndex[iLayer]);
			oclCountKernel.setArg(5, primaryVertexContext.openClPrimaryVertexContext.bTrackletsFoundForLayer);
			oclCountKernel.setArg(6, primaryVertexContext.openClPrimaryVertexContext.bClustersSize);
			if(iLayer==0)
				oclCountKernel.setArg(7, bTrackletLookUpTable);
			else
				oclCountKernel.setArg(7, primaryVertexContext.openClPrimaryVertexContext.bTrackletsLookupTable[iLayer-1]);

			int pseudoClusterNumber=clustersNum;
			if((clustersNum % workgroupSize)!=0){
				int mult=clustersNum/workgroupSize;
				pseudoClusterNumber=(mult+1)*workgroupSize;
			}

			time_t tx=clock();
			oclCommandqueues[iLayer].enqueueNDRangeKernel(
				oclCountKernel,
				cl::NullRange,
				cl::NDRange(pseudoClusterNumber),
				cl::NDRange(workgroupSize));
				//cl::NullRange);
			time_t ty=clock();
			float countTrack = ((float) ty - (float) tx) / (CLOCKS_PER_SEC / 1000);
			//std::cout<< "["<<iLayer<<"]countTrack time " << countTrack <<" ms" << std::endl;
/*
			oclCommandqueues[iLayer].finish();
			trackletsFound = (int *) oclCommandqueues[iLayer].enqueueMapBuffer(
					primaryVertexContext.openClPrimaryVertexContext.bTrackletsFoundForLayer,
					CL_TRUE, // block
					CL_MAP_READ,
					0,
					6*sizeof(int)
			);
			trackletsFound[iLayer]++;
			totalTrackletsFound+=trackletsFound[iLayer]-1;

			std::cout<<"Tracklet layer "<<iLayer<<" = "<<trackletsFound[iLayer]<<std::endl;
*/
			//ty=clock();
			//float time = ((float) ty - (float) tx) / (CLOCKS_PER_SEC / 1000);
			//std::cout<< "\tLayer " << iLayer <<" time = "<<time<<" ms" <<"\tWG = " <<workgroupSize<<"\tclusterNr = "<<clustersNum<<"\tpseudoClusterNr = "<<pseudoClusterNumber<<std::endl;
		}

#ifdef PRINT_EXECUTION_TIME
		t2 = clock();
		float countTrack = ((float) t2 - (float) t1) / (CLOCKS_PER_SEC / 1000);
		std::cout<< "countTrack time " << countTrack <<" ms" << std::endl;
#endif

#if CLOGS
		//scan
		t1=clock();
		for (int iLayer { 0 }; iLayer<Constants::ITS::TrackletsPerRoad; ++iLayer) {
			//tx=clock();
			//outFileLookUp<<"From layer "<<iLayer<<" to "<<iLayer+1<<"\n";
			clustersNum=primaryVertexContext.openClPrimaryVertexContext.iClusterSize[iLayer];
			oclCommandqueues[iLayer].finish();
			/*trackletsFound = (int *) oclCommandqueues[iLayer].enqueueMapBuffer(
					primaryVertexContext.openClPrimaryVertexContext.bTrackletsFoundForLayer,
					CL_TRUE, // block
					CL_MAP_READ,
					0,
					6*sizeof(int)
			);
			trackletsFound[iLayer]++;
			totalTrackletsFound+=trackletsFound[iLayer]-1;*/
			if(iLayer==0){
				clogs::ScanProblem problem;
				problem.setType(clogs::TYPE_UINT);
				clogs::Scan scanner(oclContext, oclDevice, problem);
				oclCommandqueues[iLayer].finish();
				scanner.enqueue(oclCommandqueues[iLayer], bTrackletLookUpTable, bTrackletLookUpTable, clustersNum);
				/*
				oclTestKernel.setArg(0, bTrackletLookUpTable);
				oclTestKernel.setArg(1, bTrackletLookUpTable);

				int pseudoClusterNumber=clustersNum;
				//std::cout<<"ClustersNum = "<<clustersNum<<std::endl;
				if((clustersNum % workgroupSize)!=0){
					int mult=clustersNum/workgroupSize;
					pseudoClusterNumber=(mult+1)*workgroupSize;
				}
				oclCommandqueues[iLayer].enqueueNDRangeKernel(
					oclTestKernel,
					cl::NullRange,
					//cl::NDRange(pseudoClusterNumber),
					cl::NDRange(clustersNum),	//se non funziona abilitare questa rigga e la successiva commentata
					//cl::NDRange(workgroupSize));
					//cl::NullRange);
*/
/*
				int* lookUpFound = (int *) oclCommandqueues[iLayer].enqueueMapBuffer(
						bTrackletLookUpTable,
						CL_TRUE, // block
						CL_MAP_READ,
						0,
						clustersNum*sizeof(int)
				);
				outFileLookUp<<"Layer "<<iLayer<<"\n";
				//std::cout<<clustersNum<<std::endl;
				for(int j=0;j<clustersNum;j++)
					outFileLookUp<<j<<"\t"<<lookUpFound[j]<<"\n";
				outFileLookUp<<"\n";
*/
			}
			else{
				clogs::ScanProblem problem;
				problem.setType(clogs::TYPE_UINT);
				clogs::Scan scanner(oclContext, oclDevice, problem);
				oclCommandqueues[iLayer].finish();
				scanner.enqueue(oclCommandqueues[iLayer], primaryVertexContext.openClPrimaryVertexContext.bTrackletsLookupTable[iLayer-1], primaryVertexContext.openClPrimaryVertexContext.bTrackletsLookupTable[iLayer-1], clustersNum);
				/*
				oclTestKernel.setArg(0, primaryVertexContext.openClPrimaryVertexContext.bTrackletsLookupTable[iLayer-1]);
				oclTestKernel.setArg(1, primaryVertexContext.openClPrimaryVertexContext.bTrackletsLookupTable[iLayer-1]);

				int pseudoClusterNumber=clustersNum;
				if((clustersNum % workgroupSize)!=0){
					int mult=clustersNum/workgroupSize;
					pseudoClusterNumber=(mult+1)*workgroupSize;
				}

				oclCommandqueues[iLayer].enqueueNDRangeKernel(
					oclTestKernel,
					cl::NullRange,
					cl::NDRange(pseudoClusterNumber),
					cl::NDRange(workgroupSize));
*//*
				int* lookUpFound = (int *) oclCommandqueues[iLayer].enqueueMapBuffer(
						primaryVertexContext.openClPrimaryVertexContext.bTrackletsLookupTable[iLayer-1],
						CL_TRUE, // block
						CL_MAP_READ,
						0,
						clustersNum*sizeof(int)
				);
				outFileLookUp<<"Layer "<<iLayer<<"\n";
				for(int j=0;j<clustersNum;j++)
					outFileLookUp<<j<<"\t"<<lookUpFound[j]<<"\n";
				outFileLookUp<<"\n";
*/
			}
			//ty=clock();
			//float time = ((float) ty - (float) tx) / (CLOCKS_PER_SEC / 1000);
			//std::cout<< "\t " << iLayer <<" time = "<<time<<" ms" << std::endl;
		}
		//std::cout<<"finish sort"<<std::endl;
#ifdef PRINT_EXECUTION_TIME
		t2 = clock();
		float scanTrack = ((float) t2 - (float) t1) / (CLOCKS_PER_SEC / 1000);
		std::cout<< "scanTrack time " << scanTrack <<" ms" << std::endl;
#endif
#else
		std::cout<<"Error during tracklets founding (Clogs not included)"<<std::endl;
		return;
#endif

		//calcolo le tracklet
		//std::cout<<"calcolo le tracklet"<<std::endl;
		t1=clock();
		for (int iLayer{ 0 }; iLayer<Constants::ITS::TrackletsPerRoad; ++iLayer) {
			//tx=clock();
			//outFileTracklet<<"Tracklets between Layer "<<iLayer<<" and "<<iLayer+1<<"\n";

			clustersNum=primaryVertexContext.openClPrimaryVertexContext.iClusterSize[iLayer];

			oclCommandqueues[iLayer].finish();
			oclComputeKernel.setArg(0, primaryVertexContext.openClPrimaryVertexContext.bPrimaryVertex);
			oclComputeKernel.setArg(1, primaryVertexContext.openClPrimaryVertexContext.bClusters[iLayer]);
			oclComputeKernel.setArg(2, primaryVertexContext.openClPrimaryVertexContext.bClusters[iLayer+1]);
			oclComputeKernel.setArg(3, primaryVertexContext.openClPrimaryVertexContext.bIndexTables[iLayer]);
			oclComputeKernel.setArg(4, primaryVertexContext.openClPrimaryVertexContext.bTracklets[iLayer]);
			oclComputeKernel.setArg(5, primaryVertexContext.openClPrimaryVertexContext.bLayerIndex[iLayer]);
			oclComputeKernel.setArg(6, primaryVertexContext.openClPrimaryVertexContext.bTrackletsFoundForLayer);
			oclComputeKernel.setArg(7, primaryVertexContext.openClPrimaryVertexContext.bClustersSize);
			if(iLayer==0)
				oclComputeKernel.setArg(8, bTrackletLookUpTable);
			else
				oclComputeKernel.setArg(8, primaryVertexContext.openClPrimaryVertexContext.bTrackletsLookupTable[iLayer-1]);

			int pseudoClusterNumber=clustersNum;
			if((clustersNum % workgroupSize)!=0){
				int mult=clustersNum/workgroupSize;
				pseudoClusterNumber=(mult+1)*workgroupSize;
			}


			oclCommandqueues[iLayer].enqueueNDRangeKernel(
					oclComputeKernel,
					cl::NullRange,
					cl::NDRange(pseudoClusterNumber),
					cl::NDRange(workgroupSize));



/*
			TrackletStruct* output = (TrackletStruct *) oclCommandqueues[iLayer].enqueueMapBuffer(
				primaryVertexContext.openClPrimaryVertexContext.bTracklets[iLayer],
				CL_TRUE, // block
				CL_MAP_READ,
				0,
				trackletsFound[iLayer] * sizeof(TrackletStruct)
			);

			for(int i=0;i<trackletsFound[iLayer];i++)
				outFileTracklet<<output[i].firstClusterIndex<<"\t"<<output[i].secondClusterIndex<<"\t"<<output[i].phiCoordinate<<"\t"<<output[i].tanLambda<<"\n";
*/
/*
			for(int i=0;i<trackletsFound[iLayer];i++)
				//outFileTracklet<<"["<<i<<"] "<<output[i].firstClusterIndex<<"\t"<<output[i].secondClusterIndex<<"\t"<<output[i].phiCoordinate<<"\t"<<output[i].tanLambda<<"\n";
				outFileTracklet<<std::setprecision(6)<<output[i].firstClusterIndex<<"\t"<<output[i].secondClusterIndex<<"\t"<<output[i].phiCoordinate<<"\t"<<output[i].tanLambda<<"\n";
			outFileTracklet<<"\n\n";
*/
			//ty=clock();
			//float time = ((float) ty - (float) tx) / (CLOCKS_PER_SEC / 1000);
			//std::cout<< "\tLayer " << iLayer <<" time = "<<time<<" ms" <<"\tWG = " <<workgroupSize<<std::endl;
		}
/*		for (int iLayer{ 0 }; iLayer<Constants::ITS::TrackletsPerRoad; ++iLayer)
			oclCommandqueues[iLayer].finish();
			//std::cout<<"finish trackletsFinding"<<std::endl;
			 *
			 */
#ifdef PRINT_EXECUTION_TIME
		t2 = clock();
		float findTrack = ((float) t2 - (float) t1) / (CLOCKS_PER_SEC / 1000);
		std::cout<< "findTrack time " << findTrack <<" ms" << std::endl;
		std::cout<<"Total tracklets found = "<<totalTrackletsFound<<std::endl;
#endif
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

	//t2 = clock();
	//float diff = ((float) t2 - (float) t0) / (CLOCKS_PER_SEC / 1000);
	//std::cout<< "compute tracklets time " << diff <<" ms" << std::endl;
}




template<>
void TrackerTraits<true>::computeLayerCells(CA::PrimaryVertexContext& primaryVertexContext)
{
	//std::array<size_t, Constants::ITS::CellsPerRoad - 1> tempSize;
	//std::array<int, Constants::ITS::CellsPerRoad - 1> trackletsNum;
	//std::array<int, Constants::ITS::CellsPerRoad - 1> cellsNum;
	//int iTrackletSize[Constants::ITS::TrackletsPerRoad];
	cl::CommandQueue oclCommandqueues[Constants::ITS::CellsPerRoad];
	cl::Buffer bLayerID;
	cl::Buffer bCellsLookUpTableForLayer0;
	int * iCellsLookUpTableForLayer0;
	cl::CommandQueue oclCommandQueue;
	//int *firstLayerLookUpTable;
	int trackletsNum;
	//time_t t1,t2;
	int workgroupSize=5*32;
	int *cellsFound;



	try{
  		cl::Context oclContext=GPU::Context::getInstance().getDeviceProperties().oclContext;
  		cl::Device oclDevice=GPU::Context::getInstance().getDeviceProperties().oclDevice;
  		cl::CommandQueue oclCommandQueue=GPU::Context::getInstance().getDeviceProperties().oclQueue;

  		//must be move to oclContext create
  		cl::Kernel oclCountKernel=GPU::Utils::CreateKernelFromFile(oclContext,oclDevice,"./src/kernel/computeLayerCells.cl","countLayerCells");
  		cl::Kernel oclComputeKernel=GPU::Utils::CreateKernelFromFile(oclContext,oclDevice,"./src/kernel/computeLayerCells.cl","computeLayerCells");

  		int tmpCellFound[]={0,0,0,0,0,0,0};
		cl::Buffer bTmpCellFound = cl::Buffer(
			oclContext,
			(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
			7*sizeof(int),
			(void *) &(tmpCellFound[0]));



  		for(int i=0;i<Constants::ITS::CellsPerRoad;i++){
  			oclCommandqueues[i]=cl::CommandQueue(oclContext, oclDevice, 0);
  		}

  		const char outputCellFileName[] = "../oclCellsFound.txt";
		std::ofstream outFileCell;
		outFileCell.open((const char*)outputCellFileName);


  		//create buffer for allocate the number of cell found for each layer
  		int *trackletsFound = (int *) oclCommandqueues[0].enqueueMapBuffer(
  							primaryVertexContext.openClPrimaryVertexContext.bTrackletsFoundForLayer,
  							CL_TRUE, // block
  							CL_MAP_READ,
  							0,
  							6*sizeof(int)
  				);

  		int firstLayerTrackletsNumber=trackletsFound[0];
  		iCellsLookUpTableForLayer0=(int*)malloc(firstLayerTrackletsNumber*sizeof(int));
		memset(iCellsLookUpTableForLayer0,-1,firstLayerTrackletsNumber*sizeof(int));
		bCellsLookUpTableForLayer0 = cl::Buffer(
				oclContext,
				(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
				firstLayerTrackletsNumber*sizeof(int),
				(void *) iCellsLookUpTableForLayer0);


  		//compute the number of cells for each layer
    	for (int iLayer { 0 }; iLayer < Constants::ITS::CellsPerRoad;++iLayer) {



    		oclCountKernel.setArg(0, primaryVertexContext.openClPrimaryVertexContext.bPrimaryVertex);  //0 fPrimaryVertex
    		oclCountKernel.setArg(1, primaryVertexContext.openClPrimaryVertexContext.bLayerIndex[iLayer]); //1 iCurrentLayer
			oclCountKernel.setArg(2, primaryVertexContext.openClPrimaryVertexContext.bTrackletsFoundForLayer);  //2 iLayerTrackletSize
			oclCountKernel.setArg(3, primaryVertexContext.openClPrimaryVertexContext.bTracklets[iLayer]); //3  currentLayerTracklets
			oclCountKernel.setArg(4, primaryVertexContext.openClPrimaryVertexContext.bTracklets[iLayer+1]); //4 nextLayerTracklets
			oclCountKernel.setArg(5, primaryVertexContext.openClPrimaryVertexContext.bTracklets[iLayer+2]); //5 next2LayerTracklets
			oclCountKernel.setArg(6, primaryVertexContext.openClPrimaryVertexContext.bClusters[iLayer]);  //6 currentLayerClusters
			oclCountKernel.setArg(7, primaryVertexContext.openClPrimaryVertexContext.bClusters[iLayer+1]);//7 nextLayerClusters
			oclCountKernel.setArg(8, primaryVertexContext.openClPrimaryVertexContext.bClusters[iLayer+2]);//8 next2LayerClusters
			oclCountKernel.setArg(9, primaryVertexContext.openClPrimaryVertexContext.bTrackletsLookupTable[iLayer]);//9  currentLayerTrackletsLookupTable

			if(iLayer==0)
				oclCountKernel.setArg(10, bCellsLookUpTableForLayer0);//9iCellsPerTrackletPreviousLayer;
			else
				oclCountKernel.setArg(10, primaryVertexContext.openClPrimaryVertexContext.bCellsLookupTable[iLayer-1]);//9iCellsPerTrackletPreviousLayer
			oclCountKernel.setArg(11, primaryVertexContext.openClPrimaryVertexContext.bCellsFoundForLayer);


			int pseudoClusterNumber=trackletsFound[iLayer];
			if((pseudoClusterNumber % workgroupSize)!=0){
				int mult=pseudoClusterNumber/workgroupSize;
				pseudoClusterNumber=(mult+1)*workgroupSize;
			}



			oclCommandqueues[iLayer].enqueueNDRangeKernel(
				oclCountKernel,
				cl::NullRange,
				cl::NDRange(pseudoClusterNumber),
				cl::NDRange(workgroupSize));



    	}



    	//scan for cells lookup table
#if CLOGS
		//scan
    	int *cellsFound;
		for (int iLayer { 0 }; iLayer<Constants::ITS::CellsPerRoad; ++iLayer) {
			//tx=clock();
			//outFileLookUp<<"From layer "<<iLayer<<" to "<<iLayer+1<<"\n";
			trackletsNum=trackletsFound[iLayer];
			oclCommandqueues[iLayer].finish();
			cellsFound = (int *) oclCommandqueues[iLayer].enqueueMapBuffer(
					primaryVertexContext.openClPrimaryVertexContext.bCellsFoundForLayer,
					CL_TRUE, // block
					CL_MAP_READ,
					0,
					5*sizeof(int)
			);
			std::cout<<"["<<iLayer<<"] : "<<cellsFound[iLayer]<<std::endl;
			if(iLayer==0){
				clogs::ScanProblem problem;
				problem.setType(clogs::TYPE_UINT);
				clogs::Scan scanner(oclContext, oclDevice, problem);
				scanner.enqueue(oclCommandqueues[iLayer], bCellsLookUpTableForLayer0, bCellsLookUpTableForLayer0, trackletsNum);
				oclCommandqueues[iLayer].finish();
				/*
				oclTestKernel.setArg(0, bTrackletLookUpTable);
				oclTestKernel.setArg(1, bTrackletLookUpTable);

				int pseudoClusterNumber=clustersNum;
				//std::cout<<"ClustersNum = "<<clustersNum<<std::endl;
				if((clustersNum % workgroupSize)!=0){
					int mult=clustersNum/workgroupSize;
					pseudoClusterNumber=(mult+1)*workgroupSize;
				}
				oclCommandqueues[iLayer].enqueueNDRangeKernel(
					oclTestKernel,
					cl::NullRange,
					//cl::NDRange(pseudoClusterNumber),
					cl::NDRange(clustersNum),	//se non funziona abilitare questa rigga e la successiva commentata
					//cl::NDRange(workgroupSize));
					//cl::NullRange);
*/
/*
				int* lookUpFound = (int *) oclCommandqueues[iLayer].enqueueMapBuffer(
						bTrackletLookUpTable,
						CL_TRUE, // block
						CL_MAP_READ,
						0,
						clustersNum*sizeof(int)
				);
				outFileLookUp<<"Layer "<<iLayer<<"\n";
				//std::cout<<clustersNum<<std::endl;
				for(int j=0;j<clustersNum;j++)
					outFileLookUp<<j<<"\t"<<lookUpFound[j]<<"\n";
				outFileLookUp<<"\n";
*/
			}
			else{
				clogs::ScanProblem problem;
				problem.setType(clogs::TYPE_UINT);
				clogs::Scan scanner(oclContext, oclDevice, problem);
				scanner.enqueue(oclCommandqueues[iLayer], primaryVertexContext.openClPrimaryVertexContext.bCellsLookupTable[iLayer-1], primaryVertexContext.openClPrimaryVertexContext.bCellsLookupTable[iLayer-1], trackletsNum);
				oclCommandqueues[iLayer].finish();
				/*
				oclTestKernel.setArg(0, primaryVertexContext.openClPrimaryVertexContext.bTrackletsLookupTable[iLayer-1]);
				oclTestKernel.setArg(1, primaryVertexContext.openClPrimaryVertexContext.bTrackletsLookupTable[iLayer-1]);

				int pseudoClusterNumber=clustersNum;
				if((clustersNum % workgroupSize)!=0){
					int mult=clustersNum/workgroupSize;
					pseudoClusterNumber=(mult+1)*workgroupSize;
				}

				oclCommandqueues[iLayer].enqueueNDRangeKernel(
					oclTestKernel,
					cl::NullRange,
					cl::NDRange(pseudoClusterNumber),
					cl::NDRange(workgroupSize));
*//*
				int* lookUpFound = (int *) oclCommandqueues[iLayer].enqueueMapBuffer(
						primaryVertexContext.openClPrimaryVertexContext.bTrackletsLookupTable[iLayer-1],
						CL_TRUE, // block
						CL_MAP_READ,
						0,
						clustersNum*sizeof(int)
				);
				outFileLookUp<<"Layer "<<iLayer<<"\n";
				for(int j=0;j<clustersNum;j++)
					outFileLookUp<<j<<"\t"<<lookUpFound[j]<<"\n";
				outFileLookUp<<"\n";
*/
			}
			//ty=clock();
			//float time = ((float) ty - (float) tx) / (CLOCKS_PER_SEC / 1000);
			//std::cout<< "\t " << iLayer <<" time = "<<time<<" ms" << std::endl;
		}
		//std::cout<<"finish sort"<<std::endl;
#ifdef PRINT_EXECUTION_TIME
		t2 = clock();
		float scanTrack = ((float) t2 - (float) t1) / (CLOCKS_PER_SEC / 1000);
		std::cout<< "scanTrack time " << scanTrack <<" ms" << std::endl;
#endif
#else
		std::cout<<"Error during tracklets founding (Clogs not included)"<<std::endl;
		return;
#endif
    	//compute cells
		//calcolo le tracklet
		//std::cout<<"calcolo le tracklet"<<std::endl;
		//t1=clock();
		for (int iLayer { 0 }; iLayer < Constants::ITS::CellsPerRoad;++iLayer) {
			outFileCell<<"Cells for layer "<<iLayer<<std::endl;
			oclComputeKernel.setArg(0, primaryVertexContext.openClPrimaryVertexContext.bPrimaryVertex);  //0 fPrimaryVertex
			oclComputeKernel.setArg(1, primaryVertexContext.openClPrimaryVertexContext.bLayerIndex[iLayer]); //1 iCurrentLayer
			oclComputeKernel.setArg(2, primaryVertexContext.openClPrimaryVertexContext.bTrackletsFoundForLayer);  //2 iLayerTrackletSize
			oclComputeKernel.setArg(3, primaryVertexContext.openClPrimaryVertexContext.bTracklets[iLayer]); //3  currentLayerTracklets
			oclComputeKernel.setArg(4, primaryVertexContext.openClPrimaryVertexContext.bTracklets[iLayer+1]); //4 nextLayerTracklets
			oclComputeKernel.setArg(5, primaryVertexContext.openClPrimaryVertexContext.bTracklets[iLayer+2]); //5 next2LayerTracklets
			oclComputeKernel.setArg(6, primaryVertexContext.openClPrimaryVertexContext.bClusters[iLayer]);  //6 currentLayerClusters
			oclComputeKernel.setArg(7, primaryVertexContext.openClPrimaryVertexContext.bClusters[iLayer+1]);//7 nextLayerClusters
			oclComputeKernel.setArg(8, primaryVertexContext.openClPrimaryVertexContext.bClusters[iLayer+2]);//8 next2LayerClusters
			oclComputeKernel.setArg(9, primaryVertexContext.openClPrimaryVertexContext.bTrackletsLookupTable[iLayer]);//9  currentLayerTrackletsLookupTable

			if(iLayer==0)
				oclComputeKernel.setArg(10, bCellsLookUpTableForLayer0);//9iCellsPerTrackletPreviousLayer;
			else
				oclComputeKernel.setArg(10, primaryVertexContext.openClPrimaryVertexContext.bCellsLookupTable[iLayer-1]);//9iCellsPerTrackletPreviousLayer
			oclComputeKernel.setArg(11, primaryVertexContext.openClPrimaryVertexContext.bCells[iLayer]);

			int pseudoClusterNumber=trackletsFound[iLayer];
			if((pseudoClusterNumber % workgroupSize)!=0){
				int mult=pseudoClusterNumber/workgroupSize;
				pseudoClusterNumber=(mult+1)*workgroupSize;
			}



			oclCommandqueues[iLayer].enqueueNDRangeKernel(
				oclComputeKernel,
				cl::NullRange,
				cl::NDRange(pseudoClusterNumber),
				cl::NDRange(workgroupSize));


			CellStruct* output = (CellStruct *) oclCommandqueues[iLayer].enqueueMapBuffer(
				primaryVertexContext.openClPrimaryVertexContext.bCells[iLayer],
				CL_TRUE, // block
				CL_MAP_READ,
				0,
				cellsFound[iLayer]
			);

			for(int i=0;i<cellsFound[iLayer];i++)
				outFileCell<<output[i].mFirstTrackletIndex<<"\t"<<output[i].mSecondTrackletIndex<<"\t"<<output[i].mCurvature<<"\t"<<output[i].mLevel<<"\t"<<output[i].mFirstClusterIndex<<"\t"<<output[i].mSecondClusterIndex<<"\t"<<output[i].mThirdClusterIndex<<"\n";


		}


  	}catch (...) {
		std::cout<<"Exception during compute cells phase"<<std::endl;
		throw std::runtime_error { "Exception during compute cells phase" };
	}
}

}
}
}
