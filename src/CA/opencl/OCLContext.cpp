#include "CAGPUContext.h"
#include "CAGPUUtils.h"
#include <sstream>
#include <vector>

#include <CL/cl.hpp>
//#define __CL_ENABLE_EXCEPTIONS //abilita le eccezioni



inline int getMaxThreadsPerSM(const int major, const int minor)
{
  return 8;
}



CAGPUContext::CAGPUContext()
{
	std::vector<cl::Platform> platformList;
	std::vector<cl::Device> deviceList;
	std::vector<std::size_t> sizeDim;
	std::string info;
	std::size_t iPlatformList;
	std::size_t iTotalDevice=0;
	std::size_t iFlagQueue=0;
	try{

		// Get the list of platform
		cl::Platform::get(&platformList);
		iPlatformList=platformList.size();
		// Pick first platform

		std::cout << "There are " << iPlatformList << " platform" << std::endl;
		std::cout << std::endl;
		for(int iPlatForm=0;iPlatForm<(int)iPlatformList;iPlatForm++){
			std::cout << "Platform #" << iPlatForm+1 << std::endl;
			cl_context_properties cprops[] = {
				CL_CONTEXT_PLATFORM, (cl_context_properties)(platformList[iPlatForm])(), 0};
			cl::Context context(CL_DEVICE_TYPE_ALL, cprops);


			//print platform information
			platformList[iPlatForm].getInfo(CL_PLATFORM_NAME,&info);
			std::cout << "Name:" 	<< info << std::endl;
			platformList[iPlatForm].getInfo(CL_PLATFORM_VENDOR,&info);
			std::cout << "Vendor:"	<< info << std::endl;
			platformList[iPlatForm].getInfo(CL_PLATFORM_VERSION,&info);
			std::cout << "Version: "<< info << std::endl;


			// Get devices associated with the first platform
			platformList[iPlatForm].getDevices(CL_DEVICE_TYPE_ALL,&deviceList);
			mDevicesNum=deviceList.size();
			mDeviceProperties.resize(iTotalDevice+mDevicesNum, CAGPUDeviceProperties { });

			std::cout << "There are " << mDevicesNum << " devices" << std::endl;

			for(int iDevice=0;iDevice<mDevicesNum;iDevice++){

				std::string name;
				deviceList[iDevice].getInfo(CL_DEVICE_NAME,&(mDeviceProperties[iTotalDevice].name));
				std::cout << "	>> Device: " << mDeviceProperties[iTotalDevice].name << std::endl;

				//compute number of compute units (cores)
				deviceList[iDevice].getInfo(CL_DEVICE_MAX_COMPUTE_UNITS,&(mDeviceProperties[iTotalDevice].oclCores));
				std::cout << "		Compute units: " << mDeviceProperties[iTotalDevice].oclCores << std::endl;

				//compute device global memory size
				deviceList[iDevice].getInfo(CL_DEVICE_GLOBAL_MEM_SIZE,&(mDeviceProperties[iTotalDevice].globalMemorySize));
				std::cout << "		Device Global Memory: " << mDeviceProperties[iTotalDevice].globalMemorySize << std::endl;

				//compute the max number of work-item in a work group executing a kernel (refer to clEnqueueNDRangeKernel)
				deviceList[iDevice].getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE,&(mDeviceProperties[iTotalDevice].maxWorkGroupSize));
				std::cout << "		Max work-group size: " << mDeviceProperties[iTotalDevice].maxWorkGroupSize << std::endl;

				//compute the max work-item dimension
				deviceList[iDevice].getInfo(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,&(mDeviceProperties[iTotalDevice].maxWorkItemDimension));
				std::cout << "		Max work-item dimension: " << mDeviceProperties[iTotalDevice].maxWorkItemDimension << std::endl;

				//compute the max number of work-item that can be specified in each dimension of the work-group to clEnqueueNDRangeKernel
				deviceList[iDevice].getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES,&(sizeDim));
				mDeviceProperties[iTotalDevice].maxWorkItemSize.x=sizeDim[0];
				mDeviceProperties[iTotalDevice].maxWorkItemSize.y=sizeDim[1];
				mDeviceProperties[iTotalDevice].maxWorkItemSize.z=sizeDim[2];
				std::cout << "		Max work-item Sizes: [" << mDeviceProperties[iTotalDevice].maxWorkItemSize.x << "," << mDeviceProperties[iTotalDevice].maxWorkItemSize.y << ","<< mDeviceProperties[iTotalDevice].maxWorkItemSize.z << "]"<< std::endl;


				//store the context
				mDeviceProperties[iTotalDevice].oclContext=context;
				iTotalDevice++;
			}

			//test iFlagQueue to make the operation only for one selected device (in this case the first device)
			if(iFlagQueue==0){
				//store the index of selected device
				iCurrentDevice=0;

				//creo la coda
				cl::CommandQueue queue(context,deviceList[iCurrentDevice],0);
				mDeviceProperties[iCurrentDevice].queue=queue;
				iFlagQueue=1;
			}
			std::cout << std::endl;
		}


	}
	catch(cl::Error err){
		std::string errString=CAGPUUtils::err_code(err.err());
		std::cout<< errString << std::endl;
		throw std::runtime_error { errString };
	}


	std::cout << std::endl<< ">> First device is selected" << std::endl;




}

CAGPUContext& CAGPUContext::getInstance()
{
  static CAGPUContext oclContext;
  return oclContext;
}


const CAGPUDeviceProperties& CAGPUContext::getDeviceProperties()
{
	return getDeviceProperties(iCurrentDevice);
}

const CAGPUDeviceProperties& CAGPUContext::getDeviceProperties(const int deviceIndex)
{
  return mDeviceProperties[deviceIndex];
}

