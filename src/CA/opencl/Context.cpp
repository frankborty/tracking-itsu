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
/// \file Context.cu
/// \brief
///

#include "ITSReconstruction/CA/gpu/Context.h"

#include <sstream>
#include <stdexcept>

//#include <cuda_runtime.h>

namespace {




//inline int getMaxThreadsPerSM(const int major, const int minor)
//{
//  return 8;
//}

}

namespace o2
{
namespace ITS
{
namespace CA
{
namespace GPU
{

Context::Context()
{
	mDevicesNum=1;
//  checkCUDAError(cudaGetDeviceCount(&mDevicesNum), __FILE__, __LINE__);
//
//  if (mDevicesNum == 0) {
//
//    throw std::runtime_error { "There are no available device(s) that support CUDA\n" };
//  }
//
//  mDeviceProperties.resize(mDevicesNum, DeviceProperties { });
//
//  int currentDeviceIndex;
//  checkCUDAError(cudaGetDevice(&currentDeviceIndex), __FILE__, __LINE__);
//
//  for (int iDevice { 0 }; iDevice < mDevicesNum; ++iDevice) {
//
//    cudaDeviceProp deviceProperties;
//
//    checkCUDAError(cudaSetDevice(iDevice), __FILE__, __LINE__);
//    checkCUDAError(cudaGetDeviceProperties(&deviceProperties, iDevice), __FILE__, __LINE__);
//
//    int major = deviceProperties.major;
//    int minor = deviceProperties.minor;
//
//    mDeviceProperties[iDevice].name = deviceProperties.name;
//    mDeviceProperties[iDevice].gpuProcessors = deviceProperties.multiProcessorCount;
//    mDeviceProperties[iDevice].cudaCores = getCudaCores(major, minor) * deviceProperties.multiProcessorCount;
//    mDeviceProperties[iDevice].globalMemorySize = deviceProperties.totalGlobalMem;
//    mDeviceProperties[iDevice].constantMemorySize = deviceProperties.totalConstMem;
//    mDeviceProperties[iDevice].sharedMemorySize = deviceProperties.sharedMemPerBlock;
//    mDeviceProperties[iDevice].maxClockRate = deviceProperties.memoryClockRate;
//    mDeviceProperties[iDevice].busWidth = deviceProperties.memoryBusWidth;
//    mDeviceProperties[iDevice].l2CacheSize = deviceProperties.l2CacheSize;
//    mDeviceProperties[iDevice].registersPerBlock = deviceProperties.regsPerBlock;
//    mDeviceProperties[iDevice].warpSize = deviceProperties.warpSize;
//    mDeviceProperties[iDevice].maxThreadsPerBlock = deviceProperties.maxThreadsPerBlock;
//    mDeviceProperties[iDevice].maxBlocksPerSM = getMaxThreadsPerSM(major, minor);
//    mDeviceProperties[iDevice].maxThreadsDim = dim3 { static_cast<unsigned int>(deviceProperties.maxThreadsDim[0]),
//        static_cast<unsigned int>(deviceProperties.maxThreadsDim[1]),
//        static_cast<unsigned int>(deviceProperties.maxThreadsDim[2]) };
//    mDeviceProperties[iDevice].maxGridDim = dim3 { static_cast<unsigned int>(deviceProperties.maxGridSize[0]),
//        static_cast<unsigned int>(deviceProperties.maxGridSize[1]),
//        static_cast<unsigned int>(deviceProperties.maxGridSize[2]) };
//  }
//
//  checkCUDAError(cudaSetDevice(currentDeviceIndex), __FILE__, __LINE__);
}

Context& Context::getInstance()
{
  static Context gpuContext;
  return gpuContext;
}

const DeviceProperties& Context::getDeviceProperties()
{
//  int currentDeviceIndex;
//  checkCUDAError(cudaGetDevice(&currentDeviceIndex), __FILE__, __LINE__);

  return getDeviceProperties(0);
}

const DeviceProperties& Context::getDeviceProperties(const int deviceIndex)
{
	return mDeviceProperties[deviceIndex];

}

}
}
}
}
