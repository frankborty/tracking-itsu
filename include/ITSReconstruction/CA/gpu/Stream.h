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
/// \file Stream.h
/// \brief
///

#ifndef TRAKINGITSU_INCLUDE_GPU_STREAM_H_
#define TRAKINGITSU_INCLUDE_GPU_STREAM_H_

#include "ITSReconstruction/CA/Definitions.h"
#if TRACKINGITSU_OCL_MODE
#include "ITSReconstruction/CA/gpu/Context.h"
#endif

namespace o2
{
namespace ITS
{
namespace CA
{
namespace GPU
{

class Stream
  final {

    public:
      Stream();
      ~Stream();

      Stream(const Stream&) = delete;
      Stream &operator=(const Stream&) = delete;

      const GPUStream& get() const;

    private:
#if TRACKINGITSU_OCL_MODE
      cl::CommandQueue oclCommandQueue;
#endif
      GPUStream mStream;
  };

}
}
}
}

#endif /* TRAKINGITSU_INCLUDE_GPU_STREAM_H_ */
