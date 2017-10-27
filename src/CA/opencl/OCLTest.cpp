//#include "OCLUtils.h"

#include <CL/cl.hpp>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#define __CL_ENABLE_EXCEPTIONS //abilita le eccezioni

#define BUFFER_SIZE 20


#include <unistd.h>
#define GetCurrentDir getcwd

#include<iostream>


int A[BUFFER_SIZE];
int B[BUFFER_SIZE];
int C[BUFFER_SIZE];

namespace OCLTest {
	int sumNum(void){
	int n1=10;
	int n2=20;
	int n3=0;

	std::vector<cl::Platform> platformList;
	std::string info;

	char buff[FILENAME_MAX];
	  GetCurrentDir( buff, FILENAME_MAX );
	  std::string current_working_dir(buff);


	std::cout << current_working_dir << std::endl;


	//seleziono la prima piattaforma e creo il contesto
	cl::Platform::get(&platformList);
	cl_context_properties cprops[]= {
	CL_CONTEXT_PLATFORM,
	(cl_context_properties)platformList[0](),
	0
	};

	cl::Context context (CL_DEVICE_TYPE_CPU,cprops);

	//selezionio il primo device associato al contesto e stampo il nome
	std::vector<cl::Device> devices= context.getInfo<CL_CONTEXT_DEVICES>();

	std::ifstream fileSrc("../src/opencl/kernel/add2Num.cl");
	//std::ifstream fileSrc("../tracking-itsuEclipse/src/opencl/kernel/add2Num.cl");
	if (!fileSrc.is_open()) {
	      std::cout << "Failed to read add2Num.cl";
	      exit(1);
	}
	std::string prog(
	  std::istreambuf_iterator<char>(fileSrc),
	  (std::istreambuf_iterator<char>()));


	//cl::Program::Sources sources(1,std::make_pair(prog.c_str(),prog.length()+1));
	cl::Program program (context, prog);


	program.build(devices);

	// Create buffer for A and copy host contents
	cl::Buffer aBuffer = cl::Buffer(
	  context,
	  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	  sizeof(int),
	  (void *) &n1);

	// Create buffer for B and copy host contents
	cl::Buffer bBuffer = cl::Buffer(
	  context,
	  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	  sizeof(int),
	  (void *) &n2);

	// Create buffer for that uses the host ptr C
	cl::Buffer cBuffer = cl::Buffer(
	  context,
	  CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
	  sizeof(int),
	  (void *) &n3);

	// Create kernel object
	cl::Kernel kernel(program, "add2Num");

	kernel.setArg(0,n1);
	kernel.setArg(1,n2);
	kernel.setArg(2,cBuffer);

	cl::CommandQueue queue(context, devices[0], 0);

	// Do the work
	queue.enqueueNDRangeKernel(
	  kernel,
	  cl::NullRange,
	  cl::NDRange(1),
	  cl::NullRange);




	std::cout << std::endl;

	int * output = (int *) queue.enqueueMapBuffer(
	  cBuffer,
	  CL_TRUE, // block
	  CL_MAP_READ,
	  0,
	  sizeof(int));

	//for (int i = 0; i < BUFFER_SIZE; i++) {
	//std::cout << "RESULT:";
	//std::cout << *output <<std::endl;
	//}

	return *output;
	}
}
