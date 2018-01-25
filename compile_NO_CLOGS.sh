echo "cd oclBuildNoClogs"
cd oclBuildNoClogs

echo "rm -rf CMakeFiles"
rm -rf CMakeFiles

echo "rm -rf src"
rm -rf src

echo "rm -f CMakeCache.txt"
rm -f CMakeCache.txt

echo "rm -f cmake_install.cmake"
rm -f cmake_install.cmake

echo "rm -f cmake_install.cmake"
rm -f cmake_install.cmake

echo "rm -f Makefile"
rm -f Makefile

echo "rm -f tracking-itsu-main"
rm -f tracking-itsu-main

echo "cmake -G"Eclipse CDT4 - Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug -DTRACKINGITSU_TARGET_DEVICE=OPEN_CL ../tracking_ocl/"
cmake -G"Eclipse CDT4 - Unix Makefiles" -O3 -std=c++11  -DTRACKINGITSU_TARGET_DEVICE=OPEN_CL -DUSE_CLOGS=NO ../tracking_ocl/

echo "make -j"
make -j

