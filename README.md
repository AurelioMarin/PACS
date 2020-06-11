# Final PACS project
# Installation and Compilation Instructions
This project is based on two external libraries that are going to be tested.

This is specified for Linux user, in my case I used Ubuntu 18.04
# MLpack
The installation and compilation is based on the information provided on  https://www.mlpack.org/doc/mlpack-git/doxygen/build.html.

You will need to have installed the following dependencies:
Armadillo>=8.400.0,Boost>=1.58 (specifically the modules math_c99, program_options, serialization, unit_test_framework, heap and spirit),
ensmallen>=2.10.0(will be download if not found) and CMake>=3.3.1.

To install the dependencies you can use the following commands:

sudo apt-get install libboost-math-dev libboost-program-options-dev libboost-random-dev libboost-test-dev libboost-serialization-dev libboost-all-dev

sudo apt-get install libxml2-dev libarmadillo-dev binutils-dev  doxygen

Once the dependencies are ready, we move to the folder with the package to make a build folder where we will execute makefile to install the library.

$cd build

$cmake -D DEBUG=ON -D PROFILE=ON ../

$make

$sudo make install

When executing cmake, it would be needed to  specifies the Boost library directory on the command line (-DBOOST_LIBRARYDIR) since it is stored in /usr/lib/x86_64-linux-gnu. Otherwise,the modules program_options, serialization, unit_test_framework will not be found.

Now, we are ready to run our examples. All the examples include a makefile to ease their compilation. The user will need to change the variable MLPACK_DIR equal to the directory "include" on the previous build directory created.

# OpenNN

The installation and compilation is based on the information provided on https://www.opennn.net/documentation/building_opennn.html

The only dependency needed is the Eigen library which will be installed automatically while installing OpenNN.

We can download the package with GitHub or SourceForce.However, during the installation we have found that the SourceForce wil not found the information to install the given examples by the library. Therefore, It would be recommended to follow the GitHub via.
Once on the library folder, we run the following to install it:

$mkdir opennn-master\Release && cd opennn-master\Release

$cmake -DCMAKE_TYPE_BUILD=Release .

$make

This will generate the libopennn.a in Release\opennn.\\
For each example is associated a makefile where only should be modified the directory of the library content,OPENNN_DIR, and the liopennn.a,LIB_DIR.