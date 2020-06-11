CXX       = g++
CXXFLAGS  = -std=c++17 
MLPACK_DIR =/home/aumar/PACS/Proyecto/mlpack-3.3.1/build/lib
INCLUDES=-L$(MLPACK_DIR) 
LDLIBS += -larmadillo -lboost_serialization -lmlpack -lboost_program_options 
CXXFLAGS += -Wall $(INCLUDES) -fopenmp

EXEC = main

SRCS = $(wildcard *.cpp)
OBJS = $(SRCS:.cpp=.o)

$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS)

clean:
	$(RM) *.o
distclean:clean
	$(RM) main
