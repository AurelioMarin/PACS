CXX       = g++
CXXFLAGS  = -std=c++17 
OPENNN_DIR =/home/aumar/PACS/opennn-master
LIB_DIR+=/home/aumar/PACS/opennn-master/Release/opennn
INCLUDES=-I$(OPENNN_DIR) -L$(LIB_DIR) 
LDLIBS += -lopennn
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
