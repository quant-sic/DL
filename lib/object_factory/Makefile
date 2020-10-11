#___________________________________________________________________________________________
#
# makefile for an object factory
#
# author  : archer
# date    :
# to-do   :
# caution :
#
# program :
#

#___________________________________________________________________________________________
# targets
TARGET = test_object_factory

all: $(TARGET)

#___________________________________________________________________________________________
# compiler and flags
CXX      := g++
CXXFLAGS := -std=c++11 -O3 -Wall -fsanitize=address
LIBS     := # -pthread

#___________________________________________________________________________________________
# directory with the source files
SRC := ./src
OBJ := ./obj
RUN := ./run
TST := ./tst

#___________________________________________________________________________________________
# includes
INC := -I $(SRC)

#___________________________________________________________________________________________
# clean 
clean:
	rm -f *~ $(OBJ)/*.o $(SRC)/*~ $(RUN)/*~ $(TST)/*~ test_object_factory

#___________________________________________________________________________________________
#
# compile commands
#___________________________________________________________________________________________
#___________________________________________________________________________________________
# run files

#___________________________________________________________________________________________
# test files
test_object_factory: $(SRC)/test_object_factory.cpp $(OBJ)/object_factory.o $(OBJ)/example_class.o $(OBJ)/timer.o
	$(CXX) -o $@ $(SRC)/test_object_factory.cpp $(LIBS) $(INC) $(CXXFLAGS)

#___________________________________________________________________________________________
# c++ object files
$(OBJ)/essentials.o: $(SRC)/essentials.h
	$(CXX) -o $@ -c $(SRC)/essentials.h $(CXXFLAGS) $(INC) $(LIBS)

$(OBJ)/timer.o: $(SRC)/timer.h $(OBJ)/essentials.o
	$(CXX) -o $@ -c $(SRC)/timer.h $(CXXFLAGS) $(INC) $(LIBS)

$(OBJ)/range.o: $(SRC)/range.h
	$(CXX) -o $@ -c $(SRC)/range.h $(CXXFLAGS) $(INC) $(LIBS)

$(OBJ)/object_factory.o: $(SRC)/object_factory.h $(OBJ)/range.o
	$(CXX) -o $@ -c $(SRC)/object_factory.h $(CXXFLAGS) $(INC) $(LIBS)

$(OBJ)/example_class.o: $(SRC)/example_class.h
	$(CXX) -o $@ -c $(SRC)/example_class.h $(CXXFLAGS) $(INC) $(LIBS)

#___________________________________________________________________________________________
# non file targets
.PHONY : all clean
