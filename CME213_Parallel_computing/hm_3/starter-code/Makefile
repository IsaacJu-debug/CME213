# Google test installation directories.
# Make sure these directories exist.
GTEST_ROOT = ./googletest-main
GTEST_DIR = $(GTEST_ROOT)/googletest
GTEST_INC = $(GTEST_ROOT)/googletest/include/
GTEST_SRC = $(GTEST_ROOT)/googletest/src/

CXX=nvcc
CXXFLAGS=-O2 -std=c++14 -arch=compute_75 -code=sm_75
SRC1=main_q1.cu
SRC2=main_q2.cu
SRC3=main_q3.cu
HDR1=recurrence.cuh test_recurrence.h
HDR2=pagerank.cuh
HDR3=util.cuh benchmark.cuh
INC=.

CPPFLAGS += -isystem $(GTEST_INC)

GTEST_HEADERS = $(GTEST_INC)/gtest/*.h \
                $(GTEST_INC)/gtest/internal/*.h

GTEST_SRCS_ = $(GTEST_SRC)/*.cc $(GTEST_SRC)/*.h $(GTEST_HEADERS)

.PHONY: default run1 run2 run3 clean

default: main_q1 main_q2 main_q3

gtest-all.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
            $(GTEST_SRC)/gtest-all.cc

gtest_main.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
            $(GTEST_SRC)/gtest_main.cc

gtest.a : gtest-all.o
	$(AR) $(ARFLAGS) $@ $^

gtest_main.a : gtest-all.o gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

main_q1.o : $(SRC1) $(HDR1) $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(SRC1)
main_q1: main_q1.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread $^ -o $@

main_q2.o : $(SRC2) $(HDR2) $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(SRC2)
main_q2: main_q2.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread $^ -o $@

main_q3.o : $(SRC3) $(HDR3) $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(SRC3)
main_q3: main_q3.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread $^ -o $@

run1: main_q1
	srun --partition=CME --gres=gpu:1 ./main_q1

run2: main_q2
	srun --partition=CME --gres=gpu:1 ./main_q2

run3: main_q3
	srun --partition=CME --gres=gpu:1 ./main_q3

clean:
	rm -f *.o *~ *~ main_q1 main_q2 main_q3
	rm -rf *.dSYM
