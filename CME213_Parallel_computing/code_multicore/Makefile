GTEST_DIR = googletest-1.13.0/googletest

# Flags passed to the preprocessor.
# Set Google Test's header directory as a system directory, such that
# the compiler doesn't generate warnings in Google Test headers.
CPPFLAGS += -isystem $(GTEST_DIR)/include

# C++ compiler
CXX = g++

# All Google Test headers. 
# Usually you shouldn't change this definition.
GTEST_HEADERS = $(GTEST_DIR)/include/gtest/*.h \
                $(GTEST_DIR)/include/gtest/internal/*.h

# Builds gtest.a and gtest_main.a.
GTEST_SRCS_ = $(GTEST_DIR)/src/*.cc $(GTEST_DIR)/src/*.h $(GTEST_HEADERS)

all: gtest.a gtest_main.a

gtest-all.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) -c \
            $(GTEST_DIR)/src/gtest-all.cc

gtest_main.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) -c \
            $(GTEST_DIR)/src/gtest_main.cc
            
gtest.a : gtest-all.o
	$(AR) $(ARFLAGS) $@ $^

gtest_main.a : gtest-all.o gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

clean:
	rm -f *.o
	rm -rf *.dSYM
	rm -rf gtest.a gtest_main.a	

clean_gtest:
	rm -rf googletest-1.13.0
	rm -f v1.13.0.zip
	rm -rf gtest_main.a gtest_main.o gtest-all.o gtest.a gtest