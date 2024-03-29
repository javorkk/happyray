#Source code directrory
SRCDIR    := src
INTERMDIR := obj

#default install location, use env var if invalid
CUDA_INSTALL_PATH=/usr/local/cuda
#CUDA_INSTALL_PATH=/usr

# Name of the execuatable to build
TARGET    := happyray

# Cuda source files (compiled with nvcc)
CUFILES   := $(shell find $(SRCDIR) -iname '*.cu')

MACHINE   := $(shell uname -s)
#Additional libraries
LIBS      := -lpng -lcudart -lGL -lSDL2

# C/C++ source files (compiled with gcc / c++)
CCFILES := $(shell find $(SRCDIR) -iname '*.cpp')

################################################################################
# Rules and targets

.SUFFIXES : .cu 

# detect OS
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = $(shell uname -s 2>/dev/nullcuda | tr [:upper:] [:lower:])

# Compilers
NVCC       := nvcc
CXX        := ccbin/g++
LINK       := ccbin/g++ -fPIC

# Includes
INCLUDES  += -I$(SRCDIR) -I$(CUDA_INSTALL_PATH)/include -I/usr/include/SDL2 -Icontrib/include
CXXFLAGS += $(INCLUDES) -std=c++11 -fPIC
CFLAGS += $(INCLUDES)
NVCCFLAGS += $(INCLUDES) -Xcompiler -fPIC


ifeq ($(sm_60), 1)
	NVCCFLAGS   += -arch sm_60 -D HAPPYRAY__CUDA_ARCH__=600
else ifeq ($(sm_50), 1)
	NVCCFLAGS   += -arch sm_50 -D HAPPYRAY__CUDA_ARCH__=500
else ifeq ($(sm_30), 1)
	NVCCFLAGS   += -arch sm_30 -D HAPPYRAY__CUDA_ARCH__=300
else ifeq ($(sm_20), 1)
	NVCCFLAGS   += -arch sm_20 -D HAPPYRAY__CUDA_ARCH__=200
else ifeq ($(sm_13), 1)
	NVCCFLAGS   += -arch sm_13 -D HAPPYRAY__CUDA_ARCH__=130
else ifeq ($(sm_12), 1)
	NVCCFLAGS   += -arch sm_12 -D HAPPYRAY__CUDA_ARCH__=120
else ifeq ($(sm_11), 1)
	NVCCFLAGS   += -arch sm_11 -D HAPPYRAY__CUDA_ARCH__=110
else
	NVCCFLAGS   += -arch sm_60 -D HAPPYRAY__CUDA_ARCH__=600
endif

ifeq ($(dbg),1)
        COMMONFLAGS += -g
        NVCCFLAGS   += -D_DEBUG
else
        COMMONFLAGS += -O2
        NVCCFLAGS   += -DNDEBUG --compiler-options -fno-strict-aliasing -use_fast_math
endif

NVCCFLAGS  += $(COMMONFLAGS)
CXXFLAGS += $(COMMONFLAGS)
CFLAGS += $(COMMONFLAGS)

COMPILATIONPHASE := 

OBJS :=

ifeq ($(cubin), 1)
	COMPILATIONPHASE += -cubin
	OBJS += $(CUFILES:$(SRCDIR)/%.cu=./$(SRCDIR)/%.cubin)
else
	COMPILATIONPHASE += -c
	OBJS += $(CCFILES:$(SRCDIR)/%.cpp=./$(SRCDIR)/%.o)
	OBJS += $(CUFILES:$(SRCDIR)/%.cu=./$(SRCDIR)/%.o)
endif



LDFLAGS += -L$(CUDA_INSTALL_PATH)/lib64 $(LIBS) 



all: ccbin $(TARGET)

$(SRCDIR)/%.cubin : $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $(COMPILATIONPHASE) $<
	
$(SRCDIR)/%.o : $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $(COMPILATIONPHASE) $<

$(SRCDIR)/%.o : $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS)  -c $< -o $@

$(TARGET): $(OBJS)
	$(LINK) -o $(TARGET) $(OBJS) $(LDFLAGS)

ccbin :
	mkdir ccbin
	ln -sf /usr/bin/g++ ccbin/g++
	
clean :
	rm -f $(OBJS)
	rm -f $(TARGET)
	rm -rf ccbin
