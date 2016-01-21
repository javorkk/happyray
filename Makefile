#Source code directrory
SRCDIR    := src
INTERMDIR := obj

#default install location, use env var if invalid
CUDA_INSTALL_PATH=/usr/local/cuda

# Name of the execuatable to build
TARGET    := happyray

# Cuda source files (compiled with nvcc)
CUFILES   := $(SRCDIR)/RT/RTEngine.cu \
$(SRCDIR)/Application/CUDAApplication.cu \
$(SRCDIR)/RT/Algorithm/TLGridHierarchySortBuilder.cu \
$(SRCDIR)/RT/Algorithm/UniformGridBuildKernels.cu \
$(SRCDIR)/RT/Primitive/LightSource.cu \
$(SRCDIR)/RT/Structure/3DTextureMemoryManager.cu \
$(SRCDIR)/RT/Structure/TLGridHierarchyMemoryManager.cu \
$(SRCDIR)/RT/Structure/TLGridMemManager.cu \
$(SRCDIR)/RT/Structure/UGridMemoryManager.cu \
$(SRCDIR)/Utils/CUDAUtil.cu \
$(SRCDIR)/Utils/Scan.cu \
$(SRCDIR)/Utils/Sort.cu

MACHINE   := $(shell uname -s)
#Additional libraries
LIBS      := -lpng -lcudart -lGL -lGLU -lGLEW -lSDL2

# C/C++ source files (compiled with gcc / c++)
CCFILES := $(shell find $(SRCDIR) -iname '*.cpp')

################################################################################
# Rules and targets

.SUFFIXES : .cu 

# detect OS
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])

# Compilers
NVCC       := $(CUDA_INSTALL_PATH)/bin/nvcc
CXX        := g++
CC         := gcc
LINK       := g++ -fPIC

# Includes
INCLUDES  += -I$(SRCDIR) -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_SDK_DIR)/common/inc -I/usr/include/SDL -Icontrib/include
CXXFLAGS += $(INCLUDES)
CFLAGS += $(INCLUDES)
NVCCFLAGS += $(INCLUDES)


ifeq ($(sm_50), 1)
	NVCCFLAGS   += -arch sm_50 -D HAPPYRAY__CUDA_ARCH__=500
else ifeq ($(sm_40), 1)
	NVCCFLAGS   += -arch sm_40 -D HAPPYRAY__CUDA_ARCH__=400
else ifeq ($(sm_20), 1)
	NVCCFLAGS   += -arch sm_20 -D HAPPYRAY__CUDA_ARCH__=200
else ifeq ($(sm_13), 1)
	NVCCFLAGS   += -arch sm_13 -D HAPPYRAY__CUDA_ARCH__=130
else ifeq ($(sm_12), 1)
	NVCCFLAGS   += -arch sm_12 -D HAPPYRAY__CUDA_ARCH__=120
else ifeq ($(sm_11), 1)
	NVCCFLAGS   += -arch sm_11 -D HAPPYRAY__CUDA_ARCH__=110
else
	NVCCFLAGS   += -arch sm_50 -D HAPPYRAY__CUDA_ARCH__=500
endif

ifeq ($(dbg),1)
        COMMONFLAGS += -g
        NVCCFLAGS   += -D_DEBUG
else
        COMMONFLAGS += -O2
        NVCCFLAGS   += -DNDEBUG --compiler-options -fno-strict-aliasing -use_fast_math -maxrregcount=32
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



LDFLAGS += -L$(CUDA_INSTALL_PATH)/lib64 -L$(CUDA_SDK_DIR)/lib -L$(CUDA_SDK_DIR)/common/lib/$(OSLOWER) $(LIBS) 



all: $(TARGET)

$(SRCDIR)/%.cubin : $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $(COMPILATIONPHASE) $<
	
$(SRCDIR)/%.o : $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $(COMPILATIONPHASE) $<

$(SRCDIR)/%.o : $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS)  -c $< -o $@

$(TARGET): $(OBJS)
	$(LINK) -o $(TARGET) $(OBJS) $(LDFLAGS)

#ccbin :
#	mkdir ccbin
#	ln -sf /usr/bin/g++-4.1 ccbin/g++
#	ln -sf /usr/bin/gcc-4.1 ccbin/gcc
	
clean :
	rm -f $(OBJS)
	rm -f $(TARGET)
	rm -rf ccbin
