# Makefile
#
# Set the compiler and compiler flags
NVCC = /usr/local/cuda-12.1/bin/nvcc
#NVCCFLAGS = -arch=sm_75

# Set path to local directory
PROGRAM = $(shell pwd)

# Get all .cu files containing "exercise" in the name
SRCS := $(wildcard *exercise*.cu)

# Create a list of corresponding executable names
EXECS := $(SRCS:.cu=)

# Create a list of corresponding object file names
OBJS := $(SRCS:.cu=.o)

# Rule to build all executables
all: $(EXECS)

# Rule to build each object file
$(OBJS): %.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Rule to build each executable
$(EXECS): %: %.o
	$(NVCC) $< -o $@

# Rule to clean up executables and object files
clean:
	rm -f $(EXECS) $(OBJS)

# Rule to run a specific executable
run: $(EXEC)
	cd $(PROGRAM) && ./$(EXEC)