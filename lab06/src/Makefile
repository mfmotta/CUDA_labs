# Makefile for CUDA program
#
# Set the compiler and compiler flags
NVCC = /usr/local/cuda-12.1/bin/nvcc
NVCCFLAGS = -arch=sm_75 -fmad=false -Xptxas=-v -diag-suppress 177 
# -Xptxas=“-verbose” ptx assembler will report per kernel register and memory use.
#-DNDEBUG

# Set the name of the output executable : exercise01 or exercise02
EXEC = exercise02

#Set path to local directory
PROGRAM = $(shell pwd)

# Set the source files and object files: exercise01.cu or exercise02.cu
SRC = exercise02.cu
OBJ = $(SRC:.cu=.o)
#DEPS= helper_cuda.h helper_string.h 

# Rule to build the object files
%.o: %.cu $(DEPS)
#	$(NVCC) -c $< -o $@
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Rule to build the executable
$(EXEC): $(OBJ)
#	$(NVCC) $^ -o $@
	$(NVCC) $(NVCCFLAGS) $^ -o $@

# Rule to clean up object and executable files
clean:
	rm -f $(OBJ) $(EXEC)

# Rule to run the program
run: $(EXEC)
	cd $(PROGRAM) && ./$(EXEC)
#	./$(EXEC)