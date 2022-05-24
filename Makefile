# NOTE: Update the following variables for your system
CC=gcc
CXX=g++
NVCC=nvcc
MPICC=mpicc
MPICXX=mpic++
LD=${CXX}
RTLD_PATH=/lib64/ld-2.17.so
CUDA_INCLUDE_PATH=/usr/local/cuda/include/

# The name will be the same as the current directory name.
KERNEL_LOADER=kernel_loader

# Wrapper library against which the target application will be linked.
CUDA_WRAPPER_LIB=cuda_wrappers
MPI_WRAPPER_LIB=mpi_wrappers
CUDA_STUB_LIB=cuda_stub
MPI_STUB_LIB=mpi_stub

CUDA_LH_BIN=cuda_lh.exe
MPI_LH_BIN=mpi_lh.exe

# Flags for compile and link configurations

NVCC_LFLAGS=-Xlinker -Ttext-segment -Xlinker 0x800000 --cudart shared
NVCC_CFLAGS=-Xcompiler -g3 -O0

MPI_LFLAGS=-lmpi
MPI_CFLAGS=

INCLUDE_FLAGS=-I. -I${CUDA_INCLUDE_PATH}

WARNING_FLAGS=-Wall -Wno-deprecated-declarations -Werror

override CFLAGS += -g3 -O0 -fPIC ${INCLUDE_FLAGS} -c -std=gnu11 \
                ${WARNING_FLAGS} -fno-stack-protector
override CXXFLAGS += -g3 -O0 -fPIC ${INCLUDE_FLAGS} -c -std=c++11 \
                  ${WARNING_FLAGS} -fno-stack-protector

# variables related to kernel loader
KERNEL_LOADER_OBJS=loader/kernel_loader.o loader/custom_loader.o mpi_lh/lower_half_mpi_if.o cuda_lh/lower_half_cuda_if.o \
		    loader/mmap_wrapper.o loader/sbrk_wrapper.o common/switch_context.o utils/procmapsutils.o utils/trampoline_setup.o utils/utils.o

KERNEL_LOADER_CFLAGS=-DSTANDALONE
KERNEL_LOADER_BIN=kernel_loader.exe

TEST_OBJS=simple_hello_world.o
TEST_EXE=simple_hello_world.exe

default: lib${CUDA_STUB_LIB}.so lib${MPI_STUB_LIB}.so ${TEST_EXE} \
   ${CUDA_LH_BIN} ${MPI_LH_BIN} ${KERNEL_LOADER_BIN}  lib${CUDA_WRAPPER_LIB}.so lib${MPI_WRAPPER_LIB}.so 

disableASLR:
	@- [ `cat /proc/sys/kernel/randomize_va_space` = 0 ] \
	|| sudo sh -c 'echo 0 > /proc/sys/kernel/randomize_va_space'

enableASLR:
	@- [ `cat /proc/sys/kernel/randomize_va_space` != 2 ] \
	&& sudo sh -c 'echo 2 > /proc/sys/kernel/randomize_va_space'

check: default
	${DMTCP_LAUNCH} ${DMTCP_LAUNCH_FLAGS} $$PWD/test/${TARGET_BIN}

.c.o:
	${CC} ${CFLAGS} $< -o $@

.cpp.o:
	${CXX} ${CXXFLAGS} $< -o $@

${TEST_OBJS}: test/simple_hello_world.c
	${CC} ${INCLUDE_FLAGS} ${NVCC_OPTFLAGS} -c $< -o test/$@

${TEST_EXE}: ${TEST_OBJS}
	${CXX} test/$< -o test/$@ -L. -l${CUDA_STUB_LIB} -l${MPI_STUB_LIB} ;
	${NVCC} -g test/$< -o test/${TEST_EXE}.native ${MPI_LFLAGS}

${KERNEL_LOADER_BIN}: ${KERNEL_LOADER_OBJS}
	${CXX} $^ -o $@ ${MPI_LFLAGS} -L/usr/local/cuda/lib64 -lcudart -ldl

${CUDA_LH_BIN}: cuda_lh/cuda_lh.o
	${NVCC} ${NVCC_FLAGS} $^ -o $@ -lcuda -ldl

${MPI_LH_BIN}: mpi_lh/mpi_lh.o
	${MPICC} ${MPI_OPTFLAGS} $^ -o $@ ${MPI_FLAGS}

lib${CUDA_WRAPPER_LIB}.so: cuda_wrappers/cuda_wrappers.o
	${CC} -shared -fPIC -g3 -O0 -o $@ $^

lib${MPI_WRAPPER_LIB}.so: mpi_wrappers/mpi_wrappers.o
	${CC} -shared -fPIC -g3 -O0 -o $@ $^

lib${CUDA_STUB_LIB}.so: cuda_wrappers/cuda_stub.o
	${CC} -shared -fPIC -g3 -O0 -o $@ $^

lib${MPI_STUB_LIB}.so: mpi_wrappers/mpi_stub.o
	${CC} -shared -fPIC -g3 -O0 -o $@ $^

run: ${KERNEL_LOADER_BIN} ${TEST_EXE}
	TARGET_LD=${RTLD_PATH} ./$< $$PWD/test/${TARGET_BIN} arg1 arg2 arg3

gdb: ${KERNEL_LOADER_BIN} ${TEST_EXE}
	TARGET_LD=${RTLD_PATH} gdb --args ./$< $$PWD/test/${TEST_EXE} arg1 arg2 arg3

vi vim:
	vim ${FILE}.cpp

tags:
	gtags .

dist: clean
	(dir=`basename $$PWD` && cd .. && tar zcvf $$dir.tgz $$dir)
	(dir=`basename $$PWD` && ls -l ../$$dir.tgz)

tidy:
	rm -f *.exe

clean: tidy
	rm -f */*.o *.so test/*.o test/*.exe*

.PHONY: dist vi vim clean gdb tags tidy enableASLR disableASLR check
