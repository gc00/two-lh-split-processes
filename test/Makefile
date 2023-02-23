#	nvcc -o simple_hello_world ./simple_hello_world.cu _lmpi
FILE=simple_hello_world

# ${FILE}.exe

build: ${FILE}.exe

${FILE}.exe: ${FILE}.c
	mpicc -o simple_hello_world.exe ./simple_hello_world.c

check: simple_hello_world.exe
	mpirun -np 2 ./$<

clean:
	rm -f simple_hello_world.exe simple_hello_world.o
