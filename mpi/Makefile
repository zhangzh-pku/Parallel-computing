CC = mpicc
OBJ = *.o
EXE = allreduce
FLAGS = -O3 -march=native -g -Wall
all:${EXE}

allreduce: allreduce.c
	$(CC) -o $@ $^ $(FLAGS) 

clean:
	rm -f $(OBJ) $(EXE)
