CC = g++
CFLAGS = -lm -pthread -Ofast -march=native -Wall -funroll-loops -Wno-unused-result -lgsl -lm -lgslcblas

pl_warp: pl_warp.cpp utils.o hierarchy.o
	$(CC) $(CFLAGS) -o pl_warp pl_warp.cpp utils.o hierarchy.o

utils.o: utils.cpp
	$(CC) $(CFLAGS) -c utils.cpp

hierarchy.o: hierarchy.cpp
	$(CC) $(CFLAGS) -c hierarchy.cpp

clean :
	rm -rf *.o pl_warp
