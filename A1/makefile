# Andre Driedger 1805536
# CMPT399 A1 matrixmulti makefile

CC = gcc
CFLAGS = -Wall -lpthread

matrix: matrix.c
	$(CC) -O -o $@ $^ $(CFLAGS)

bench: timer matrix
	./timer ./matrix 4096 1
