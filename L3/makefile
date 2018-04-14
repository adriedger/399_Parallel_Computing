# Andre Driedger 1805536
# CMPT399 A1 matrixmulti makefile

CC = gcc
CFLAGS = -Wall -lpthread -msse3

super: ssemulti.c
	$(CC) -O -o $@ $^ $(CFLAGS)

simple: matrix2.c
	$(CC) -O -o $@ $^ $(CFLAGS)

bench: timer matrix
	./timer ./super 4096 1

times: times.dat times.gnuplot
	gnuplot < times.gnuplot

speedups: speedups.dat speedups.gnuplot
	gnuplot < speedups.gnuplot

all: super simple times speedups
