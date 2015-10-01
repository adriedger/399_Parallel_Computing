# Andre Driedger
# Lab1 cmpt399 pthreads

CC = gcc
CFLAGS = -Wall -lpthread

#all: lab1 L1-fixed

#lab1: lab1.c
#	$(CC) -o $@ $^ $(CFLAGS)

#lab1p2: lab1p2.c
#	$(CC) -o $@ $^ $(CFLAGS)

L1modified: L1modified.c
	$(CC) -o -O $@ $^ $(CFLAGS)

bench: L1modified mytime
	./mytime ./L1modified
