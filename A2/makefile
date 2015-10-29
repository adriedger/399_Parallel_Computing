# Andre Driedger 1805536
# AS2 cuda makefile

#CC = gcc
#CFLAGS = -std=c99 -ltiff

greyscale: greyscale.cu
	nvcc -o $@ $^ -ltiff

#square: square.cu
#	nvcc -o $@ $^

#tiffcopy: tiffcopy.c
#	$(CC) -o $@ $^ $(CFLAGS)

#all: filter square tiffcopy

#clean:
#	rm filter square tiffcopy
