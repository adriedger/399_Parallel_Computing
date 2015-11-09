# Andre Driedger 1805536
# AS2 cuda makefile

#CC = gcc
#CFLAGS = -std=c99 -ltiff


bleh: greyscale.cu
	nvcc -o $@ $^ -ltiff
	./bleh lena.tif copy.tif
	eog copy.tif


greyscale: greyscale.cu
	nvcc -o $@ $^ -ltiff

test: greyscale2.cu
	nvcc -o $@ $^ -ltiff

#square: square.cu
#	nvcc -o $@ $^

#tiffcopy: tiffcopy.c
#	$(CC) -o $@ $^ $(CFLAGS)

#all: filter square tiffcopy

#clean:
#	rm filter square tiffcopy
