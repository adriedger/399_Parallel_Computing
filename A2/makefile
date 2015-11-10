# Andre Driedger 1805536
# AS2 cuda makefile

#CC = gcc
#CFLAGS = -std=c99 -ltiff

all: greyscale blur 


greyscale: greyscale.cu
	nvcc -o $@ $^ -ltiff
	./greyscale lena.tif lena_bw.tif 
	./greyscale cinque_terre_small.tif cts_bw.tif
	
blur: blur.cu
	nvcc -o $@ $^ -ltiff
	./blur lena.tif lena_blr.tif
	./blur cinque_terre_small.tif cts_blr.tif

view:
	eog lena.tif &
	eog lena_bw.tif &
	eog lena_blr.tif &
	eog cinque_terre_small.tif &
	eog cts_bw.tif &
	eog cts_blr.tif &

#square: square.cu
#	nvcc -o $@ $^

#tiffcopy: tiffcopy.c
#	$(CC) -o $@ $^ $(CFLAGS)

#all: filter square tiffcopy

#clean:
#	rm filter square tiffcopy
