# Andre Driedger 1805536
# example code makefile

all: filter square tiffcopy

filter: make_filter.cu
	nvcc -o $@ $^

square: square.cu
	nvcc -o $@ $^

tiffcopy: tiffcopy.cu
	nvcc -o $@ $^ -ltiff

clean:
	rm filter square tiffcopy
