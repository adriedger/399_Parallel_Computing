#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <tiffio.h>
#include <stdint.h>

__global__ void greyscale(uint8_t *d_out, uint8_t *d_in, int size){
	
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id%3 == 0)
		d_out[id] = 0.299f * d_in[id] + 0.587f * d_in[id+1] + 0.114f * d_in[id+2];
	else if(id%3 == 1)
		d_out[id] = 0.299f * d_in[id-1] + 0.587f * d_in[id] + 0.114f * d_in[id+1];
	else
		d_out[id] = 0.299f * d_in[id-2] + 0.587f * d_in[id-1] + 0.114f * d_in[id];
	
//	d_out[id] = avg;
//	d_out[id] = d_in[id];
}

int main(int argc, char **argv){
  
	uint32_t    width, length;
	TIFF       *iimage;
	uint16_t    bits_per_sample, photometric;
	uint16_t    planar_config;
	uint16_t    samples_per_pixel;
	int size;

	assert(argc == 3);

	iimage = TIFFOpen(argv[1], "r");
	assert(iimage);
	assert(TIFFGetField(iimage, TIFFTAG_IMAGEWIDTH, &width));
	assert(width > 0);
	assert(TIFFGetField(iimage, TIFFTAG_IMAGELENGTH, &length));
	assert(length > 0);
	assert(TIFFGetField(iimage, TIFFTAG_BITSPERSAMPLE, &bits_per_sample) != 0);
	assert(bits_per_sample == 8);
	assert(TIFFGetField(iimage, TIFFTAG_PHOTOMETRIC, &photometric));
	assert(photometric == PHOTOMETRIC_RGB);
	assert(TIFFGetField(iimage, TIFFTAG_PLANARCONFIG, &planar_config) != 0);
	assert(TIFFGetField(iimage, TIFFTAG_SAMPLESPERPIXEL, &samples_per_pixel));
	assert(samples_per_pixel == 3);

	size = width * length * samples_per_pixel * sizeof(char);

	printf("size is %d\n",size);
	printf("spp is %d\n",samples_per_pixel);
	char     *idata = (char *) malloc(size);
	assert(idata != NULL);

	char     *curr = idata;
	int      count = TIFFNumberOfStrips(iimage);
	size_t in;
	for (int i = 0; i < count; ++i) {
		in = TIFFReadEncodedStrip(iimage, i, curr, -1);
//		assert(in != -1);
		printf("%li\n", in);
		curr += in;
	}

//	for(int i=0; i<size; i++)
//		printf("%c\n", idata[i]);

	TIFFClose(iimage);

	char       *odata = (char *) malloc(size);

	// copy the image, could've used memcpy too
	// FIXME: of course, you have to do more than copy the image :)
//	for (int i = 0; i < size; i++)
//		printf("%u\n", idata[i]);
//	memcpy(odata, idata, in);
	uint8_t* d_in;
	cudaMalloc((void**) &d_in, size);
	cudaMemcpy(d_in, idata, size, cudaMemcpyHostToDevice);
	uint8_t* d_out;
	cudaMalloc((void**) &d_out, size);

//	cudaMemcpy(d_out, odata, in, cudaMemcpyHostToDevice);
//	for(int i=0; i<9; i++)
//		printf("%u ", (uint8_t) idata[i]);
//	dim3 blockDim(512, 1, 1);
//	dim3 gridDim((uint8_t) ceil((double)(size/blockDim.x)), 1, 1); 
	greyscale<<<size/width, width>>>(d_out, d_in, size);
	
	cudaMemcpy(odata, d_out, size, cudaMemcpyDeviceToHost);

	assert(odata != NULL);
	TIFF       *oimage = TIFFOpen(argv[2], "w");
	assert(oimage);

	assert(TIFFSetField(oimage, TIFFTAG_IMAGEWIDTH, width));
	assert(TIFFSetField(oimage, TIFFTAG_IMAGELENGTH, length));
	assert(TIFFSetField(oimage, TIFFTAG_BITSPERSAMPLE, bits_per_sample));
	assert(TIFFSetField(oimage, TIFFTAG_COMPRESSION, COMPRESSION_DEFLATE));
	assert(TIFFSetField(oimage, TIFFTAG_PHOTOMETRIC, photometric));
	assert(TIFFSetField(oimage, TIFFTAG_SAMPLESPERPIXEL, samples_per_pixel));
	assert(TIFFSetField(oimage, TIFFTAG_PLANARCONFIG, planar_config));
	assert(TIFFSetField(oimage, TIFFTAG_ROWSPERSTRIP, length));

	size_t    on = size;
	assert(TIFFWriteEncodedStrip(oimage, 0, odata, on) == on);
	TIFFClose(oimage);
	free(idata);
	free(odata);
	cudaFree(d_in);
	cudaFree(d_out);
//	char       *tdata = (char *) malloc(size);	cudaMemcpy(d_out, tdata, in, cudaMemcpyHostToDevice);
//	for(int i=0; i<9; i++)
//		printf("%u ", (uint8_t) tdata[i]);
//	free(tdata);


	return 0;
}
