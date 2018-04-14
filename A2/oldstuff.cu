// Andre Driedger 1805536
// A2 cuda greyscale source code

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <assert.h>
#include <stdint.h>
#include <tiffio.h>


__global__ void greyscale(float *d_out, float* r, float* g, float* b){
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int id = tid*(bid+1);
	int I = 0.299f * r[id] + 0.587f * g[id] +0.114f * b[id];
	//re-pack the data
	unsigned int pack = I << 24 | I << 16 | I << 8 | 255;
	d_out[id] = (float)pack;
}

int main(int argc, char **argv){
	TIFF* tif = TIFFOpen(argv[1], "r");
	uint32_t w, h;
	uint16_t  bits_per_sample, photometric, planar_config, samples_per_pixel;
	size_t npixels;
	uint32_t *raster, *raster_out;

	TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
	TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
//	
	assert(TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bits_per_sample) != 0);
	assert(bits_per_sample == 8);
	assert(TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &photometric));
	assert(photometric == PHOTOMETRIC_RGB);
	assert(TIFFGetField(tif, TIFFTAG_PLANARCONFIG, &planar_config) != 0);
	assert(TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samples_per_pixel));
	assert(samples_per_pixel == 3);
//
	npixels = w * h;
	raster = (uint32_t*) _TIFFmalloc(npixels * sizeof(uint32_t));
	TIFFReadRGBAImage(tif, w, h, raster, 0);
		
	int rArr[npixels], gArr[npixels], bArr[npixels];
	for(int i = 0; i<npixels; i++){
//		printf("^^%u^^ ", raster[i]);
//		printf("%u ", TIFFGetR(raster[i]));
		rArr[i] =  (int)TIFFGetR(raster[i]);

//		printf("%u ", TIFFGetG(raster[i]));
		gArr[i] =  (int)TIFFGetG(raster[i]);

//		printf("%u\n", TIFFGetB(raster[i]));
		bArr[i] =  (int)TIFFGetB(raster[i]);
	}

	float *d_out, *r, *g, *b;
	cudaMalloc((void**) &d_out, npixels * sizeof(float));
	cudaMalloc((void**) &r, npixels * sizeof(float));
	cudaMemcpy(r, rArr, npixels * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((void**) &g, npixels * sizeof(float));
	cudaMemcpy(g, gArr, npixels * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((void**) &b, npixels * sizeof(float));
	cudaMemcpy(b, bArr, npixels * sizeof(float), cudaMemcpyHostToDevice);

	greyscale<<<npixels/1024, 1024>>>(d_out, r, g, b);

	char* odata = (char*) malloc(npixels*sizeof(char));
	cudaMemcpy(odata, d_out, npixels * sizeof(char), cudaMemcpyDeviceToHost);
	for(int i=0; i<npixels; i++){
		printf("%d\n", odata[i]);
	}

	TIFF *tif_out = TIFFOpen(argv[2], "w");
	assert(tif_out);
	
	assert(TIFFSetField(tif_out, TIFFTAG_IMAGEWIDTH, w));
	assert(TIFFSetField(tif_out, TIFFTAG_IMAGELENGTH, h));
//
	assert(TIFFSetField(tif_out, TIFFTAG_BITSPERSAMPLE, bits_per_sample));
	assert(TIFFSetField(tif_out, TIFFTAG_COMPRESSION, COMPRESSION_DEFLATE));
	assert(TIFFSetField(tif_out, TIFFTAG_PHOTOMETRIC, photometric));
	assert(TIFFSetField(tif_out, TIFFTAG_SAMPLESPERPIXEL, samples_per_pixel));
	assert(TIFFSetField(tif_out, TIFFTAG_PLANARCONFIG, planar_config));
	assert(TIFFSetField(tif_out, TIFFTAG_ROWSPERSTRIP, h));
//	
	size_t on = npixels * sizeof(uint32_t);
	assert(TIFFWriteRawStrip(tif_out, 0, raster_out, on) == on);
	TIFFClose(tif_out);
//	free(idata);
//	free(odata);
	
	return 0;
	
}


