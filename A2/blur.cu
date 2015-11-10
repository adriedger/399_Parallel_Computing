#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <tiffio.h>
#include <stdint.h>

__global__ void blur(uint8_t *d_out, uint8_t *d_in, int width, int height){
	
	int id = (blockIdx.x*blockDim.x)+threadIdx.x;
/*	float *h_filter;
	const float blurKernelSigma = 2;
	float filterSum = 0.f;
	for(int row = -height/2; row <= height/2; ++row){
			for(int col= -width/2; col <= width/2; ++col){
				float filterValue = expf(-(float)(col * col + row * row) / (2.f * blurKernelSigma *blurKernelSigma));
				(h_filter)[(row + height/2) * width + col + 
*/
	
	if(id < width*height){
		int x = id % width;
		int y = (id - x) / width;
		int filter_size = 4;
		float r_out = 0, g_out = 0, b_out = 0;
		int count = 0;
		for(int col = -filter_size; col <= filter_size; ++col){
			for(int row = -filter_size; row <= filter_size; ++row){
				if((x+col)>=0 && (x+col)<width && (y+row)>=0 && (y+row)<height){
					int surroundingIds = (id+col+(row*width))*3;
					r_out += d_in[surroundingIds];
					g_out += d_in[surroundingIds+1];
					b_out += d_in[surroundingIds+2];
					count++;
				}
			}
		}
		d_out[id*3] = r_out/count;
		d_out[id*3+1] = g_out/count;
		d_out[id*3+2] = b_out/count;
	}


//	d_out[id] = avg;
//	d_out[id] = d_in[id];
}

int main(int argc, char **argv){
  
	uint32_t    width, height;
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
	assert(TIFFGetField(iimage, TIFFTAG_IMAGELENGTH, &height));
	assert(height > 0);
	assert(TIFFGetField(iimage, TIFFTAG_BITSPERSAMPLE, &bits_per_sample) != 0);
	assert(bits_per_sample == 8);
	assert(TIFFGetField(iimage, TIFFTAG_PHOTOMETRIC, &photometric));
	assert(photometric == PHOTOMETRIC_RGB);
	assert(TIFFGetField(iimage, TIFFTAG_PLANARCONFIG, &planar_config) != 0);
	assert(TIFFGetField(iimage, TIFFTAG_SAMPLESPERPIXEL, &samples_per_pixel));
	assert(samples_per_pixel == 3);

	size = width * height * samples_per_pixel * sizeof(char);

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
	TIFFClose(iimage);

	char       *odata = (char *) malloc(size);
	uint8_t* d_in;
	cudaMalloc((void**) &d_in, size);
	cudaMemcpy(d_in, idata, size, cudaMemcpyHostToDevice);
	uint8_t* d_out;
	cudaMalloc((void**) &d_out, size);
	blur<<<size/width, width>>>(d_out, d_in, width, height);
	cudaMemcpy(odata, d_out, size, cudaMemcpyDeviceToHost);

	assert(odata != NULL);
	TIFF       *oimage = TIFFOpen(argv[2], "w");
	assert(oimage);

	assert(TIFFSetField(oimage, TIFFTAG_IMAGEWIDTH, width));
	assert(TIFFSetField(oimage, TIFFTAG_IMAGELENGTH, height));
	assert(TIFFSetField(oimage, TIFFTAG_BITSPERSAMPLE, bits_per_sample));
	assert(TIFFSetField(oimage, TIFFTAG_COMPRESSION, COMPRESSION_DEFLATE));
	assert(TIFFSetField(oimage, TIFFTAG_PHOTOMETRIC, photometric));
	assert(TIFFSetField(oimage, TIFFTAG_SAMPLESPERPIXEL, samples_per_pixel));
	assert(TIFFSetField(oimage, TIFFTAG_PLANARCONFIG, planar_config));
	assert(TIFFSetField(oimage, TIFFTAG_ROWSPERSTRIP, height));

	size_t    on = size;
	assert(TIFFWriteEncodedStrip(oimage, 0, odata, on) == on);
	TIFFClose(oimage);
	free(idata);
	free(odata);
	cudaFree(d_in);
	cudaFree(d_out);

	return 0;
}
