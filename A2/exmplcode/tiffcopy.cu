#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <tiffio.h>
#include <stdint.h>
/*
# this program extracts the pixels from a tiff image and copies them to a new image
# the names of the original and copy should be the parameters
#
# to compile:
# gcc -std=c99 tiffcopy.c -o tiffcopy -ltiff
#
# to run:
# ./tiffcopy lena.tif lena.copy.tif
*/
int main(int argc, char **argv)

{
  
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

  for (int i = 0; i < count; ++i) {
    tsize_t    in = TIFFReadEncodedStrip(iimage, i, curr, -1);
    assert(in != -1);
    curr += in;
  }

  TIFFClose(iimage);

  char       *odata = (char *) malloc(size);

  // copy the image, could've used memcpy too
  // FIXME: of course, you have to do more than copy the image :)
  for (int i = 0; i < size; i++) {
    odata[i] = idata[i];
  }

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

  tsize_t    on = size;
  assert(TIFFWriteEncodedStrip(oimage, 0, odata, on) == on);
  TIFFClose(oimage);
  free(idata);
  free(odata);

  return 0;
}
