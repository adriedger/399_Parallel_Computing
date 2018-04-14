#include <stdio.h>
#include <stdlib.h>
#include <math.h>

float* make_matrix(const int blurKernelWidth);

int main(int argc, char ** argv) {

  float simple_matrix[] = {0.0f, 0.2f, 0.0f, 0.2f, 0.2f, 0.2f, 0.0f, 0.2f, 0.0f};

  make_matrix(3);
  printf("\n");
  make_matrix(9);

}

float* make_matrix(const int blurKernelWidth) {

  float *h_filter;

  //now create the filter that they will use
  const float blurKernelSigma = 2.;

  //create and fill the filter we will convolve with
  h_filter = (float *)malloc(sizeof(float)* blurKernelWidth * blurKernelWidth);

  float filterSum = 0.f; //for normalization

  for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
    for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
      float filterValue = expf( -(float)(c * c + r * r) / (2.f * blurKernelSigma * blurKernelSigma));
      (h_filter)[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] = filterValue;
      filterSum += filterValue;
    }
  }

  float normalizationFactor = 1.f / filterSum;

  for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
    for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
      h_filter[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] *= normalizationFactor;
    }
  }

  //blurred

  for (int r = 0; r < blurKernelWidth; ++r) {
    for (int c = 0; c < blurKernelWidth; ++c) {
      printf("%f ", h_filter[r*blurKernelWidth + c]);
    }
    printf("\n");
  }
  return h_filter;
}


