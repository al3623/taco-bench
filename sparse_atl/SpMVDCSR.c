#include <stdlib.h>
#include "SpMVDCSR.h"

void SpMVDCSR(double *MM, double *data, int *pos, int *pos0, int *crd1, int *crd2, int N, int M,double*output){
for (int H22 = 0; H22 < pos[1]; H22++) {
double tmp2 = 0;
for (int H23 = pos0[H22]; H23 < pos0[H22 + 1]; H23++) {
tmp2 += data[H23] * MM[crd2[H23]];
}
double x4 = tmp2;
output[crd1[H22]] += x4;
}
}
