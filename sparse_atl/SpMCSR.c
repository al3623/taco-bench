#include <stdlib.h>
#include "SpMCSR.h"
#define max(a,b) ((a) > (b) ? a : b)
#define min(a,b) ((a) < (b) ? a : b)

void SpMCSR(double *VV, double *data, int *crd0, int *pos, int N,double*output){

for (int H22 = 0; H22 < N; H22++) {
double tmp2 = 0;
for (int H23 = pos[H22]; H23 < pos[H22 + 1]; H23++) {
tmp2 += data[H23] * VV[crd0[H23]];
}
double x4 = tmp2;
output[H22] = x4;
}
}
