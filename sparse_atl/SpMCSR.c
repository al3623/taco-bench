#include <stdlib.h>
#include "SpMCSR.h"

void SpMCSR(double *VV, double *data, int *crd0, int *pos, int N,double*output){

for (int H22 = 0; H22 < N; H22++) {
for (int H23 = pos[H22]; H23 < pos[H22 + 1]; H23++) {
output[H22] += data[H23] * VV[crd0[H23]];
}
}
}
