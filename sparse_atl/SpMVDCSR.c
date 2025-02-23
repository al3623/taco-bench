#include <stdlib.h>
#include "SpMVDCSR.h"

void SpMVDCSR(double *MM, double *data, int *pos, int *pos0, int *crd1, int *CRD, int N, int M,double*output){
for (int H22 = 0; H22 < pos[1]; H22++) {
for (int H23 = pos0[H22]; H23 < pos0[H22 + 1]; H23++) {
output[crd1[H22]] += data[H23] * MM[CRD[H23]];
}
}
}
