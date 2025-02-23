#include <stdlib.h>
#include "SpMDCSC.h"

void SpMDCSC(double *l, double *data, int *pos, int *pos0, int *crd1, int *CRD, int N,double*output){
for (int H22 = 0; H22 < pos[1]; H22++) {
for (int H23 = pos0[H22]; H23 < pos0[H22 + 1]; H23++) {
output[CRD[H23]] += data[H23] * l[crd1[H22]];
}
}
}
