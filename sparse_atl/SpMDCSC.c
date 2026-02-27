#include <stdlib.h>
#include "SpMDCSC.h"
#define max(a,b) ((a) > (b) ? a : b)
#define min(a,b) ((a) < (b) ? a : b)

void SpMDCSC(double *l, double *data, int *pos, int *pos0, int *crd1, int *crd2, int N,double*output){
for (int H22 = 0; H22 < pos[1]; H22++) {
for (int H23 = pos0[H22]; H23 < pos0[H22 + 1]; H23++) {
output[crd2[H23]] += data[H23] * l[crd1[H22]];
}
}
}
