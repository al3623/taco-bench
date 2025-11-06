#include <stdlib.h>
#include "SpMCSC.h"
#define max(a,b) ((a) > (b) ? a : b)
#define min(a,b) ((a) < (b) ? a : b)

void SpMCSC(double *l, double *data, int *pos, int *crd0, int N, int M,double*output){
for (int H22 = 0; H22 < M; H22++) {
for (int H23 = pos[H22]; H23 < pos[H22 + 1]; H23++) {
output[crd0[H23]] += data[H23] * l[H22];
}
}
}
