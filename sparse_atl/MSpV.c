#include <stdlib.h>
#include "MSpV.h"

void MSpV(double *l, double *data, int *pos, int *crd0, int N, int M,double*output){

for (int H22 = 0; H22 < N; H22++) {
for (int H23 = 0; H23 < pos[1]; H23++) {
double tmp2 = 0;
tmp2 = l[(((M)) * (H22)) + crd0[H23]];
output[H22] += data[H23] * tmp2;
}
}
}
