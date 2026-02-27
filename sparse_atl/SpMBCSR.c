#include <stdlib.h>
#include "SpMBCSR.h"
#define max(a,b) ((a) > (b) ? a : b)
#define min(a,b) ((a) < (b) ? a : b)

void SpMBCSR(double *data,double *l,int *crd0,int *pos,int M,int N,int K1,int K2,double*output){

for (int H28 = 0; H28 < ((N) / (2)); H28++) {
for (int H29 = pos[H28]; H29 < pos[H28 + 1]; H29++) {

for (int H30 = 0; H30 < 2; H30++) {
double tmp2 = 0;
for (int H32 = 0; H32 < 4; H32++) {
tmp2 += data[((H29) * (2) + H30) * (4) + H32] * l[(crd0[H29]) * (4) + H32];
}
double x4 = tmp2;
output[(H28) * (2) + H30] += x4;
}
}
}
}
