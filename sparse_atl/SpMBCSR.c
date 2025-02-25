#include <stdlib.h>
#include "SpMBCSR.h"

void SpMBCSR(double *data,double *l,int *crdd,int *pos,int M,int N,int K1,int K2,double*output){

for (int H26 = 0; H26 < N / (K1); H26++) {
for (int H27 = pos[H26]; H27 < pos[H26 + 1]; H27++) {

for (int H28 = 0; H28 < K1; H28++) {
double tmp2 = 0;
for (int H30 = 0; H30 < K2; H30++) {
tmp2 += data[((H27) * (K1) + H28) * (K2) + H30] * l[(crdd[H27]) * (K2) + H30];
}
// float x4 = tmp2;
output[(H26) * (K1) + H28] += tmp2; // x4;
}
}
}
}
