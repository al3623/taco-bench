#include <stdlib.h>
#include "SpMBCSR.h"

void SpMBCSR(double *data,double *l,int *crdd,int *pos,int M,int N,int K1,int K2,double*output){

for (int H26 = 0; H26 < ((N) / (K1)); H26++) {
for (int H27 = pos[H26]; H27 < pos[H26 + 1]; H27++) {

for (int H28 = 0; H28 < K1; H28++) {
for (int H29 = 0; H29 < K2; H29++) {
output[(H26) * (K1) + H28] += data[((H27) * (K1) + H28) * (K2) + H29] * l[(crdd[H27]) * (K2) + H29];
}
}
}
}
}
