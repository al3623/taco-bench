#include <stdlib.h>
#include "MTTKRP.h"

void MTTKRP(double *data, double *CC, double *BB, int *pos0, int *pos1, int *CRD1, int *CRD2, int *CRD, int *crd0, int *pos, int S,double*output){
for (int H51 = 0; H51 < pos[1] - (pos[0]); H51++) {
for (int H52 = 0; H52 < pos0[H51 + 1] - (pos0[H51]); H52++) {
for (int H53 = 0; H53 < pos1[pos0[H51] + H52 + 1] - (pos1[pos0[H51] + H52]); H53++) {

for (int H54 = 0; H54 < S; H54++) {
double tmp2 = 0;
double tmp3 = 0;
tmp3 = BB[(((S)) * (CRD2[pos1[pos0[H51] + H52] + H53])) + H54];
tmp2 = data[pos1[pos0[H51] + H52] + H53] * tmp3;
double tmp4 = 0;
tmp4 = CC[(((S)) * (CRD1[pos0[H51] + H52])) + H54];
output[(S) * (CRD[H51]) + H54] += tmp2 * tmp4;
}
}
}
}
}
