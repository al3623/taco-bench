#include <stdlib.h>
#include "SpMSpVCSRCSF.h"
#define max(a,b) ((a) > (b) ? a : b)
#define min(a,b) ((a) < (b) ? a : b)

void SpMSpVCSRCSF(double *data,double *data0,int *crd2,int *crd1,int *pos,int *pos0,int N,double*output){

for (int H22 = 0; H22 < N; H22++) {
int idxs_0 = 0;
int idxs_1 = pos[H22];
while (idxs_0 < pos0[1] && idxs_1 < pos[H22 + 1]) {
int m = min(crd1[idxs_0],crd2[idxs_1]);
if (crd1[idxs_0] == crd2[idxs_1]) {
output[H22] += data0[idxs_0] * data[idxs_1];
}
idxs_0 += (int) (m == crd1[idxs_0]);
idxs_1 += (int) (m == crd2[idxs_1]);
}
}
}
