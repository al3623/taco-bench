#include <stdlib.h>
#include "SpMSpVCSRCSF.h"

void SpMSpVCSRCSF(double *data,double *data0,int *crd2,int *crd1,int *pos,int *pos0,int N,double*output){

for (int H22 = 0; H22 < N; H22++) {
for (int H23 = 0; H23 < pos[H22 + 1] - (pos[H22]); H23++) {
for (int H24 = 0; H24 < pos0[1]; H24++) {
if (crd1[H24] == crd2[pos[H22] + H23]) {
output[H22] += data[pos[H22] + H23] * data0[H24];
}
}
}
}
}
