#include <stdlib.h>
#include "SpMSpVTTV.h"
#define max(a,b) ((a) > (b) ? a : b)
#define min(a,b) ((a) < (b) ? a : b)

void SpMSpVTTV(double *data, double *data0, int *pos, int *pos0, int *pos1, int *pos2, int *CRD2, int *crd3, int *crd4, int *crd5, int M, int N,double*output){
for (int H29 = 0; H29 < pos[1]; H29++) {
for (int H30 = 0; H30 < pos0[H29 + 1] - (pos0[H29]); H30++) {
int idxs_0 = 0;
int idxs_1 = pos1[pos0[H29] + H30];
while (idxs_0 < pos2[1] && idxs_1 < pos1[pos0[H29] + H30 + 1]) {
int m = min(crd4[idxs_0],CRD2[idxs_1]);
if (crd4[idxs_0] == CRD2[idxs_1]) {
output[(M) * (crd3[H29]) + crd5[pos0[H29] + H30]] += data0[idxs_0] * data[idxs_1];
}
idxs_0 += (int) (m == crd4[idxs_0]);
idxs_1 += (int) (m == CRD2[idxs_1]);
}
}
}
}
