#include <stdlib.h>
#include "TTV.h"
#define max(a,b) ((a) > (b) ? a : b)
#define min(a,b) ((a) < (b) ? a : b)

void TTV(double *data, double *BB, int *pos, int *pos0, int *pos1, int *CRD2, int *crd3, int *crd2, int M, int N,double*output){
for (int H29 = 0; H29 < pos[1]; H29++) {
for (int H30 = 0; H30 < pos0[H29 + 1] - (pos0[H29]); H30++) {
for (int H31 = 0; H31 < pos1[pos0[H29] + H30 + 1] - (pos1[pos0[H29] + H30]); H31++) {
output[(M) * (crd2[H29]) + crd3[pos0[H29] + H30]] += data[pos1[pos0[H29] + H30] + H31] * BB[CRD2[pos1[pos0[H29] + H30] + H31]];
}
}
}
}
