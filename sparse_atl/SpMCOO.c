#include <stdlib.h>
#include "SpMCOO.h"

void SpMCOO(double *l, double *data, int *coo0, int *coo1, int *pos, int N,double*output){
for (int H22 = 0; H22 < pos[1]; H22++) {
output[coo0[H22]] += data[H22] * l[coo1[H22]];
}
}
