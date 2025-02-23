#include <stdlib.h>
#include "SpMB.h"

void SpMB(double *data, double *l, int *crd0, int *pos, int M, int N, int K,double*output){

for (int H23 = 0; H23 < N; H23++) {
for (int H24 = 0; H24 < M; H24++) {
for (int H25 = 0; H25 < pos[((H23) / (K)) + 1] - (pos[((H23) / (K))]); H25++) {
if (crd0[pos[((H23) / (K))] + H25] == ((H23) % (K))) {
output[H23] += data[(pos[((H23) / (K))] + H25) * (M) + H24] * l[H24];
}
}
}
}
}
