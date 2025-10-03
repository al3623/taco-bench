#include <stdlib.h>
#include "SpMDIA.h"

void SpMDIA(double *data, double *l, int *crd0, int *pos, int M, int N,double*output){

for (int H23 = 0; H23 < N; H23++) {
for (int H24 = 0; H24 < M + N - (1); H24++) {
for (int H25 = 0; H25 < pos[H24 + 1] - (pos[H24]); H25++) {
if (H23 - ((((0) < (N - ((((N) < (M)) ? (N) : (M))) - ((((0) < (H24 + H23 - (N) + 1 - (H23) + N - (1) - ((((N) < (M)) ? (N) : (M)) - (1)))) ? (H24 + H23 - (N) + 1 - (H23) + N - (1) - ((((N) < (M)) ? (N) : (M)) - (1))) : (0))))) ? (N - ((((N) < (M)) ? (N) : (M))) - ((((0) < (H24 + H23 - (N) + 1 - (H23) + N - (1) - ((((N) < (M)) ? (N) : (M)) - (1)))) ? (H24 + H23 - (N) + 1 - (H23) + N - (1) - ((((N) < (M)) ? (N) : (M)) - (1))) : (0)))) : (0))) == crd0[pos[H24] + H25] && H24 + H23 - (N) + 1 < M && 0 <= H24 + H23 - (N) + 1) {
output[H23] += data[pos[H24] + H25] * l[H24 + H23 - (N) + 1];
}
}
}
}
}
