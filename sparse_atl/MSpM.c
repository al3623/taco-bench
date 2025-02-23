#include <stdlib.h>
#include "MSpM.h"

void MSpM(double *MM, double *data, int *pos, int *crd0, int N, int M, int L,double*output){

for (int H29 = 0; H29 < N; H29++) {

for (int H30 = 0; H30 < L; H30++) {
for (int H31 = 0; H31 < pos[H29 + 1] - (pos[H29]); H31++) {
float tmp2 = 0;
tmp2 = MM[(((L)) * (crd0[pos[H29] + H31])) + H30];
output[(L) * (H29) + H30] += data[pos[H29] + H31] * tmp2;
}
}
}
}
