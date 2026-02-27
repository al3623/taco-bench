#include <stdlib.h>
#include "SpSpTTV.h"
#define max(a,b) ((a) > (b) ? a : b)
#define min(a,b) ((a) < (b) ? a : b)

void SpSpTTV(double *Bval, double *xval, int *Bpos1, int *Bpos2, int *Bpos3, int *xpos, int *Bcrd1, int *Bcrd2, int *Bcrd3, int *xcrd, int N, int M,double*output){
  for (int H32 = 0; H32 < Bpos1[1]; H32++) {
    for (int H33 = 0; H33 < Bpos2[H32 + 1] - (Bpos2[H32]); H33++) {
      int idxs_0 = 0;
      int idxs_1 = Bpos3[Bpos2[H32] + H33];
      while (idxs_0 < xpos[1] && idxs_1 < Bpos3[Bpos2[H32] + H33 + 1]) {
	int m = min(xcrd[idxs_0],Bcrd3[idxs_1]);
	if (xcrd[idxs_0] == Bcrd3[idxs_1]) {
	  output[(M) * (Bcrd1[H32]) + Bcrd2[Bpos2[H32] + H33]] +=
	    xval[idxs_0] * Bval[idxs_1];
	}
	idxs_0 += (int) (m == xcrd[idxs_0]);
	idxs_1 += (int) (m == Bcrd3[idxs_1]);
      }
    }
  }
}
