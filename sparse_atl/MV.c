#include <stdlib.h>
#include "MV.h"

void MV(double *MM, double *V, int N, int M, double*output){
	for (int i = 0; i < N; i ++) {
		for (int j = 0; j < M; j++) {
			output[j] = MM[i * M + j] * V[j];
		}
	}
}
