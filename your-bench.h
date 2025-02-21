#include "taco/tensor.h"

using namespace taco;
using namespace std;

// Add your include here
// ...
extern "C" {
	#include "SpMCSR.h"
	#include "SpMCSC.h"
	#include "SpMVDCSR.h"
}
 
void myValidate(Tensor<double> t, double *t2, int n) {
	double *t1 = (double *) (*t.getStorage()).vals;
	bool success = true;
	for (int i = 0; i < n; i++) {
		if (t1[i] != t2[i]) { 
			fprintf(stderr,"%d: %f <> %f\n",i,t1[i],t2[i]);
			success = false;
		}	
	}	
	if (success) {
		fprintf(stderr,"success!\n");
	}
}

void exprToYOUR(BenchExpr Expr, map<string,vector<Tensor<double>>> exprOperands,int repeat, taco::util::TimeResults timevalue) {

	Tensor<double> v = exprOperands.at("x")[0];
	Tensor<double> out = exprOperands.at("yRef")[0];

	TensorStorage vstor = v.getStorage();
	struct taco_tensor_t vstruct = *vstor;
	double *vvals = (double *)vstruct.vals;
	int vdim0 = vstruct.dimensions[0];

	vector<Tensor<double>> ms = exprOperands.at("A"); 
	for (Tensor<double> m : ms) {
		Tensor<double> v = exprOperands.at("x")[0];

		/*for (int i = 0; i < 3; i++) {
			fprintf(stderr,"%f,",vvals[i]);
		}*/
		if (m.getFormat()==CSR) {
			TensorStorage mstor = m.getStorage();
			struct taco_tensor_t mstruct = *mstor;
			
			int mdim0 = mstruct.dimensions[0];
			int *pos = (int *)mstruct.indices[1][0];
			int *crd = (int *)mstruct.indices[1][1];
			double *data = (double *)mstruct.vals;
			
			double *output = (double *) malloc(vdim0 * sizeof(double));
			TACO_BENCH(SpMCSR(vvals,data,crd,pos,vdim0,output);, 
								"\nATL CSR",repeat,timevalue,true)
			/*for (int i = 0; i < 12; i++) {
				fprintf(stderr,"%f,",data[i]);
			}*/
			myValidate(out,output,vdim0);
			free(output);
		}
		if (m.getFormat()==CSC) {
			TensorStorage mstor = m.getStorage();
			struct taco_tensor_t mstruct = *mstor;
			
			int mdim0 = mstruct.dimensions[0];
			int mdim1 = mstruct.dimensions[1];
			int *pos = (int *)mstruct.indices[1][0];
			int *crd = (int *)mstruct.indices[1][1];
			double *data = (double *)mstruct.vals;
			
			double *output = (double *) malloc(vdim0 * sizeof(double));
			TACO_BENCH(SpMCSC(vvals,data,pos,crd,mdim0,mdim1,output);, 
								"\nATL CSC",repeat,timevalue,true)
			myValidate(out,output,vdim0);
			free(output);
		}
		if (m.getFormat()==DCSR) {
			TensorStorage mstor = m.getStorage();
			struct taco_tensor_t mstruct = *mstor;

			int mdim0 = mstruct.dimensions[0];
			int mdim1 = mstruct.dimensions[1];

			int *pos0 = (int *)mstruct.indices[0][0];
			int *crd0 = (int *)mstruct.indices[0][1];
			
			int *pos1 = (int *)mstruct.indices[1][0];
			int *crd1 = (int *)mstruct.indices[1][1];
			
			double *data = (double *)mstruct.vals;
			
			double *output = (double *) malloc(vdim0 * sizeof(double));
			TACO_BENCH(SpMVDCSR(vvals,data,pos0,pos1,crd1,crd0,mdim0,mdim1,output);, 
								"\nATL DCSR",repeat,timevalue,true)
			myValidate(out,output,vdim0);
			free(output);
			
		}
		if (m.getFormat()==DCSC) {
			fprintf(stderr,"\nTODO: DCSC");
		}
	}
}

