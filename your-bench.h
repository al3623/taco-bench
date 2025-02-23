#include "taco/tensor.h"

using namespace taco;
using namespace std;

// Add your include here
// ...
extern "C" {
	#include "SpMCSR.h"
	#include "SpMCSC.h"
	#include "SpMVDCSR.h"
	#include "SpMDCSC.h"
	#include "SpMCOO.h"
	#include "SpMBCSR.h"
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
		// fprintf(stderr,"success!\n");
	}
}

void exprToYOUR(BenchExpr Expr, map<string,vector<Tensor<double>>> exprOperands,int repeat, taco::util::TimeResults timevalue) {

	Tensor<double> v = exprOperands.at("x")[0];
	Tensor<double> out = exprOperands.at("yRef")[0];

	TensorStorage vstor = v.getStorage();
	struct taco_tensor_t vstruct = *vstor;
	double *vvals = (double *)vstruct.vals;
	int vdim0 = vstruct.dimensions[0];

	/*for (int i = 0; i < 3; i ++) {
		fprintf(stderr, "%f,", ((double *)((*(out.getStorage())).vals))[i]);
	}*/
	vector<Tensor<double>> ms = exprOperands.at("A");
	for (Tensor<double> m : ms) {
		Tensor<double> v = exprOperands.at("x")[0];

		if (m.getFormat()==CSR) {
			TensorStorage mstor = m.getStorage();
			struct taco_tensor_t mstruct = *mstor;
			
			int mdim0 = mstruct.dimensions[0];
			int *pos = (int *)mstruct.indices[1][0];
			int *crd = (int *)mstruct.indices[1][1];
			double *data = (double *)mstruct.vals;
			
			double *output = (double *) calloc(mdim0, sizeof(double));
			TACO_BENCH(SpMCSR(vvals,data,crd,pos,mdim0,output);, 
								"ATL CSR",repeat,timevalue,true)
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
			
			double *output = (double *) calloc(vdim0,sizeof(double));
			TACO_BENCH(SpMCSC(vvals,data,pos,crd,mdim0,mdim1,output);, 
								"ATL CSC",repeat,timevalue,true)
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
			
			double *output = (double *) calloc(vdim0, sizeof(double));
			TACO_BENCH(SpMVDCSR(vvals,data,pos0,pos1,crd0,crd1,mdim0,mdim1,output);, 
								"ATL DCSR",repeat,timevalue,true)
			myValidate(out,output,vdim0);
			free(output);
			
		}
		if (m.getFormat()==DCSC) {
			TensorStorage mstor = m.getStorage();
			struct taco_tensor_t mstruct = *mstor;

			int mdim0 = mstruct.dimensions[0];
			int mdim1 = mstruct.dimensions[1];

			int *pos0 = (int *)mstruct.indices[0][0];
			int *crd0 = (int *)mstruct.indices[0][1];
			
			int *pos1 = (int *)mstruct.indices[1][0];
			int *crd1 = (int *)mstruct.indices[1][1];
			
			double *data = (double *)mstruct.vals;
			
			double *output = (double *) calloc(vdim0, sizeof(double));
			TACO_BENCH(SpMDCSC(vvals,data,pos0,pos1,crd0,crd1,mdim0,output);, 
								"ATL DCSC",repeat,timevalue,true)
			myValidate(out,output,vdim0);
			free(output);
		}
		if (m.getFormat() == COO(2)) {
			TensorStorage mstor = m.getStorage();
			struct taco_tensor_t mstruct = *mstor;

			int mdim0 = mstruct.dimensions[0];
			int mdim1 = mstruct.dimensions[1];

			int *pos = (int *)mstruct.indices[0][0];
			int *crd0 = (int *)mstruct.indices[0][1];
			int *crd1 = (int *)mstruct.indices[1][1];
			
			double *data = (double *)mstruct.vals;
			
			double *output = (double *) calloc(vdim0, sizeof(double));
			TACO_BENCH(SpMCOO(vvals,data,crd0,crd1,pos,mdim0,output);, 
								"ATL COO",repeat,timevalue,true)
			myValidate(out,output,vdim0);
			free(output);
		}

		if (isDense(m.getFormat())) {
			// let's try BCSR
			int blockSize1 = 32;
			int blockSize2 = 32;
			int rows= m.getDimension(0);
			int cols= m.getDimension(1);
			
			// Tile the input matrix
			Tensor<double> Ab(
							{rows/blockSize1,cols/blockSize2,blockSize1,blockSize2},
							Format({Dense,Sparse,Dense,Dense}));


			for (auto& value : iterate<double>(m)) {
				Ab.insert({value.first[0]/blockSize1
								, value.first[1]/blockSize2
								, value.first[0]%blockSize1
								, value.first[1]%blockSize2}
								, value.second);
			}
  			Ab.pack();
			
			TensorStorage mstor = Ab.getStorage();
			struct taco_tensor_t mstruct = *mstor;

			int mdim0 = mstruct.dimensions[0];
			int mdim1 = mstruct.dimensions[1];
			int mdim2 = mstruct.dimensions[2];
			int mdim3 = mstruct.dimensions[3];

			int *pos = (int *)mstruct.indices[1][0];
			int *crd = (int *)mstruct.indices[1][1];
			double *data = (double *)mstruct.vals;
			
			double *output = (double *) calloc(vdim0, sizeof(double));
			
			TACO_BENCH(SpMBCSR(data,vvals,crd,pos,rows,cols,blockSize1,blockSize2,output);, 
								"ATL BCSR",repeat,timevalue,true) 
			myValidate(out,output,vdim0);

			free(output);
			
		}
	}
}

