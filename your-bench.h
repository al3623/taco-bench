#include "taco/tensor.h"

using namespace taco;
using namespace std;

// Add your include here
// ...
extern "C" {
        #include "SpMSpVCSRCSF.h"
	#include "SpMCSR.h"
	#include "SpMCSC.h"
	#include "SpMVDCSR.h"
	#include "SpMDCSC.h"
	#include "SpMCOO.h"
	#include "SpMBCSR.h"
	#include "TTV.h"
	#include "SpSpTTV.h"
	#include "MTTKRP.h"
}

#define ATL_TIME_REPEAT(SETUP, CODE, TAKEDOWN, ONCE, REPEAT, RES, COLD) {  \
    taco::util::Timer timer;                         \
    for(int i=0; i<REPEAT; i++) {                    \
      SETUP;                                 \
      if(COLD)                                       \
        timer.clear_cache();                         \
      timer.start();                                 \
      CODE;                                          \
      timer.stop();                                  \
	  if (i == 0) { \
		ONCE; \
	  } \
      TAKEDOWN;                                  \
    }                                                \
    RES = timer.getResult();                         \
  }

void myValidate(Tensor<double> t, double *t2, int n) {
  double *t1 = (double *) (*t.getStorage()).vals;
  bool success = true;
  for (int i = 0; i < n; i++) {
    if (fabs(t1[i] -  t2[i]) / fabs(t1[i]) > 1e-2) {
      // fprintf(stderr,"%d: %lf <> %lf\n",i,t1[i],t2[i]);
      success = false;
    }	
  }	
  if (!success) {
    fprintf(stdout,"failed :(\n");
  }
}

void exprToYOUR(BenchExpr Expr, map<string,vector<Tensor<double>>> exprOperands,std::vector<double> Sparsities,int repeat, taco::util::TimeResults timevalue) {

  if (Expr==SpMV) {
    Tensor<double> v = exprOperands.at("x")[0];
    Tensor<double> out = exprOperands.at("yRef")[0];

    TensorStorage vstor = v.getStorage();
    struct taco_tensor_t vstruct = *vstor;
    double *vvals = (double *)vstruct.vals;
    int vdim0 = vstruct.dimensions[0];
    vector<Tensor<double>> ms = exprOperands.at("A");
    for (Tensor<double> m : ms) {
      if (m.getFormat()==CSR) {
	TensorStorage mstor = m.getStorage();
	struct taco_tensor_t mstruct = *mstor;
	    
	int mdim0 = mstruct.dimensions[0];
	int *pos = (int *)mstruct.indices[1][0];
	int *crd = (int *)mstruct.indices[1][1];
	double *data = (double *)mstruct.vals;
	double *output;
	ATL_TIME_REPEAT(output = (double *) calloc(mdim0, sizeof(double))
			, SpMCSR(vvals,data,crd,pos,mdim0,output)
			, free(output)
			, myValidate(out,output,mdim0)
			, repeat, timevalue, true)
	  cout << "ATL CSR\n" << timevalue << endl;	
      } else if (m.getFormat()==CSC) {
	TensorStorage mstor = m.getStorage();
	struct taco_tensor_t mstruct = *mstor;
	    
	int N = mstruct.dimensions[0];
	int M = mstruct.dimensions[1];
	int *pos = (int *)mstruct.indices[1][0];
	int *crd = (int *)mstruct.indices[1][1];
	double *data = (double *)mstruct.vals;
	double *output;
	
	ATL_TIME_REPEAT(output = (double *) calloc(N, sizeof(double))
			, SpMCSC(vvals,data,pos,crd,M,N,output)
			, free(output)
			, myValidate(out,output,vdim0)
			, repeat, timevalue, true)
	  cout << "ATL CSC\n" << timevalue << endl;	
      } else if (m.getFormat()==DCSR) {
	TensorStorage mstor = m.getStorage();
	struct taco_tensor_t mstruct = *mstor;
	
	int mdim0 = mstruct.dimensions[0];
	int mdim1 = mstruct.dimensions[1];
	    
	int *pos0 = (int *)mstruct.indices[0][0];
	int *crd0 = (int *)mstruct.indices[0][1];
			
	int *pos1 = (int *)mstruct.indices[1][0];
	int *crd1 = (int *)mstruct.indices[1][1];
			
	double *data = (double *)mstruct.vals;
	double *output;

	ATL_TIME_REPEAT(output = (double *) calloc(mdim0, sizeof(double))
			,  SpMVDCSR(vvals,data,pos0,pos1,crd0,crd1,mdim0,mdim1,output)
			, free(output)
			, myValidate(out,output,vdim0)
			, repeat, timevalue, true)
	  cout << "ATL DCSR\n" << timevalue << endl;
      } else if (m.getFormat()==DCSC) {
	TensorStorage mstor = m.getStorage();
	struct taco_tensor_t mstruct = *mstor;
	    
	int mdim0 = mstruct.dimensions[0];
	int mdim1 = mstruct.dimensions[1];
	    
	int *pos0 = (int *)mstruct.indices[0][0];
	int *crd0 = (int *)mstruct.indices[0][1];
			
	int *pos1 = (int *)mstruct.indices[1][0];
	int *crd1 = (int *)mstruct.indices[1][1];
			
	double *data = (double *)mstruct.vals;
	double *output;

	ATL_TIME_REPEAT(output = (double *) calloc(mdim0, sizeof(double))
			, SpMDCSC(vvals,data,pos0,pos1,crd0,crd1,mdim0,output)
			, free(output)
			, myValidate(out,output,vdim0)
			, repeat, timevalue, true)
	  cout << "ATL DCSC\n" << timevalue << endl;
      } else if (m.getFormat() == COO(2)) {
	    TensorStorage mstor = m.getStorage();
	    struct taco_tensor_t mstruct = *mstor;
	    
	    int mdim0 = mstruct.dimensions[0];
	    int mdim1 = mstruct.dimensions[1];
	    
	    int *pos = (int *)mstruct.indices[0][0];
	    int *crd0 = (int *)mstruct.indices[0][1];
	    int *crd1 = (int *)mstruct.indices[1][1];
			
	    double *data = (double *)mstruct.vals;
	    double *output;

	    ATL_TIME_REPEAT(output = (double *) calloc(mdim0, sizeof(double))
			    , SpMCOO(vvals,data,crd0,crd1,pos,mdim0,output)
			    , free(output)
			    , myValidate(out,output,vdim0)
			    , repeat, timevalue, true)
	      cout << "ATL COO\n" << timevalue << endl;
      } else /* Gotta be BCSR */ {
	// let's try BCSR
	// Pad to the next multiple of blockSize1 and blockSize2 for the respective dimension
	int blockSize1 = 2;
	int blockSize2 = 4;
	TensorStorage mstor = m.getStorage();
	struct taco_tensor_t mstruct = *mstor;
	
	int mdim0 = mstruct.dimensions[0];
	int mdim1 = mstruct.dimensions[1];
	int mdim2 = mstruct.dimensions[2];
	int mdim3 = mstruct.dimensions[3];
	int rows = mdim0*mdim2;
	int cols = mdim1*mdim3;
	cout << mdim0 << "*" << mdim1 << "*" << mdim2 << "*" << mdim3 << endl;
	
	int *pos = (int *)mstruct.indices[1][0];
	int *crd = (int *)mstruct.indices[1][1];
	double *data = (double *)mstruct.vals;
	double *output;
	
	ATL_TIME_REPEAT(output = (double *) calloc(rows, sizeof(double))
			,SpMBCSR(data,vvals,crd,pos,cols,rows,
				 blockSize1,blockSize2,output)
			, free(output)
			, myValidate(out,output,rows)
			, repeat, timevalue, true)
	  cout << "ATL BCSR\n" << timevalue << endl;
      }
    }
  } else if (Expr==SpMSpV) {
    Tensor<double> A = exprOperands.at("A")[0];
    TensorStorage Astor = A.getStorage();
    struct taco_tensor_t Astruct = *Astor;
    int A_dimension = (int)(Astruct.dimensions[0]);
    int * A_pos = (int*)(Astruct.indices[1][0]);
    int * A_crd = (int*)(Astruct.indices[1][1]);
    double* A_vals = (double*)(Astruct.vals);
  
    for (auto sparsity : Sparsities) {
      auto x = exprOperands["x" + std::to_string(sparsity)][0];
      TensorStorage xstor = x.getStorage();
      struct taco_tensor_t xstruct = *xstor;
      int * x_pos = (int*)(xstruct.indices[0][0]);
      int * x_crd = (int*)(xstruct.indices[0][1]);
      double * x_vals = (double*)(xstruct.vals);
      
      auto yref = exprOperands["y" + std::to_string(sparsity)][0];

      double *output;
      ATL_TIME_REPEAT(output = (double *) calloc(A_dimension, sizeof(double))
		      , SpMSpVCSRCSF(A_vals,x_vals,A_crd,x_crd,A_pos,x_pos,A_dimension,output)
		    , free(output)
		    , myValidate(yref,output,A_dimension)
		    , repeat, timevalue, true)
	cout << "ATL SpMSpV " << sparsity << "\n" << timevalue << endl;
    
      cout << sparsity << endl;
    }
    
  } else if (Expr==SparsityTTV) {
    // ARef(i,j) = B(i,j,k) * x(k)
    Tensor<double> ARef = exprOperands.at("ARef")[0];
    
    Tensor<double> x = exprOperands.at("x")[0];
    TensorStorage xstor = x.getStorage();
    struct taco_tensor_t xstruct = *xstor;
    double *xvals = (double *)xstruct.vals;
    int xdim0 = xstruct.dimensions[0];
    
    // Assuming we are only testing one format and it is sparse,sparse,sparse
    Tensor<double> B = exprOperands.at("B")[0];

    TensorStorage Bstor = B.getStorage();
    struct taco_tensor_t Bstruct = *Bstor;
    int Bdim0 = Bstruct.dimensions[0];
    int Bdim1 = Bstruct.dimensions[1];
    int Bdim2 = Bstruct.dimensions[2];

    int *pos0 = (int *)Bstruct.indices[0][0];
    int *crd0 = (int *)Bstruct.indices[0][1];
    int *pos1 = (int *)Bstruct.indices[1][0];
    int *crd1 = (int *)Bstruct.indices[1][1];
    int *pos2 = (int *)Bstruct.indices[2][0];
    int *crd2 = (int *)Bstruct.indices[2][1];
    double *data = (double *)Bstruct.vals;
    
    double *output;

    
    ATL_TIME_REPEAT(output = (double *) calloc(Bdim0*Bdim1, sizeof(double))
		    , TTV(data,xvals,pos0,pos1,
			  pos2,crd2,crd1,crd0,Bdim1,Bdim0,output)
		    , free(output)
		    , myValidate(ARef,output,Bdim0*Bdim1)
		    , repeat, timevalue, true)
    cout << "ATL NELL\n" << timevalue << endl;
    
    for (auto sparsity : Sparsities) {
      Tensor<double> A = exprOperands.at("A" + std::to_string(sparsity))[0];
      Tensor<double> B = exprOperands.at("B" + std::to_string(sparsity))[0];
      Tensor<double> x = exprOperands.at("x_sparsity")[0];

      TensorStorage xstor = x.getStorage();
      struct taco_tensor_t xstruct = *xstor;
      double *xvals = (double *)xstruct.vals;
      int xdim0 = xstruct.dimensions[0];
      
      TensorStorage Bstor = B.getStorage();
      struct taco_tensor_t Bstruct = *Bstor;
      
      int Bdim0 = Bstruct.dimensions[0];
      int Bdim1 = Bstruct.dimensions[1];
      int Bdim2 = Bstruct.dimensions[2];
      int *pos0 = (int *)Bstruct.indices[0][0];
      int *crd0 = (int *)Bstruct.indices[0][1];
      int *pos1 = (int *)Bstruct.indices[1][0];
      int *crd1 = (int *)Bstruct.indices[1][1];
      int *pos2 = (int *)Bstruct.indices[2][0];
      int *crd2 = (int *)Bstruct.indices[2][1];
      double *data = (double *)Bstruct.vals;    
      double *output;
      
      
      ATL_TIME_REPEAT(output = (double *) calloc(Bdim0*Bdim1, sizeof(double))
		      , TTV(data,xvals,pos0,pos1,
			    pos2,crd2,crd1,crd0,Bdim1,Bdim0,output)
		      , free(output)
		      , myValidate(A,output,Bdim0*Bdim1)
		      , repeat, timevalue, true)
	cout << "ATL " << sparsity << ":\n" << timevalue << endl;
    }
    
  } else if (Expr==SpTTV) {
    // ARef(i,j) = B(i,j,k) * x(k)
    Tensor<double> B = exprOperands.at("B")[0];
    cout << "spspttv" << endl;
    for (auto sparsity:Sparsities) {
      Tensor<double> A = exprOperands.at("A" + std::to_string(sparsity))[0];
      Tensor<double> x = exprOperands.at("x" + std::to_string(sparsity))[0];
      cout << "operands retrieved" << endl;
      
      TensorStorage Bstor = B.getStorage();
      struct taco_tensor_t Bstruct = *Bstor;
      int dim1 = Bstruct.dimensions[0];
      int dim2 = Bstruct.dimensions[1];
      int dim3 = Bstruct.dimensions[2];
      int *pos0 = (int *)Bstruct.indices[0][0];
      int *crd0 = (int *)Bstruct.indices[0][1];
      int *pos1 = (int *)Bstruct.indices[1][0];
      int *crd1 = (int *)Bstruct.indices[1][1];
      int *pos2 = (int *)Bstruct.indices[2][0];
      int *crd2 = (int *)Bstruct.indices[2][1];
      double *data = (double *)Bstruct.vals;

      TensorStorage xstor = x.getStorage();
      struct taco_tensor_t xstruct = *xstor;
      int * x_pos = (int*)(xstruct.indices[0][0]);
      int * x_crd = (int*)(xstruct.indices[0][1]);
      double * x_vals = (double*)(xstruct.vals);

      double *output = NULL;
      ATL_TIME_REPEAT(output = (double *) calloc(dim1*dim2, sizeof(double))
		      , SpSpTTV(data,x_vals,pos0,pos1,
				pos2,x_pos,crd0,crd1,crd2,x_crd,
				dim1,dim2,output)
		      , free(output)
		      , myValidate(A,output,dim1*dim2)
		      , repeat, timevalue, true)
	cout << "ATL " << sparsity << ":\n" << timevalue << endl;

      for (auto sparsity : Sparsities) {
	/* process various sparsities of B */
      }
      
    }
    
  } else if (Expr==SpMTTKRP) {
    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j)
    Tensor<double> ARef = exprOperands.at("ARef")[0];
    Tensor<double> B = exprOperands.at("B")[0];
    Tensor<double> C = exprOperands.at("C")[0];
    Tensor<double> D = exprOperands.at("D")[0];
    
    TensorStorage cstor = C.getStorage();
    struct taco_tensor_t cstruct = *cstor;
    double *cvals = (double *)cstruct.vals;
    TensorStorage dstor = D.getStorage();
    struct taco_tensor_t dstruct = *dstor;
    double *dvals = (double *)dstruct.vals;
    
    TensorStorage Bstor = B.getStorage();
    struct taco_tensor_t Bstruct = *Bstor;
    
    int dim0 = Bstruct.dimensions[0];
    int dim1 = Bstruct.dimensions[1];
    int dim2 = Bstruct.dimensions[2];
    int dim3 = dstruct.dimensions[1];

    int *pos0 = (int *)Bstruct.indices[0][0];
    int *crd0 = (int *)Bstruct.indices[0][1];
    int *pos1 = (int *)Bstruct.indices[1][0];
    int *crd1 = (int *)Bstruct.indices[1][1];
    int *pos2 = (int *)Bstruct.indices[2][0];
    int *crd2 = (int *)Bstruct.indices[2][1];
    double *data = (double *)Bstruct.vals;
    
    double *output;
    ATL_TIME_REPEAT(output = (double *) calloc(dim0*dim3, sizeof(double))
		    , MTTKRP(data,dvals,cvals,pos1,pos2,crd1,crd2,
			     crd0,pos0,dim3,output)
		    , free(output)
		    , myValidate(ARef,output,dim0*dim3)
		    , repeat, timevalue, true)
      cout << "ATL MTTKRP\n" << timevalue << endl;
  } else {
    cout << "No ATL library linked for this expression!" << endl;
  }
}

