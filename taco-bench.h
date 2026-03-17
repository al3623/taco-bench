
#include "taco.h"

using namespace taco;
using namespace std;

// MACRO to benchmark some CODE with REPEAT times and COLD/WARM cache
#define TACO_BENCH(CODE, NAME, REPEAT, TIMER, COLD) {               \
    TACO_TIME_REPEAT(CODE, REPEAT, TIMER, true);                    \
    cout << NAME << " time (ms)" << endl << TIMER << endl;  \
}

#define CHECK_PRODUCT(NAME) {                                   \
    if (products.at(NAME)) {                                    \
      cout << "taco-bench was not compiled with "<< NAME << " and will not use it" << endl; \
      products.at(NAME)=false;                                  \
    }                                                           \
}


// Enum of possible expressions to Benchmark
enum BenchExpr {SpMSpV, SpMV, PLUS3, MATTRANSMUL, RESIDUAL, SDDMM, SparsitySpMV, SparsityTTV, SparsitySpMDM, SpMTTKRP, SpTTV};

// Compare two tensors of different formats
bool compare(const Tensor<double>&Dst, const Tensor<double>&Ref) {
  if (Dst.getDimensions() != Ref.getDimensions()) {
    return false;
  }

  std::set<std::vector<int>> coords;
  for (const auto& val : Dst) {
    if (!coords.insert(val.first.toVector()).second) {
      return false;
    }
  }

  vector<std::pair<std::vector<int>,double>> valsDst;
  for (const auto& val : Dst) {
    if (val.second != 0) {
      valsDst.push_back(make_pair(val.first.toVector(),val.second));
    }
  }

  vector<std::pair<std::vector<int>,double>> valsRef;
  for (const auto& val : Ref) {
    if (val.second != 0) {
      valsRef.push_back(make_pair(val.first.toVector(),val.second));
    }
  }
  std::sort(valsRef.begin(), valsRef.end());
  std::sort(valsDst.begin(), valsDst.end());
  return valsDst == valsRef;
}

void validate (string name, const Tensor<double>& Dst, const Tensor<double>& Ref) {
  if (Dst.getFormat()==Ref.getFormat()) {
    if (!equals (Dst, Ref))
      cout << "\033[1;31m  Validation Error with " << name << " \033[0m" << endl;
  }
  else {
    if (!compare(Dst,Ref))
      cout << "\033[1;31m  Validation Error with " << name << " \033[0m" << endl;
  }
}

void myValidate(Tensor<double> t, double *t2, int n) {
  double *t1 = (double *) (*t.getStorage()).vals;
  bool success = true;
  int eq = 0;
  for (int i = 0; i < n; i++) {
    if (fabs(t1[i] -  t2[i]) / fabs(t1[i]) > 1e-4) {
      // fprintf(stderr,"%d: %lf <> %lf\n",i,t1[i],t2[i]);
      // fprintf(stderr,"%d ",i);
      success = false;
    } else {
      // fprintf(stderr,"%d: %lf == %lf\n",i,t1[i],t2[i]);
      eq++;
    }
  }	
  if (!success) {
    fprintf(stdout,"failed :(\n");
    fprintf(stdout,"eq: %d\n", eq);
  }
}

void clamp(Tensor<double> &A) {
  double *vals = (double *)A.getStorage().getValues().getData();
  size_t nnz = A.getStorage().getValues().getSize();
  double threshhold = 1e-6;
  for (size_t k = 0; k < nnz; k++) {
    if (fabs(vals[k]) < threshhold) {
      vals[k] = threshhold;
    }
  }
}      
