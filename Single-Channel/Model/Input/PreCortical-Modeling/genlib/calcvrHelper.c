#include "mex.h"

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
	double *y, *w, *e, *z;
	unsigned int n;
	
	/* Input pointers */
	w = mxGetPr(prhs[0]);
	e = mxGetPr(prhs[1]);
	n = mxGetM(prhs[0])*mxGetN(prhs[0]);
	
	/* Output pointers */
	y = mxGetPr(plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL));
	
	/* Code */
	z = y + n - 1;
	while(y < z) {
		*++y = (*y + *w++)*(*e++); /* y[i + 1] = (y[i] + w[i])*e[i]; i++; */
	}
}