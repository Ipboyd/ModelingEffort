#include "mex.h"
#include "math.h"

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
    double *x, *y, *in2, in1;
	bool *out;
    mwSize tr, tc, N1, N2;
    int i, j;
    mxArray *lhs[1];
    
    /* Check for proper number of arguments. */
    if(nrhs != 2) {
        mexErrMsgTxt("Exactly two inputs required.");
    } else if(nlhs>1) {
        mexErrMsgTxt("Too many output arguments");
    }
    
    /* The inputs must be noncomplex double row vectors. */
    if(!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) || !mxIsDouble(prhs[1]) || mxIsComplex(prhs[1])) {
        mexErrMsgTxt("Incorrect inputs!");
    }
    
    /* Set max lengths */
    x = mxGetPr(prhs[0]);
    y = mxGetPr(prhs[1]);
    N1 = mxGetM(prhs[0])*mxGetN(prhs[0]);
    N2 = mxGetM(prhs[1])*mxGetN(prhs[1]);
    
    /* Special case for row vector */
    plhs[0] = mxCreateLogicalMatrix(N1, 1);
    out = mxGetLogicals(plhs[0]);

    for(i=0; i<N1; i++) {
		in1 = *x++;
		in2 = y;
        for(j=0; j<N2; j++) {
            if(in1 == *in2++) {
                out[i] = true;
				j = N2;
			}
        }
    }
}
