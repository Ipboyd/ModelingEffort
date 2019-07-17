#include "mex.h"
#include "math.h"

#ifndef max
#define max(a,b)   (((a) > (b)) ? (a) : (b))
#endif
#ifndef min
#define min(a,b)   (((a) < (b)) ? (a) : (b))
#endif

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
    double *x, *out, pole;
    mwSize tr, tc;
    int i, j;
    mxArray *lhs[1];
    
    /* Check for proper number of arguments. */
    if(nrhs<3) {
        mexErrMsgTxt("Three inputs required.");
    } else if(nlhs>1) {
        mexErrMsgTxt("Too many output arguments");
    }
    
    /* The inputs must be noncomplex double row vectors. */
    tr = mxGetM(prhs[0]);
    tc = mxGetN(prhs[0]);
    if( nrhs < 2 || !mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) || !mxIsScalar(prhs[1]) ||
    !mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]) || !mxIsScalar(prhs[2])) {
        mexErrMsgTxt("Incorrect inputs!\nN.B. All three inputs must be doubles, try passing double(var)");
    }
    
    /* Set max lengths */
    x = mxGetPr(prhs[0]);
    pole = exp(-1/(mxGetScalar(prhs[2])*mxGetScalar(prhs[1])));
    
    /* Special case for row vector */
    if(tr == 1) {
        plhs[0] = mxCreateDoubleMatrix(1, tc, mxREAL);
        tr = tc;
        tc = 1;
    }
    else {
        plhs[0] = mxCreateDoubleMatrix(tr, tc, mxREAL);
    }
    
    if(tr > 0) {
        out = mxGetPr(plhs[0]);

        for(i=0; i<tc; i++) {
            out[0] = x[0];
            for(j=1; j<tr; j++)
                out[j] = x[j] + pole*out[j-1];
            out += tr;
            x += tr;
        }
    }
}
