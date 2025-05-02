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
    double *x, *counts, a, b;
    mwSize tr, tc;
    int i, j, rmp1;
    mxArray *lhs[1];
    
    rmp1 = RAND_MAX+1;
    
    /* Check for proper number of arguments. */
    if(nrhs<1) {
        mexErrMsgTxt("Three inputs required.");
    } else if(nlhs>1) {
        mexErrMsgTxt("Too many output arguments");
    }
    
     /* The inputs must be noncomplex double row vectors. */
    tr = mxGetM(prhs[0]);
    tc = mxGetN(prhs[0]);
    if( !mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) || tr*tc < 1) {
        mexErrMsgTxt("Incorrect inputs!");
    }
    x = mxGetPr(prhs[0]);
    plhs[0] = mxCreateDoubleMatrix(tr, tc, mxREAL);
    counts = mxGetPr(plhs[0]);
    memset(counts,0,sizeof(counts[0])*tr*tc);
    
    for(i=0; i<tc; i++) {
        for(j=0; j<tr; j++) {
            a = exp(-x[j]);
            b = rand()/((double)rmp1);
            while(b>a) {
                counts[j]++;
                b *= rand()/((double)rmp1);
            }
        }
        counts += tr;
        x += tr;
    }
}
