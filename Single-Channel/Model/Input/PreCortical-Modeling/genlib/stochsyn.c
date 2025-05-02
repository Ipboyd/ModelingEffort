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
    double *x, *out, *pvec, pole, p;
    mwSize tr, tc;
    int i, j, k;
    mxArray *lhs[1], *tlhs[1], *trhs[1];
    
    /* Check for proper number of arguments. */
    if(nrhs<3) {
        mexErrMsgTxt("Three inputs required.");
    } else if(nlhs>1) {
        mexErrMsgTxt("Too many output arguments");
    }
    tr = mxGetM(prhs[0]);
    tc = mxGetN(prhs[0]);

    /* The inputs must be noncomplex double row vectors. */
    if( nrhs < 4 ||
    !mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) ||
    !mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) || !mxIsScalar(prhs[1]) ||
    !mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]) || !mxIsScalar(prhs[2]) ||
    !mxIsDouble(prhs[3]) || mxIsComplex(prhs[3]) || !mxIsScalar(prhs[3])
    ) {
        mexErrMsgTxt("Incorrect inputs!");
    }

    /* Special case for row vector */
    if(tr == 1) {
        plhs[0] = mxCreateDoubleMatrix(1, tc, mxREAL);
        tr = tc;
        tc = 1;
    }
    else {
        plhs[0] = mxCreateDoubleMatrix(tr, tc, mxREAL);
    }

    /* Get some random numbers
    trhs[0] = mxCreateDoubleMatrix(1, 2, mxREAL);
    noise = mxGetPr(trhs[0]);
    noise[0] = 1;
    noise[1] = origlen;
    mexCallMATLAB(1, tlhs, 1, trhs, "rand");
    mxDestroyArray(trhs[0]);
    noise = mxGetPr(tlhs[0]);

    /* Initialize vars */
    x = mxGetPr(prhs[0]);
    pole = exp(-1/(mxGetScalar(prhs[1])*mxGetScalar(prhs[2])));
    p = RAND_MAX*mxGetScalar(prhs[3]);
    out = mxGetPr(plhs[0]);
    pvec = (double*)mxCalloc(tr,sizeof(double));
    
    for(i=0; i<tc; i++) {
        /* Find first spike, keep it */
        j = 0;
        while(x[j] < 1 && j < tr)
            j++;
        if(j<tr) {
            pvec[j] = p;
            out[j] = 1;
            for(k=1;k<x[j];k++)
                if(pvec[j] <= rand()) {
                    pvec[j] += p;
                    out[j]++;
                }
            j++;

            for(;j<tr;j++) {
                pvec[j] = pole*pvec[j-1];
                for(k=0;k<x[j];k++)
                    if(pvec[j] <= rand()) {
                        pvec[j] += p;
                        out[j]++;
                    }
            }
        }
        out += tr;
        x += tr;
    }
    mxFree(pvec);
}
