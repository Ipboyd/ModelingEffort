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
    double *x, *depVec, *depExp;
    double mag, mtauXfs, pole, precision, t0dep, t0frac;
    mwSize tr, tc, klim, explen, llim;
    int i, j, k, l;
    mxArray *lhs[1];
    
    /* Check for proper number of arguments. */
    if(nrhs < 3) {
        mexErrMsgTxt("Three inputs required.");
    } else if(nlhs > 1) {
        mexErrMsgTxt("Too many output arguments");
    }

    /* The inputs must be noncomplex double row vectors. */
    if( nrhs < 4 || nrhs > 5 || !mxIsDouble(prhs[0]) ||
    !mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) || !mxIsScalar(prhs[1]) ||
    !mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]) || !mxIsScalar(prhs[2]) ||
    !mxIsDouble(prhs[3]) || mxIsComplex(prhs[3]) || !mxIsScalar(prhs[3]) ||
    (nrhs == 5 && (!mxIsDouble(prhs[4]) || mxIsComplex(prhs[4]) || !mxIsScalar(prhs[4])))
    ) {
        mexErrMsgTxt("Incorrect inputs!\nN.B. All inputs must be doubles, try passing double(var)");
    }
    if(nrhs == 5) {
        precision = mxGetScalar(prhs[4]);
        if(precision <=0)
            mexErrMsgTxt("Precision must be > 0");
    }
    else
        precision = 0.01;

    /* Set max lengths */
    tr = mxGetM(prhs[0]);
    tc = mxGetN(prhs[0]);
    x = mxGetPr(prhs[0]);
    mtauXfs = -mxGetScalar(prhs[2])*mxGetScalar(prhs[1]);
    pole = exp(1/mtauXfs);
    mag = mxGetScalar(prhs[3]) - 1;

    /* Special case for row vector */
    if(tr == 1) {
        plhs[0] = mxCreateDoubleMatrix(1, tc, mxREAL);
        tr = tc;
        tc = 1;
    }
    else {
        plhs[0] = mxCreateDoubleMatrix(tr, tc, mxREAL);
    }

    depVec = mxGetPr(plhs[0]);
    explen = max(1,min(ceil(mtauXfs*log(precision)),tr)); /* Decay to 1% */
    depExp = (double*)mxCalloc(explen,sizeof(double));

    /* Calculate offsetted exponential kernel using IIR filter */
    depExp[0] = 1+mag;
    for(k=1;k<explen;k++)
        depExp[k] = 1 + pole*(depExp[k-1]-1);
    
    for(i=0; i<tc; i++) {
        for(j=0; j<tr; j++) {
            depVec[j] = 1;
        }
        for(j=0; j<tr; j++) {
            if(x[0]) {
                klim = min(tr-j,explen+1);
                llim = (mwSize)x[0];
                t0dep = 0;
                t0frac = 1;
                for(l=0;l<llim;l++) {
                    t0dep += t0frac;
                    t0frac *= -mag;
                    for(k=1;k<klim;k++)
                        depVec[k] *= depExp[k-1];
                }
                depVec[0] *= t0dep/(double)llim;
            }
            depVec++;
            x++;
        }
    }
    
    mxFree(depExp);
}
