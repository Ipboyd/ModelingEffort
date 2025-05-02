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
    double *x, *out;
    double f, fs, checktype, totype;
    mwSize tr, tc, Ns;
    int ds, i, j, ind, fds;
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
    if( !mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) || !mxIsScalar(prhs[1]) ||
    !mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]) || !mxIsScalar(prhs[2]) ||
    !(tr==1 || tc==1) ) {
        mexErrMsgTxt("Incorrect inputs!\nN.B. All inputs must be doubles, try passing double(var)");
    }
    
    /* Set max lengths */
    x = mxGetPr(prhs[0]);
    f = mxGetScalar(prhs[1]);
    if(tc>tr) {
        Ns = tc;
        plhs[0] = mxCreateDoubleMatrix(1, Ns, mxREAL);
    }
    else {
        Ns = tr;
        plhs[0] = mxCreateDoubleMatrix(Ns, 1, mxREAL);
    }
    out = mxGetPr(plhs[0]);
    memcpy(out, x, sizeof(double)*Ns);
    
    if(!f) {
        return;
    }
    else if(f > 0.0) {
        checktype = 0;
        totype = 1;
    }
    else {
        checktype = 1;
        totype = 0;
    }
    
    fs = mxGetScalar(prhs[2]);
    fds = (int)( ((double)RAND_MAX + 1)*abs(f)/fs);
    ds = ceil(fs/abs(f));
    
    for(i=0; i<Ns; i++) {
        /* Randomly add or delete a spike as close to position i as possible */
        if(rand() < fds) {
            j = 0;
            while(j<ds) {
                ind = i+j;
                if(ind < Ns && x[ind] == checktype) {
                    out[ind] = totype;
                    break;
                }
                ind = i-j;
                if(ind >=0 && x[ind] == checktype) {
                    out[ind] = totype;
                    break;
                }
                j++;
            }
        }
    }
}
