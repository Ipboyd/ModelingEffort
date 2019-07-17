#include "mex.h"
#include "math.h"

#if !defined(min)
#define	min(A, B)	((A) < (B) ? (A) : (B))
#endif

#if !defined(max)
#define	max(A, B)	((A) > (B) ? (A) : (B))
#endif

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
    double *x, *out, *noise;
    double jittersig;
    mwSize tr, tc, Ns, Nsm1;
    int i, j, ind, stind, ept;
    mxArray *tlhs[1], *trhs[1];
    
    /* Check for proper number of arguments. */
    if(nrhs<2) {
        mexErrMsgTxt("Two inputs required.");
    } else if(nlhs>1) {
        mexErrMsgTxt("Too many output arguments");
    }
    
    /* The inputs must be noncomplex double row vectors. */
    tr = mxGetM(prhs[0]);
    tc = mxGetN(prhs[0]);
    if( !mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) || !mxIsScalar(prhs[1]) ||
    !(tr==1 || tc==1) ) {
        mexErrMsgTxt("Incorrect inputs!\nN.B. All inputs must be doubles, try passing double(var)");
    }
    
    /* Set max lengths */
    if(tc>tr) {
        Ns = tc;
        plhs[0] = mxCreateDoubleMatrix(1, Ns, mxREAL);
    }
    else {
        Ns = tr;
        plhs[0] = mxCreateDoubleMatrix(Ns, 1, mxREAL);
    }
    Nsm1 = Ns-1;
    out = mxGetPr(plhs[0]);
    memset(out, 0, sizeof(double)*Ns);
    x = mxGetPr(prhs[0]);
    jittersig = mxGetScalar(prhs[1]);
    ept = (int)(5*jittersig);
    
    /* Get some WGN */
    trhs[0] = mxCreateDoubleMatrix(1, 2, mxREAL);
    noise = mxGetPr(trhs[0]);
    noise[0] = 1;
    noise[1] = Ns;
    mexCallMATLAB(1, tlhs, 1, trhs, "randn");
    mxDestroyArray(trhs[0]);
    noise = mxGetPr(tlhs[0]);

    for(i=0; i<Ns; i++) {
        /* Move a spike at/near position i */
        if(x[i] == 1) {
            /* Generate a normally-distributed random number */
            stind = max(min(i + ( jittersig * noise[i] ),Nsm1),0);
            j = 0;
            while(j<=ept) {
                ind = stind + j;
                if(ind < Ns && ind >= 0 && out[ind] == 0) {
                    out[ind] = 1;
                    break;
                }
                ind = stind - j;
                if(ind < Ns && ind >= 0 && out[ind] == 0) {
                    out[ind] = 1;
                    break;
                }
                j++;
            }
        }
    }
}
