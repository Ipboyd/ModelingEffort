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
    double *cosPhase, *sinPhase, *out, a, b, normer;
    mwSize tr, tc;
    int r, i, j;
    mxArray *lhs[1];
    
    /* Check for proper number of arguments. */
    if(nrhs!=2) {
        mexErrMsgTxt("Two inputs required.");
    } else if(nlhs>1) {
        mexErrMsgTxt("Too many output arguments");
    }
    
    /* The inputs must be noncomplex double row vectors. */
    tr = mxGetM(prhs[0]);
    tc = mxGetN(prhs[0]);
    if( !mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) ||
        mxGetM(prhs[1])!=tr || mxGetN(prhs[1])!=tc ) {
        mexErrMsgTxt("Incorrect inputs!\nN.B. All three inputs must be doubles, try passing double(var)");
    }

    /* Set max lengths */
    cosPhase = mxGetPr(prhs[0]);
    sinPhase = mxGetPr(prhs[1]);
    plhs[0] = mxCreateDoubleMatrix(tc, 1, mxREAL);
    out = mxGetPr(plhs[0]);
	normer = tr*(tr-1)/2;
    for(r=0; r<tc; r++) {
        *out = 0;
        for(i=0; i<tr; i++) {
			a = cosPhase[i];
			b = sinPhase[i];
            for(j=i+1; j<tr; j++) {
                *out += a*cosPhase[j] + b*sinPhase[j];
            }
        }
		*out = *out/normer;
        out++;
		cosPhase+=tr;
		sinPhase+=tr;
    }
}
