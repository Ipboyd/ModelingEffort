#include "mex.h"

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
	double *t1, *t2, *wout, *tout;
	unsigned int n1, n2, nout, c1, c2, c;
	
	/* Input pointers */
	t1 = mxGetPr(prhs[0]);
	t2 = mxGetPr(prhs[1]);
	n1 = mxGetM(prhs[0])*mxGetN(prhs[0]);
	n2 = mxGetM(prhs[1])*mxGetN(prhs[1]);
    nout = n1+n2;
    c=c1=c2=0;
	
	/* Output pointers */
	tout = mxGetPr(plhs[0] = mxCreateDoubleMatrix(nout, 1, mxREAL));
	wout = mxGetPr(plhs[1] = mxCreateDoubleMatrix(nout, 1, mxREAL));
	
	/* Code */
    if(n1>0 && n2>0) {
        for (; c<nout; c++) {
            if(*t1 < *t2) {
                *tout++ = *t1++;
                c1++;
                *wout++ = 1;
                if(c1>=n1)
                    break;
            }
            else {
                *tout++ = *t2++;
                c2++;
                *wout++ = -1;
                if(c2>=n2)
                    break;
            }
        }
    }
    
    if(c1>=n1) {
        if(c2<n2) {
            for(c++; c<nout; c++) {
                *tout++ = *t2++;
                *wout++ = -1;
            }
        }
    }
    else if(c2>=n2) {
        if(c1<n1) {
            for(c++; c<nout; c++) {
                *tout++ = *t1++;
                *wout++ = 1;
            }
        }
    }
}