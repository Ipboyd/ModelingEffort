#include "mex.h"
#include "math.h"

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
	double *distMat, *correct, numTargets, count;
	unsigned int n, i, j;
	
	/* Input pointers */
	distMat = mxGetPr(prhs[0]);
	numTargets = mxGetN(prhs[0]);
	n = mxGetM(prhs[0]);
	
	/* Output pointers */
	correct = mxGetPr(plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL));
	
	/* Code */
	for(i=0; i<n; i++) {
        count = 1;
        *correct = 1;
        for(j=1; j<numTargets; j++) {
            if(*distMat <= distMat[j*n]) {
                if(*distMat == distMat[j*n])
                    count++;
            }
            else {
                *correct = 0;
                count=0;
                j=numTargets;
            }
        }
        if(count>1) {
            *correct = 1/count;
        }
        correct++;
        distMat++;
	}
}