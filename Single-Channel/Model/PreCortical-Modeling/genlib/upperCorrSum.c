#include "mex.h"
#include "math.h"

#ifndef max
#define max(a,b)   (((a) > (b)) ? (a) : (b))
#endif
#ifndef min
#define min(a,b)   (((a) < (b)) ? (a) : (b))
#endif

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int tr,tc;
    double *x, *tempx, *tempy, *matNorms, *matPr1, *matPr2;
    double r, rtemp0, rtemp1, minNorm;
    int i,j,k;
    mxArray *tempMat[1];
    
    /* Check for proper number of arguments. */
    if(nrhs != 1) {
        mexErrMsgTxt("Exactly one input required");
    }
    
    /* The inputs must be noncomplex double row vectors. */
    tr = mxGetM(prhs[0]);
    tc = mxGetN(prhs[0]);
    if( mxIsComplex(prhs[0]) || !mxIsDouble(prhs[0]) || tr <= 1 || tc <= 1) {
        mexErrMsgTxt("Incorrect input");
    }
    
    /* Prepare variables*/
    x = mxGetPr(prhs[0]);
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    
    matNorms = (double*)mxMalloc(tc*sizeof(double));
    tempx = x;
    minNorm = min(0.1/tr,0.00001);
    for(i=0;i<tc;i++) {
        matNorms[i] = 0;
        for(k=0;k<tr;k++) {
            matNorms[i] += tempx[k]*tempx[k];
        }
        matNorms[i] = 1/max(sqrt(matNorms[i]),minNorm);
        tempx += tr;
    }
    tempMat[1] = mxCreateDoubleMatrix(tc, tc, mxREAL);
    if(nlhs>=2)
        plhs[1] = tempMat[1];
    matPr1 = mxGetPr(tempMat[1]);
   
    tempx = x;
    tempy = x;
    r = 0;
    /* For each column */
    for(i=0;i<tc;i++) {
        tempy = tempx+tr;
        matPr1 += i;
        matPr2 = matPr1 + tc;
        *matPr1++ = 1;
        rtemp0 = 0;
        /* For each column after it */
        for(j=i+1;j<tc;j++) {
            /* Add up individual points */
            rtemp1 = 0;
            for(k=0;k<tr;k++) {
                rtemp1 += (*tempx++)*(*tempy++);
            }
            tempx -= tr;
            rtemp0 += rtemp1*matNorms[j]; /* Normalize contribution by column j */
            *matPr2 = *matPr1++ = rtemp1*matNorms[j]*matNorms[i];
            matPr2 += tc;
        }
        r += rtemp0*matNorms[i]; /* Normalize contribution by column i */
        tempx += tr;
    }
    *(mxGetPr(plhs[0])) = r * 2 / (tc*(tc-1));
    mxFree(matNorms);
}

/*
// matNorms = max(sqrt(sum(convMat.^2)),1/len);
// r = 0;
// for ii = 1:Ntrials
//     rtemp = 0;
//     for jj = ii+1:Ntrials
//         rtemp2 = 0;
//         for kk = 1:len
//             rtemp2 = rtemp2 + convMat(kk,ii)*convMat(kk,jj);
//         end
//         rtemp = rtemp + rtemp2 / matNorms(jj);
//     end
//     r = r + rtemp/matNorms(ii);
// end
// r = r*2 / (Ntrials*(Ntrials-1));
*/
