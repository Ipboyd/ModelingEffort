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
    double *x;
    double sigma;
    double *out;
    int i,j,pad,templen;
    double q,qq,qqq,b0,b1,b2,b3,B,Bsum,temp;
    double *tempd0,*tempd1,*tempd2;
    double bc;
    
    /* Check for proper number of arguments. */
    if(nrhs < 2 || nrhs > 3) {
        mexErrMsgTxt("Two or three inputs required:\n      input matrix (double)\n      sigma (samples)\n      padding (samples) (optional, default = 3*sigma)");
    } else if(nlhs>1) {
        mexErrMsgTxt("Too many output arguments");
    }
    
    /* The inputs must be noncomplex double row vectors. */
    tr = mxGetM(prhs[0]);
    tc = mxGetN(prhs[0]);
    if( !mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) || !mxIsScalar(prhs[1])
     || (tr < 4 && tc < 4) || (nrhs >=3 && (mxIsComplex(prhs[2]) || !mxIsScalar(prhs[2])))) {
        mexErrMsgTxt("Two or three inputs required:\n      input matrix (double)\n      sigma (samples)\n      padding (samples) (optional, default = 4*sigma)");
    }
    
    /* Prepare variables*/
    x = mxGetPr(prhs[0]);
    sigma = mxGetScalar(prhs[1]);
    if(nrhs >= 3) {
        pad = max(mxGetScalar(prhs[2]),0);
    }
    else {
        pad = 4*sigma;
    }
    
    q = 1.31564 * (sqrt(1 + 0.490811 * sigma*sigma) - 1);
    qq = q*q;
    qqq = qq*q;
    b0 = 1.0/(1.57825 + 2.44413*q + 1.4281*qq + 0.422205*qqq);
    b1 = (2.44413*q + 2.85619*qq + 1.26661*qqq)*b0;
    b2 = (-1.4281*qq - 1.26661*qqq)*b0;
    b3 = 0.422205*qqq*b0;
    Bsum = b1 + b2 + b3;
    B = 1.0 - Bsum;
    
    /* Special case for row vector; ensure that tr = the number of elements */
    if(tr == 1) {
        plhs[0] = mxCreateDoubleMatrix(1, tc, mxREAL);
        tr = tc;
        tc = 1;
    }
    else {
        plhs[0] = mxCreateDoubleMatrix(tr, tc, mxREAL);
    }
    
    out = mxGetPr(plhs[0]);
    templen = tr + 2*pad;
    
    tempd0 = (double*)mxCalloc(templen,sizeof(double));
    tempd1 = (double*)mxCalloc(templen,sizeof(double));
    tempd2 = (double*)mxCalloc(templen,sizeof(double));
    
    for(i=0; i<tc; i++) {
        memcpy(tempd0+pad, x, sizeof(double)*tr);

        /* Convolve one direction...? */
        temp = Bsum*tempd0[0];
        for (j=0;j<3;j++) {
            tempd1[j] = B*tempd0[j] + temp;
        }
        for (j=3;j<templen;j++) {
            tempd1[j] = B*tempd0[j] + b1*tempd1[j-1] + b2*tempd1[j-2] + b3*tempd1[j-3];
        }

        /* Convolve the other...? */
        temp = Bsum*tempd1[templen-1];
        for (j=1;j<4;j++) {
            tempd2[templen-j] = B*tempd1[templen-j] + temp;
        }
        for (j=templen-4;j>=0;j--) {
            tempd2[j] = B*tempd1[j] + b1*tempd2[j+1] + b2*tempd2[j+2] + b3*tempd2[j+3];
        }

        memcpy(out, tempd2+pad, sizeof(double)*tr);
        out += tr;
        x += tr;
    }
    mxFree(tempd0);
    mxFree(tempd1);
    mxFree(tempd2);
}
