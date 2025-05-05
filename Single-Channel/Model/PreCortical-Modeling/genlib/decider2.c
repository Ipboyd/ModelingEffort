#include "mex.h"
#include "math.h"

#ifndef max
#define max(a,b)   (((a) > (b)) ? (a) : (b))
#endif
#ifndef min
#define min(a,b)   (((a) < (b)) ? (a) : (b))
#endif

bool checkArgs( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
    double **Iin, **r, *S, **noise, *Inoise;
    double dt, taus, tauampa, noisesd, sdeff, dtOtauampa, gamma, I0;
    double JN11, JN12, a, b, d, tempsum, x;
    int ni, oi, ti, tim1;
    mwSize Nn, Ns;
    mxArray *tlhs[1], *trhs[1];

    Ns = mxGetM(prhs[0]);
    Nn = mxGetN(prhs[0]);

    if(checkArgs(nlhs,plhs,nrhs,prhs))
        return;
    
    dt = mxGetScalar(prhs[1]);
    taus = mxGetScalar(prhs[2]);
    tauampa = mxGetScalar(prhs[3]);
    gamma = mxGetScalar(prhs[4]);
    noisesd = mxGetScalar(prhs[5]);
    I0 = mxGetScalar(prhs[6]);
    JN11 = *mxGetPr(prhs[7]);
    JN12 = *(mxGetPr(prhs[7]) + 1);
    S = mxGetPr(prhs[8]); /* Temporarily use S */
    a = S[0];
    b = S[1];
    d = S[2];

    sdeff = noisesd*sqrt(tauampa/dt);
    dtOtauampa = dt/tauampa;

    /* Get some AWGN */
    noise = (double**)mxCalloc(Nn,sizeof(double*));
    trhs[0] = mxCreateDoubleMatrix(1, 2, mxREAL);
    noise[0] = mxGetPr(trhs[0]);
    noise[0][0] = Ns;
    noise[0][1] = Nn;
    mexCallMATLAB(1, tlhs, 1, trhs, "randn");
    mxDestroyArray(trhs[0]);
    noise[0] = mxGetPr(tlhs[0]);

    /* Deal with double pointers and mallocs */
    Iin = (double**)mxCalloc(Nn,sizeof(double*));
    Iin[0] = mxGetPr(prhs[0]);
    r = (double**)mxCalloc(Nn,sizeof(double*));
    plhs[0] = mxCreateDoubleMatrix(Ns, Nn, mxREAL);
    r[0] = mxGetPr(plhs[0]);
    for(ni=1;ni<Nn; ni++) {
        tim1 = ni-1;
        Iin[ni] = Iin[tim1] + Ns;
        r[ni] = r[tim1] + Ns;
        noise[ni] = noise[tim1] + Ns;
    }
    S = (double*)mxCalloc(Nn,sizeof(double));
    Inoise = (double*)mxCalloc(Nn,sizeof(double));
    
    /* Begin integration: */
    for(ti=1; ti<Ns; ti++) {
        tim1 = ti-1;
        tempsum = 0;
        for(ni=0; ni<Nn; ni++)
            tempsum += S[ni];

        for(ni=0; ni<Nn; ni++) {
            Inoise[ni] -= dtOtauampa*(Inoise[ni] - noise[ni][ti]*sdeff);
            x = JN11*S[ni] - JN12*(tempsum-S[ni]) + I0 + Iin[ni][ti] + Inoise[ni];
            r[ni][tim1] = (a*x-b)/(1-exp(d*(b-a*x)));
            S[ni] -= dt*( S[ni]/taus - (1-S[ni])*gamma*r[ni][tim1] );
        }
    }
    for(ni=0; ni<Nn; ni++) {
        r[ni][Ns] = r[ni][Ns-1];
    }
    
    mxFree(r[0]);
    mxFree(r);
    mxFree(S);
    mxFree(Inoise);
    mxFree(noise);
}

bool checkArgs( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    int tempint;
    mwSize Ns, Nn;

    /* Check for proper number of arguments. */
    if(nrhs < 9 || nlhs > 1) {
        mexErrMsgTxt("9 inputs and at most 1 output required.");
    }

    /* The inputs must be noncomplex double row vectors. */
    Ns = mxGetM(prhs[0]);
    Nn = mxGetN(prhs[0]);

    /* 01: Iin (Ns x Nn) */
    if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]))
        mexErrMsgTxt("Error 01: Iin must be a real double Ns x Nn matrix");
    /* 02: dt (1) */
    tempint = mxGetM(prhs[1])*mxGetN(prhs[1]);
    if (!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) || 1 != tempint)
        mexErrMsgTxt("Error 02: dt must be a real double scalar");
    if(mxGetScalar(prhs[1]) <= 0) mexErrMsgTxt("Error 02: dt <= 0");
    /* 03: Taus (1) */
    tempint = mxGetM(prhs[2])*mxGetN(prhs[2]);
    if (!mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]) || 1 != tempint)
        mexErrMsgTxt("Error 03: taus must be a real double scalar");
    if(mxGetScalar(prhs[2]) <= 0) mexErrMsgTxt("Error 03: taus <= 0");
    /* 04: Tauampa (1) */
    tempint = mxGetM(prhs[3])*mxGetN(prhs[3]);
    if (!mxIsDouble(prhs[3]) || mxIsComplex(prhs[3]) || 1 != tempint)
        mexErrMsgTxt("Error 04: tauampa must be a real double scalar");
    if(mxGetScalar(prhs[3]) <= 0) mexErrMsgTxt("Error 04: tauampa <= 0");
    /* 05: Gamma (1) */
    tempint = mxGetM(prhs[4])*mxGetN(prhs[4]);
    if (!mxIsDouble(prhs[4]) || mxIsComplex(prhs[4]) || 1 != tempint)
        mexErrMsgTxt("Error 05: gamma must be a real double scalar");
    if(mxGetScalar(prhs[4]) <= 0) mexErrMsgTxt("Error 05: gamma <= 0");
    /* 06: Noisesd (1) */
    tempint = mxGetM(prhs[5])*mxGetN(prhs[5]);
    if (!mxIsDouble(prhs[5]) || mxIsComplex(prhs[5]) || 1 != tempint)
        mexErrMsgTxt("Error 06: noisesd must be a real double scalar");
    if(mxGetScalar(prhs[5]) <= 0) mexErrMsgTxt("Error 06: noisesd <= 0");
    /* 07: I0 (1) */
    tempint = mxGetM(prhs[6])*mxGetN(prhs[6]);
    if (!mxIsDouble(prhs[6]) || mxIsComplex(prhs[6]) || 1 != tempint)
        mexErrMsgTxt("Error 07: I0 must be a real double scalar");
    /* 08: JN (Nn x Nn) */
    tempint = mxGetM(prhs[7])*mxGetN(prhs[7]);
    if (!mxIsDouble(prhs[7]) || mxIsComplex(prhs[7]) || ((Nn != mxGetM(prhs[7]) || Nn != mxGetN(prhs[7])) && tempint != 2))
        mexErrMsgTxt("Error 08: JN must be a real double 2-element vector");
    /* 09: ABD (3) */
    tempint = mxGetM(prhs[8])*mxGetN(prhs[8]);
    if (!mxIsDouble(prhs[8]) || mxIsComplex(prhs[8]) || tempint != 3)
        mexErrMsgTxt("Error 09: ABD must be a real double 3-element vector");
    
    return 0;
}
