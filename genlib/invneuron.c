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
    double *Psi, *V, *rmI, *outcount, *noise, *logicMat, *indMat;
    double dt, dtOtaum, Ese, Esi, rmgse, rmgsi, Vap, Vth, Vre, noiselev, El, Vi, Vim1;
    int ti, tim1, tinext, count, minAPDist;
    bool outType;
    mwSize Psr, Psc, Ns;
    mxArray *tlhs[1], *trhs[1];
    
    /* Check for proper number of arguments. */
    if(nrhs<12) {
        mexErrMsgTxt("12 inputs required.");
    } else if(nlhs>2) {
        mexErrMsgTxt("Too many output arguments");
    }
    
    /* The inputs must be noncomplex double row vectors. */
    Psr = mxGetM(prhs[0]);
    Psc = mxGetN(prhs[0]);
    if(nrhs >= 13)
        outType = mxGetScalar(prhs[12]);
    else
        outType = 0;
    if( nrhs < 12 ||
    !mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]) ||
    !mxIsDouble(prhs[3]) || mxIsComplex(prhs[3]) ||
    !mxIsDouble(prhs[4]) || mxIsComplex(prhs[4]) ||
    !mxIsDouble(prhs[5]) || mxIsComplex(prhs[5]) ||
    !mxIsDouble(prhs[6]) || mxIsComplex(prhs[6]) ||
    !mxIsDouble(prhs[7]) || mxIsComplex(prhs[7]) ||
    !mxIsDouble(prhs[8]) || mxIsComplex(prhs[8]) ||
    !mxIsDouble(prhs[9]) || mxIsComplex(prhs[9]) ||
    !mxIsDouble(prhs[10])|| mxIsComplex(prhs[10])||
    !mxIsDouble(prhs[11])|| mxIsComplex(prhs[11])||
    !mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) || 
    !mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) ||
    !(Psr==1 || Psc==1)  ||
    Psr*Psc != mxGetM(prhs[1])*mxGetN(prhs[1])
    ) {
        mexErrMsgTxt("Incorrect inputs!");
    }

    /* Set max lengths */
    if(Psc>Psr) Ns = Psc; else Ns = Psr;
    
    /* Psi, rmI, arp, dt, Esi, rmgsi, taum, Vap, Vth, Vre, El (i.e. Vss), noiselev */
    Psi = mxGetPr(prhs[0]);
    rmI = mxGetPr(prhs[1]);
    dt = mxGetScalar(prhs[3]);
    if(dt <= 0)
        mexErrMsgTxt("Error: dt <= 0");
    Esi = mxGetScalar(prhs[4]);
    rmgsi = mxGetScalar(prhs[5]);
    dtOtaum = dt/mxGetScalar(prhs[6]);
    Vap = mxGetScalar(prhs[7]);
    Vre = mxGetScalar(prhs[8]);
    Vth = mxGetScalar(prhs[9]);
    El = mxGetScalar(prhs[10]);
    noiselev = mxGetScalar(prhs[11]);
    minAPDist = (int)(mxGetScalar(prhs[2])/dt);

    /* Get some AWGN */
    trhs[0] = mxCreateDoubleMatrix(1, 2, mxREAL);
    noise = mxGetPr(trhs[0]);
    noise[0] = 1;
    noise[1] = Ns;
    mexCallMATLAB(1, tlhs, 1, trhs, "randn");
    mxDestroyArray(trhs[0]);
    noise = mxGetPr(tlhs[0]);
    
    if(outType > 0) {
        /* If they want a voltage trace as the output */
        if(outType < 2) {
            plhs[0] = mxCreateDoubleMatrix(1, Ns, mxREAL);
            V = mxGetPr(plhs[0]);
            V[0] = El;
            logicMat = NULL;
            indMat = NULL;
        }
        else {
            indMat = (double*)mxCalloc(ceil(Ns/2),sizeof(double));
            V = NULL;
            Vim1 = El;
            logicMat = NULL;
        }
    }
    else {
        plhs[0] = mxCreateDoubleMatrix(1, Ns, mxREAL);
        V = NULL;
        Vim1 = El;
        logicMat = mxGetPr(plhs[0]);
        indMat = NULL;
    }

    /* Begin integration: */
    /* taum dVdt = rm*gsi*Psi*(Esi - V) + El - V + noise + rm*I */
    tinext = 0;
    count = 0;
    if(outType > 0) {
        /* inhibition, V[] */
        if(outType < 2) {
            for(ti=1; ti<Ns; ti++) {
                tim1 = ti-1;
                V[ti] = max(V[tim1] + dtOtaum*(rmgsi*Psi[ti]*(Esi - V[tim1]) + El - V[tim1] + noiselev*noise[ti] + rmI[ti]) , Esi);
                if(V[ti] >= Vth) {
                    if( ti >= tinext ) {
                        tinext = ti + minAPDist;
                        V[ti++] = Vap;
                        count++;
                        if(ti<Ns) {
                            V[ti] = Vre;
                        }
                    }
                    else {
                        V[ti] = Vth;
                    }
                }
            }
        }
        /* inhibition, indMat[] */
        else {
            for(ti=1; ti<Ns; ti++) {
                tim1 = ti-1;
                Vi = max(Vim1 + dtOtaum*(rmgsi*Psi[ti]*(Esi - Vim1) + El - Vim1 + noiselev*noise[ti] + rmI[ti]) , Esi);
                if(Vi >= Vth) {
                    if(ti >= tinext ) {
                        tinext = ti + minAPDist;
                        indMat[count++] = ++ti;
                        Vim1 = Vre;
                    }
                    else {
                        Vim1 = Vth;
                    }
                }
                else {
                    Vim1 = Vi;
                }
            }
        }
    }
    /* inhibition, logicMat[] */
    else {
        for(ti=1; ti<Ns; ti++) {
            tim1 = ti-1;
            Vi = max(Vim1 + dtOtaum*(rmgsi*Psi[ti]*(Esi - Vim1) + El - Vim1 + noiselev*noise[ti] + rmI[ti]) , Esi);
            if(Vi >= Vth) {
                if( ti >= tinext ) {
                    tinext = ti + minAPDist;
                    logicMat[ti++] = 1;
                    count++;
                    Vim1 = Vre;
                }
                else {
                    Vim1 = Vth;
                }
            }
            else {
                Vim1 = Vi;
            }
        }
    }
    
    /* indMat[] */
    if(outType >= 2) {
        plhs[0] = mxCreateDoubleMatrix(1, count, mxREAL);
        V = mxGetPr(plhs[0]);
        memcpy(V, indMat, sizeof(double)*count);
        mxFree(indMat);
    }

    if(nlhs >= 2) {
        plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
        outcount = mxGetPr(plhs[1]);
        outcount[0] = count;
    }
}
