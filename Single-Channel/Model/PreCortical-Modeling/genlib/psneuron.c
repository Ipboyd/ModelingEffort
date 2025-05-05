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
    double *Pse, *Psi, *V, *outcount, *noise, *logicMat, *indMat;
    double dt, dtOtaum, Ese, Esi, rmgse, rmgsi, Vap, Vth, Vre, noiselev, El, Vi, Vim1;
    int ti, tim1, tinext, count, minAPDist;
    bool withInhibition, outType;
    mwSize Psr, Psc, Ns;
    mxArray *tlhs[1], *trhs[1];
    
    /* Check for proper number of arguments. */
    if(nrhs<14) {
        mexErrMsgTxt("11 inputs required.");
    } else if(nlhs>2) {
        mexErrMsgTxt("Too many output arguments");
    }
    
    /* The inputs must be noncomplex double row vectors. */
    Psr = mxGetM(prhs[0]);
    Psc = mxGetN(prhs[0]);
    withInhibition = (mxGetN(prhs[1])>1 || mxGetM(prhs[1])>1);
    if(nrhs >= 15)
        outType = mxGetScalar(prhs[14]);
    else
        outType = 0;
    if( nrhs < 14 ||
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
    !mxIsDouble(prhs[12])|| mxIsComplex(prhs[12])||
    !mxIsDouble(prhs[13])|| mxIsComplex(prhs[13])||
    !mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) || 
    !mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) ||
    !(Psr==1 || Psc==1)  ||
    (Psr*Psc != mxGetM(prhs[1])*mxGetN(prhs[1]) && withInhibition)
    ) {
        mexErrMsgTxt("Incorrect inputs!");
    }

    /* Set max lengths */
    if(Psc>Psr) Ns = Psc; else Ns = Psr;
    
    /* Pse, Psi, arp, dt, Ese, Esi, rmgse, rmgsi, taum, Vap, Vth, Vre, El (i.e. Vss), noiselev */
    Pse = mxGetPr(prhs[0]);
    Psi = mxGetPr(prhs[1]);
    dt = mxGetScalar(prhs[3]);
    if(dt <= 0)
        mexErrMsgTxt("Error: dt <= 0");
    El = mxGetScalar(prhs[12]);

    /* DO NOT translate calculations so El = 0 (denormalization!) */
    Ese = mxGetScalar(prhs[4]);
    Esi = mxGetScalar(prhs[5]);
    rmgse = mxGetScalar(prhs[6]);
    rmgsi = mxGetScalar(prhs[7]);
    dtOtaum = dt/mxGetScalar(prhs[8]);
    Vap = mxGetScalar(prhs[9]);
    Vre = mxGetScalar(prhs[10]);
    Vth = mxGetScalar(prhs[11]);
    noiselev = mxGetScalar(prhs[13]);
    minAPDist = (int)(mxGetScalar(prhs[2])/dt);

    /* Get some AWGN */
    if(noiselev != 0) {
        trhs[0] = mxCreateDoubleMatrix(1, 2, mxREAL);
        noise = mxGetPr(trhs[0]);
        noise[0] = 1;
        noise[1] = Ns;
        mexCallMATLAB(1, tlhs, 1, trhs, "randn");
        mxDestroyArray(trhs[0]);
        noise = mxGetPr(tlhs[0]);
    }
    else {
        trhs[0] = mxCreateDoubleMatrix(1, Ns, mxREAL);
        noise = mxGetPr(trhs[0]);
    }
    plhs[0] = mxCreateDoubleMatrix(1, Ns, mxREAL);
    
    if(outType > 0) {
        /* If they want a voltage trace as the output */
        if(outType < 2) {
            V = mxGetPr(plhs[0]);
            V[0] = El;
            logicMat = NULL;
            indMat = NULL;
        }
        else {
            indMat = mxGetPr(plhs[0]);
            V = NULL;
            Vim1 = El;
            logicMat = NULL;
        }
    }
    else {
        V = NULL;
        Vim1 = El;
        logicMat = mxGetPr(plhs[0]);
        indMat = NULL;
    }

    /* Begin integration: */
    /* taum dVdt = rm*gse*Pse*(Ese - V) + rm*gsi*Psi*(Esi - V) + El - V + noise */
    tinext = 0;
    count = 0;
    if(withInhibition) {
        if(outType > 0) {
            /* inhibition, V[] */
            if(outType < 2) {
                for(ti=1; ti<Ns; ti++) {
                    tim1 = ti-1;
                    V[ti] = max(V[tim1] + dtOtaum*(rmgse*Pse[ti]*(Ese - V[tim1]) + rmgsi*Psi[ti]*(Esi - V[tim1]) - V[tim1] + El + noiselev*noise[ti] ) , Esi);
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
                    Vi = max(Vim1 + dtOtaum*(rmgse*Pse[ti]*(Ese - Vim1) + rmgsi*Psi[ti]*(Esi - Vim1) - Vim1 + El + noiselev*noise[ti] ) , Esi);
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
                Vi = max(Vim1 + dtOtaum*(rmgse*Pse[ti]*(Ese - Vim1) + rmgsi*Psi[ti]*(Esi - Vim1) - Vim1 + El + noiselev*noise[ti] ) , Esi);
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
    }
    else {
        if(outType > 0) {
            /* no inhibition, V[] */
            if(outType < 2) {
                for(ti=1; ti<Ns; ti++) {
                    tim1 = ti-1;
                    V[ti] = V[tim1] + dtOtaum*(rmgse*Pse[ti]*(Ese - V[tim1]) - V[tim1] + El + noiselev*noise[ti] );
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
            /* no inhibition, indMat[] */
            else {
                for(ti=1; ti<Ns; ti++) {
                    tim1 = ti-1;
                    Vi = Vim1 + dtOtaum*(rmgse*Pse[ti]*(Ese - Vim1) - Vim1 + El + noiselev*noise[ti] );
                    if(Vi >= Vth) {
                        if( ti >= tinext ) {
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
        /* no inhibition, logicMat[] */
        else {
            for(ti=1; ti<Ns; ti++) {
                tim1 = ti-1;
                Vi = Vim1 + dtOtaum*(rmgse*Pse[ti]*(Ese - Vim1) - Vim1 + El + noiselev*noise[ti] );
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
    }
    
    /* indMat[] */
    if(outType >= 2) {
        if(count) {
            mxSetN(plhs[0],count);
            indMat = (double*)mxRealloc((void*)indMat,sizeof(double)*count);
        }
        else {
            plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
        }
    }

    if(nlhs >= 2) {
        plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
        outcount = mxGetPr(plhs[1]);
        outcount[0] = count;
    }
}
