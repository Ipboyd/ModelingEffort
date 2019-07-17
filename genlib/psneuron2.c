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
    double *Pse, *Psi;
    double dt, Ese, Esi, dtOtaum, Vap, Vre, Vth, El, minAPDist, Vim1;
    int count = 0;
    int outType = 0;
    int ti = 0, tinext;
    int ii = 0;
    bool *logicMat, withInhibition;
    int maxOut = -1;
    double *V;
    mwSize Psr, Psc, Ns;
    
    /* Check for proper number of arguments. */
    if(nrhs<11) {
        mexErrMsgTxt("11 inputs required.");
    } else if(nlhs>2) {
        mexErrMsgTxt("Too many output arguments");
    }
    
    /* The inputs must be noncomplex double row vectors. */
    Psr = mxGetM(prhs[0]);
    Psc = mxGetN(prhs[0]);
    withInhibition = mxGetN(prhs[1])*mxGetM(prhs[1]) > 1;
    if(nrhs >= 12) {
        outType = mxGetScalar(prhs[11]);
        if(nrhs >= 13)
            maxOut = mxGetScalar(prhs[12]);
    }
    if(
    !mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]) ||
    !mxIsDouble(prhs[3]) || mxIsComplex(prhs[3]) ||
    !mxIsDouble(prhs[4]) || mxIsComplex(prhs[4]) ||
    !mxIsDouble(prhs[5]) || mxIsComplex(prhs[5]) ||
    !mxIsDouble(prhs[6]) || mxIsComplex(prhs[6]) ||
    !mxIsDouble(prhs[7]) || mxIsComplex(prhs[7]) ||
    !mxIsDouble(prhs[8]) || mxIsComplex(prhs[8]) ||
    !mxIsDouble(prhs[9]) || mxIsComplex(prhs[9]) ||
    !mxIsDouble(prhs[10])|| mxIsComplex(prhs[10])||
    !mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) || 
    !mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) ||
    !(Psr==1 || Psc==1)  ||
    (Psr*Psc != mxGetM(prhs[1])*mxGetN(prhs[1]) && withInhibition)
    ) {
        mexErrMsgTxt("Incorrect inputs!");
    }

    /* Set max lengths */
    Ns = Psr;
    if(Psc>Psr)
        Ns = Psc;
    if(maxOut < 0)
        maxOut = Ns;
    
    /* Pse, Psi, arp, dt, Ese, Esi, taum, Vap, Vth, Vre, El (i.e. Vss) */
    Pse = mxGetPr(prhs[0]);
    Psi = mxGetPr(prhs[1]);
    dt = mxGetScalar(prhs[3]);
    if(dt <= 0)
        mexErrMsgTxt("Error: dt <= 0");

    /* DO NOT translate calculations so El = 0 (denormalization!) */
    Ese = mxGetScalar(prhs[4]);
    Esi = mxGetScalar(prhs[5]);
    dtOtaum = dt/mxGetScalar(prhs[6]);
    Vap = mxGetScalar(prhs[7]);
    Vre = mxGetScalar(prhs[8]);
    Vth = mxGetScalar(prhs[9]);
    El = mxGetScalar(prhs[10]);
    minAPDist = max((int)(mxGetScalar(prhs[2])/dt),1);

    if(maxOut <= 0)
        maxOut = Ns;

    switch(outType) {
        case 0:
            Ns = min(Ns,maxOut);
            plhs[0] = mxCreateLogicalMatrix(1, Ns);
            logicMat = mxGetLogicals(plhs[0]);
            break;
        case 1:
            Ns = min(Ns,maxOut);
            plhs[0] = mxCreateDoubleMatrix(1, Ns, mxREAL);
            V = mxGetPr(plhs[0]);
            break;
        case 2:
            plhs[0] = mxCreateDoubleMatrix(1, maxOut, mxREAL);
            V = mxGetPr(plhs[0]);
            break;
    }
    Vim1 = El;

    /* Begin integration: */
    /* taum dVdt = rmgsePse*(Ese - V) + rmgsiPsi*(Esi - V) + El - V */
    tinext = -Ns;
    if(withInhibition) {
        switch(outType) {
            case 1:
                if(Ns > 0)
                    *V = El;
                for(ti=1-Ns; ti; ti++) {
                    *++V = Vim1 + dtOtaum*(*++Pse*(Ese - Vim1) + *++Psi*(Esi - Vim1) - Vim1 + El );
                    if(*V < Esi)
                        *V = Esi;
                    else if(*V >= Vth) {
                        *V = Vap;
                        count++;
                        tinext = min(-(1+ti),max(minAPDist,1));
                        for(ii=tinext;ii;ii--)
                            *++V = Vre;
                        ti += tinext;
                        Pse += tinext;
                        Psi += tinext;
                    }
                    Vim1 = *V;
                }
                break;
            case 2:
                for(ti=1-Ns; ti; ti++) {
                    Vim1 += dtOtaum*(*++Pse*(Ese - Vim1) + *++Psi*(Esi - Vim1) - Vim1 + El );
                    if(Vim1 < Esi)
                        Vim1 = Esi;
                    else if(Vim1 >= Vth) {
                        count++;
                        tinext = min(-(1+ti),max(minAPDist,1));
                        *V++ = ti+1+Ns;
                        ti += tinext;
                        Pse += tinext;
                        Psi += tinext;
                        Vim1 = Vre;
                        if(count>=maxOut)
                            ti = -1;
                    }
                }
                break;
            default:
                for(ti=1-Ns; ti; ti++) {
                    Vim1 += dtOtaum*(*++Pse*(Ese - Vim1) + *++Psi*(Esi - Vim1) - Vim1 + El );
                    if(Vim1 < Esi)
                        Vim1 = Esi;
                    else if(Vim1 >= Vth) {
                        count++;
                        tinext = min(-(1+ti),max(minAPDist,1));
                        logicMat[ti+1+Ns] = true;
                        ti += tinext;
                        Pse += tinext;
                        Psi += tinext;
                        Vim1 = Vre;
                    }
                }
                break;
        }
    }
    else {
        switch(outType) {
            case 1:
                if(Ns > 0)
                    *V = El;
                for(ti=1-Ns; ti; ti++) {
                    *++V = Vim1 + dtOtaum*(*++Pse*(Ese - Vim1) - Vim1 + El );
                    if(*V >= Vth) {
                        *V = Vap;
                        count++;
                        tinext = min(-(1+ti),max(minAPDist,1));
                        for(ii=tinext;ii;ii--)
                            *++V = Vre;
                        ti += tinext;
                        Pse += tinext;
                    }
                    Vim1 = *V;
                }
                break;
            case 2:
                for(ti=1-Ns; ti; ti++) {
                    Vim1 += dtOtaum*(*++Pse*(Ese - Vim1) - Vim1 + El );
                    if(Vim1 >= Vth) {
                        count++;
                        tinext = min(-(1+ti),max(minAPDist,1));
                        *V++ = ti+1+Ns;
                        ti += tinext;
                        Pse += tinext;
                        Vim1 = Vre;
                        if(count>=maxOut)
                            ti = -1;
                    }
                }
                break;
            default:
                for(ti=1-Ns; ti; ti++) {
                    Vim1 += dtOtaum*(*++Pse*(Ese - Vim1) - Vim1 + El );
                    if(Vim1 >= Vth) {
                        count++;
                        tinext = min(-(1+ti),max(minAPDist,1));
                        logicMat[ti+1+Ns] = true;
                        ti += tinext;
                        Pse += tinext;
                        Vim1 = Vre;
                    }
                }
                break;
        }
    }
    
    /* V[] == indMat[] */
    if(outType >= 2) {
        if(count) {
            mxSetN(plhs[0],count);
            mxSetPr(plhs[0], mxRealloc(mxGetPr(plhs[0]),sizeof(double)*count));
        }
        else {
            plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
        }
    }

    if(nlhs >= 2) {
        plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
        V = mxGetPr(plhs[1]);
        V[0] = count;
    }
}
