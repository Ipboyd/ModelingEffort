#include "mex.h"
#include "math.h"

#ifndef max
#define max(a,b)   (((a) > (b)) ? (a) : (b))
#endif
#ifndef min
#define min(a,b)   (((a) < (b)) ? (a) : (b))
#endif

double* scalarOrMat2Mat(const mxArray *in, mwSize Nn);

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
    double **rmgsPse, **rmgsPsi, **rmI, *rmgsPsein, *rmgsPsiin, **V, **rmgse, **rmgsi, **noise;
    double *arp, *taum, *dtOtaum, *Ese, *Esi, *Vap, *Vth, *Vre, *noiselev, *El, *taue, *pole, *delay;
    double dt, *Vi, *Vim1, *count, outType;
    int ni, oi, ti, tim1, tempint, *tinext, *minAPdist, **intDelay;
    mwSize Nn, Ns;
    mxArray *tlhs[1], *trhs[1];
    
    /* Check for proper number of argumeNss. */
    if(nrhs < 16 || nlhs > 2) {
        mexErrMsgTxt("16, 17, or 18 inputs and at most 2 outputs required.");
    }
    
    /* The inputs must be noncomplex double row vectors. */
    Ns = mxGetM(prhs[0]);
    Nn = mxGetN(prhs[0]);
    
    /* Initialize double pointers */
    V = (double**)mxCalloc(Nn,sizeof(double*));
    rmI = (double**)mxCalloc(Nn,sizeof(double*));
    rmgse = (double**)mxCalloc(Nn,sizeof(double*));
    rmgsi = (double**)mxCalloc(Nn,sizeof(double*));
    rmgsPse = (double**)mxCalloc(Nn,sizeof(double*));
    rmgsPsi = (double**)mxCalloc(Nn,sizeof(double*));
    noise = (double**)mxCalloc(Nn,sizeof(double*));
    intDelay = (int**)mxCalloc(Nn,sizeof(int*));
    intDelay[0] = (int*)mxCalloc(Nn*Nn,sizeof(int));
    outType = 0;
    
    /* 01: rmgsPsein (Ns x Nn) */
    if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]))
        mexErrMsgTxt("Error 01: rmgsPsein must be a real double Ns x Nn matrix");
    /* 02: rmgsPsiin (Ns x Nn) */
    if(!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) || Ns != mxGetM(prhs[1]) || Nn != mxGetN(prhs[1]))
        mexErrMsgTxt("Error 02: rmgsPsiin must be a real double Ns x Nn matrix");
    /* 03: rmI (Ns x Nn) */
    if(!mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]) || Ns != mxGetM(prhs[2]) || Nn != mxGetN(prhs[2]))
        mexErrMsgTxt("Error 03: rmI must be a real double Ns x Nn matrix");
    /* 04: arp (Nn) */
    tempint = mxGetM(prhs[3])*mxGetN(prhs[3]);
    if(!mxIsDouble(prhs[3]) || mxIsComplex(prhs[3]) || (Nn != tempint && 1 != tempint ))
        mexErrMsgTxt("Error 04: arp must be a real double Nn vector or scalar");
    /* 05: dt (1) */
    tempint = mxGetM(prhs[4])*mxGetN(prhs[4]);
    if (!mxIsDouble(prhs[4]) || mxIsComplex(prhs[4]) || 1 != tempint)
        mexErrMsgTxt("Error 05: dt must be a real double scalar");
    dt = mxGetScalar(prhs[4]);
    if(dt <= 0) mexErrMsgTxt("Error 05: dt <= 0");
    /* 06: Ese (Nn) */
    tempint = mxGetM(prhs[5])*mxGetN(prhs[5]);
    if (!mxIsDouble(prhs[5]) || mxIsComplex(prhs[5]) || (Nn != tempint && 1 != tempint ))
        mexErrMsgTxt("Error 06: Ese must be a real double Nn vector or scalar");
    /* 07: Esi (Nn) */
    tempint = mxGetM(prhs[6])*mxGetN(prhs[6]);
    if (!mxIsDouble(prhs[6]) || mxIsComplex(prhs[6]) || (Nn != tempint && 1 != tempint ))
        mexErrMsgTxt("Error 07: Esi must be a real double Nn vector or scalar");
    /* 08: rmgse (Nn x Nn) */
    if (!mxIsDouble(prhs[7]) || mxIsComplex(prhs[7]) || Nn != mxGetM(prhs[7]) || Nn != mxGetN(prhs[7]))
        mexErrMsgTxt("Error 08: rmgse must be a real double Nn x Nn matrix");
    /* 09: rmgsi (Nn x Nn)  */
    if (!mxIsDouble(prhs[8]) || mxIsComplex(prhs[8]) || Nn != mxGetM(prhs[8]) || Nn != mxGetN(prhs[8]))
        mexErrMsgTxt("Error 09: rmgsi must be a real double Nn x Nn matrix");
    /* 10: taum (Nn) */
    tempint = mxGetM(prhs[9])*mxGetN(prhs[9]);
    if (!mxIsDouble(prhs[9]) || mxIsComplex(prhs[9]) || (Nn != tempint && 1 != tempint ))
        mexErrMsgTxt("Error 10: taum must be a real double Nn vector or scalar");
    /* 11: Vap (Nn) */
    tempint = mxGetM(prhs[10])*mxGetN(prhs[10]);
    if (!mxIsDouble(prhs[10]) || mxIsComplex(prhs[10]) || (Nn != tempint && 1 != tempint ))
        mexErrMsgTxt("Error 11: Vap must be a real double Nn vector or scalar");
    /* 13: Vre (Nn) */
    tempint = mxGetM(prhs[11])*mxGetN(prhs[11]);
    if (!mxIsDouble(prhs[11])|| mxIsComplex(prhs[11]) || (Nn != tempint && 1 != tempint ))
        mexErrMsgTxt("Error 12: Vre must be a real double Nn vector or scalar");
    /* 12: Vth (Nn) */
    tempint = mxGetM(prhs[12])*mxGetN(prhs[12]);
    if (!mxIsDouble(prhs[12])|| mxIsComplex(prhs[12]) || (Nn != tempint && 1 != tempint ))
        mexErrMsgTxt("Error 13: Vth must be a real double Nn vector or scalar");
    /* 14: El (Nn) */
    tempint = mxGetM(prhs[13])*mxGetN(prhs[13]);
    if (!mxIsDouble(prhs[13])|| mxIsComplex(prhs[13]) || (Nn != tempint && 1 != tempint ))
        mexErrMsgTxt("Error 14: El must be a real double Nn vector or scalar");
    /* 15: noiselev (Nn) */
    tempint = mxGetM(prhs[14])*mxGetN(prhs[14]);
    if (!mxIsDouble(prhs[14])|| mxIsComplex(prhs[14]) || (Nn != tempint && 1 != tempint ))
        mexErrMsgTxt("Error 15: noiselev must be a real double Nn vector or scalar");
    /* 16: taue (Nn) */
    tempint = mxGetM(prhs[15])*mxGetN(prhs[15]);
    if (!mxIsDouble(prhs[15])|| mxIsComplex(prhs[15]) || (Nn != tempint && 1 != tempint ))
        mexErrMsgTxt("Error 16: taue must be a real double Nn vector or scalar");
    /* 17: delay (Nn x Nn, scalar, or empty) */
    if (nrhs >= 17) {
        if(!mxIsEmpty(prhs[16])) {
            tempint = mxGetM(prhs[16])*mxGetN(prhs[16]);
            if(!mxIsDouble(prhs[16]) || mxIsComplex(prhs[16])) {
                mexErrMsgTxt("Error 17: delay must be a real double Nn x Nn matrix, scalar, or empty matrix");
            }
            else if(Nn != mxGetM(prhs[16]) || Nn != mxGetN(prhs[16])) {
                if(tempint == 1) {
                    tempint = (int)(mxGetScalar(prhs[16])/dt);
                    ti = Nn*Nn;
                    for(ni=0;ni<ti; ni++)
                        intDelay[0][ni] = tempint;
                    }
                else {
                    mexErrMsgTxt("Error 17: delay must be a real double Nn x Nn matrix, scalar, or empty matrix");
                }
            }
            else {
                delay = mxGetPr(prhs[16]);
                ti = Nn*Nn;
                for(ni=0;ni<ti; ni++)
                    intDelay[0][ni] = (int)(delay[ni]/dt);
            }
        }
    /* 17: outType (1) */
        if(nrhs >= 18) {
            if(!mxIsDouble(prhs[17]) || mxIsComplex(prhs[17]) || 1 != mxGetM(prhs[17])*mxGetN(prhs[17]))
                mexErrMsgTxt("Error 18: outType must be a real scalar");
            outType = mxGetScalar(prhs[17]);
        }
    }
    
    rmgsPsein = mxGetPr(prhs[0]);
    rmgsPsiin = mxGetPr(prhs[1]);
    rmI[0] = mxGetPr(prhs[2]);
    arp = scalarOrMat2Mat(prhs[3],Nn);
    Ese = scalarOrMat2Mat(prhs[5],Nn);
    Esi = scalarOrMat2Mat(prhs[6],Nn);
    rmgse[0] = mxGetPr(prhs[7]);
    rmgsi[0] = mxGetPr(prhs[8]);
    taum = scalarOrMat2Mat(prhs[9],Nn);
    Vap = scalarOrMat2Mat(prhs[10],Nn);
    Vre = scalarOrMat2Mat(prhs[11],Nn);
    Vth = scalarOrMat2Mat(prhs[12],Nn);
    El = scalarOrMat2Mat(prhs[13],Nn);
    noiselev = scalarOrMat2Mat(prhs[14],Nn);
    taue = scalarOrMat2Mat(prhs[15],Nn);
    
    /* Initialize V */
    plhs[0] = mxCreateDoubleMatrix(Ns, Nn, mxREAL);
    V[0] = mxGetPr(plhs[0]);
    
    /* Get some AWGN */
    trhs[0] = mxCreateDoubleMatrix(1, 2, mxREAL);
    noise[0] = mxGetPr(trhs[0]);
    noise[0][0] = Ns;
    noise[0][1] = Nn;
    mexCallMATLAB(1, tlhs, 1, trhs, "randn");
    mxDestroyArray(trhs[0]);
    noise[0] = mxGetPr(tlhs[0]);
    
    /* Fix up double pointers, initialize V */
    /* Convert ARP to #-of-samples; put in minAPdist; combine dt & taums; convert taue to poles */
    if(taum[0] <= 0) mexErrMsgTxt("Error 10: taum <= 0");
    if(taue[0] <= 0) mexErrMsgTxt("Error 16: taue <= 0");
    
    rmgsPse[0] = (double*)mxCalloc(Nn*Ns,sizeof(double));
    rmgsPsi[0] = (double*)mxCalloc(Nn*Ns,sizeof(double));
    minAPdist = (int*)mxCalloc(Nn,sizeof(int));
    dtOtaum = (double*)mxCalloc(Nn,sizeof(double));
    tinext = (int*)mxCalloc(Nn,sizeof(int));
    pole = (double*)mxCalloc(Nn,sizeof(double));
    V[0][0] = El[0];
    minAPdist[0] = (int)(arp[0]/dt);
    dtOtaum[0] = dt/taum[0];
    pole[0] = exp(-dt/taue[0]);
    for(ni=1;ni<Nn; ni++) {
        if(taum[ni] <= 0) mexErrMsgTxt("Error 10: taum <= 0");
        if(taue[ni] <= 0) mexErrMsgTxt("Error 16: taue <= 0");
        rmgse[ni] = rmgse[ni-1] + Nn;
        rmgsi[ni] = rmgsi[ni-1] + Nn;
        intDelay[ni] = intDelay[ni-1] + Nn;
        rmgsPse[ni] = rmgsPse[ni-1] + Ns;
        rmgsPsi[ni] = rmgsPsi[ni-1] + Ns;
        rmI[ni] = rmI[ni-1] + Ns;
        V[ni] = V[ni-1] + Ns;
        V[ni][0] = El[ni];
        noise[ni] = noise[ni-1] + Ns;
        minAPdist[ni] = (int)(arp[ni]/dt);
        dtOtaum[ni] = dt/taum[ni];
        pole[ni] = exp(-dt/taue[ni]);
    }
    
    /* Initialize Pse and Psi */
    memcpy(rmgsPse[0],rmgsPsein,sizeof(double)*Ns*Nn);
    memcpy(rmgsPsi[0],rmgsPsiin,sizeof(double)*Ns*Nn);
    
    /* Initialize counting variable */
    if(nlhs >= 2) {
        plhs[1] = mxCreateDoubleMatrix(1, Nn, mxREAL);
        count = mxGetPr(plhs[1]);
    }
    else
        count = (double*)mxCalloc(Nn,sizeof(double));
    
    /* Begin integration: */
    for(ti=1; ti<Ns; ti++) {
        tim1 = ti-1;
        
        /* Propegate APs from previous time step */
        for(ni=0; ni<Nn; ni++) {
            if(V[ni][tim1] == Vap[ni])
                for(oi=0; oi<Nn; oi++) {
                    tempint = ti + intDelay[ni][oi];
                    if(tempint < Ns) {
                        rmgsPse[oi][tempint] += rmgse[ni][oi];
                        rmgsPsi[oi][tempint] += rmgsi[ni][oi];
                    }
                }
        }
        
        /* Perform expconv-like convolution on Ps !!!!! */
        for(ni=0; ni<Nn; ni++) {
            rmgsPse[ni][ti] += pole[ni]*rmgsPse[ni][tim1];
            rmgsPsi[ni][ti] += pole[ni]*rmgsPsi[ni][tim1];
        }
        
        /* taum dVdt = rmgsPse*(Ese - V) + rmgsPsi*(Esi - V) + El - V + noise */
        for(ni=0; ni<Nn; ni++) {
            if(V[ni][tim1] == Vap[ni])
                V[ni][ti] = Vre[ni];
            else {
                V[ni][ti] = V[ni][tim1] + dtOtaum[ni]*(rmgsPse[ni][ti]*(Ese[ni] - V[ni][tim1]) + rmgsPsi[ni][ti]*(Esi[ni] - V[ni][tim1]) + El[ni] - V[ni][tim1] + noiselev[ni]*noise[ni][ti] + rmI[ni][ti]);
                if(V[ni][ti] >= Vth[ni]) {
                    if( ti >= tinext[ni] ) {
                        tinext[ni] = ti + minAPdist[ni];
                        V[ni][ti] = Vap[ni];
                        count[ni]++;
                    }
                    else {
                        V[ni][ti] = Vth[ni];
                    }
                }
            }
        }
    }
    
    if(nlhs < 2)
        mxFree(count);
    if(outType <= 0)
        for(ni=0;ni<Nn;ni++)
            for(ti=0;ti<Ns;ti++)
                V[ni][ti] = (V[ni][ti] >= Vap[ni]);
    
    mxFree(rmgsPse[0]);
    mxFree(rmgsPsi[0]);
    mxFree(intDelay[0]);
    mxFree(rmgsPse);
    mxFree(rmgsPsi);
    mxFree(rmgse);
    mxFree(rmgsi);
    mxFree(V);
    mxFree(rmI);
    mxFree(noise);
    mxFree(minAPdist);
    mxFree(dtOtaum);
    mxFree(tinext);
    mxFree(intDelay);
}

double* scalarOrMat2Mat(const mxArray *in, mwSize Nn)
{
    double *out, *inmat, inscalar;
    int ni, tempint;
    
    tempint = mxGetM(in)*mxGetN(in);
    if(tempint == 1) {
        inscalar = mxGetScalar(in);
        out = (double*)mxCalloc(Nn,sizeof(double));
        for(ni=0;ni<Nn;ni++)
            out[ni] = inscalar;
    }
    else
        out = mxGetPr(in);
    
    return out;
}

/* Nn=4;Ns=100;rmgsi=eye(Nn);rmgse=0.3*(ones(Nn)-eye(Nn));z=zeros(Ns,Nn);z1=zeros(Nn,1);[a b] = psneurons2(z,z,0.002,0.001,0,-90,rmgse,rmgsi,0.02,2,-55,-80,-70,100.5,0.1,1);plot(a) */
