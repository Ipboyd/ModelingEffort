#include "mex.h"
#include "math.h"
#include "float.h"

/* #define IFSC_PREC 1e-4 */

extern double _taus;
void IFSC_Init(double taus); /* Init (create tables) and set constants */
void IFSC_Done();            /* Delete the tables*/
void IFSC_IncomingSpike(double t, double w, double *V, double *g, double *Es, double Ep, double Em); /* 1) Update {V,g,Es} after incoming spike @ time t */
double IFSC_OutgoingSpike(double t, double g); /* 2) Return updated value of variable g after a spike @ t */
double IFSC_SpikeTiming(double V, double g, double Es); /* 3) Calculate the time of next spike */
bool IFSC_SpikeTest(double V, double g, double Es); /* The spike test (mostly for internal use) */

double SpikeTimingARP(double V, double g, double Es, double Ese, double Esi, double arp, double dtLastOutput);

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] ) {
    double *inTimes, *weights, *outMat, scaler, taus, taum;
    double arp, dur, Ese, Esi, Vth, Vre, El, V, g, Es, Tl;
    double nextIn, lastOut, nextOut, spikingTime, prevInTime;
    int ti, count, Nspikes;
    mwSize maxOut;

    /* Check for proper number of arguments. */
    if(nrhs<10) {
        mexErrMsgTxt("10, 11, or 12 inputs required.");
    }
    if(nrhs>12) {
        mexErrMsgTxt("10, 11, or 12 inputs required.");
    }
    if(nlhs>1) {
        mexErrMsgTxt("Too many output arguments");
    }

    /* The inputs must be noncomplex double row vectors. */
    if(     !mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) ||
            !mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) ||
            !mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]) || mxIsEmpty(prhs[2]) ||
            !mxIsDouble(prhs[3]) || mxIsComplex(prhs[3]) || mxIsEmpty(prhs[3]) ||
            !mxIsDouble(prhs[4]) || mxIsComplex(prhs[4]) || mxIsEmpty(prhs[4]) ||
            !mxIsDouble(prhs[5]) || mxIsComplex(prhs[5]) || mxIsEmpty(prhs[5]) ||
            !mxIsDouble(prhs[6]) || mxIsComplex(prhs[6]) || mxIsEmpty(prhs[6]) ||
            !mxIsDouble(prhs[7]) || mxIsComplex(prhs[7]) || mxIsEmpty(prhs[7]) ||
            !mxIsDouble(prhs[8]) || mxIsComplex(prhs[8]) || mxIsEmpty(prhs[8]) ||
            !mxIsDouble(prhs[9]) || mxIsComplex(prhs[9]) || mxIsEmpty(prhs[9])
            ) {
        mexErrMsgTxt("Inputs must be nonempty real doubles!");
    }

    /* If input is empty, output is empty */
    if(mxIsEmpty(prhs[0]) || mxIsEmpty(prhs[1])) {
        plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
        return;
    }

    /* Spikes, Weights, arp, Ese, Esi, taum, taus, Vre, Vth, El (i.e. Vss), fs */
    /* Translate calculations so El = 0, translate back later if necessary */
    inTimes = mxGetPr(prhs[0]);
    weights = mxGetPr(prhs[1]);
    arp = mxGetScalar(prhs[2]);
    Vth = mxGetScalar(prhs[8]);
    El = mxGetScalar(prhs[9]);
    scaler = Vth - El; if(scaler <= 0) mexErrMsgTxt("Vth must be greater than El!");
    Vth = (Vth-El)/scaler;
    Ese = (mxGetScalar(prhs[3]) - El)/scaler;
    Esi = (mxGetScalar(prhs[4]) - El)/scaler;
    taum = mxGetScalar(prhs[5]); if(taum <= 0) mexErrMsgTxt("taum must be greater than 0!");
    taus = mxGetScalar(prhs[6])/taum;
    arp /= taum;
    Vre = (mxGetScalar(prhs[7]) - El)/scaler;
    Nspikes = mxGetNumberOfElements(prhs[0]); if(Nspikes != mxGetNumberOfElements(prhs[1])) mexErrMsgTxt("Number of input spikes must equal\nnumber of weights!");

    if(nrhs >= 11) {
        if(!mxIsDouble(prhs[10])|| mxIsComplex(prhs[10]) || mxIsEmpty(prhs[10]))
            mexErrMsgTxt("dur must be a nonempty real double!");
        else
            dur = mxGetScalar(prhs[10]);
    }
    else {
        if(Nspikes)
            dur = inTimes[Nspikes-1]+5*(taum+taus*taum); /* Go 10 time constants beyond last input */
        else
            dur = 0;
    }
    if(nrhs >= 12) {
        if(!mxIsDouble(prhs[11])|| mxIsComplex(prhs[11]) || mxIsEmpty(prhs[11]))
            mexErrMsgTxt("maxOut must be a nonempty real double!");
        else
            maxOut = mxGetScalar(prhs[11]);
    }
    else {
        maxOut = (int)(1000*dur); /* Save room for 1000 Hz firing rate response */
    }
    plhs[0] = mxCreateDoubleMatrix(maxOut, 1, mxREAL);
    outMat = mxGetPr(plhs[0]);
    IFSC_Init(taus);

    Tl = 0; /* Time of last update */
    V = 0;  /* Membrane potential */
    g = 0;  /* Synaptic conductance */
    Es = 0; /* Effective reversal potential */
    lastOut = arp; /* How long ago was the last outputted spike */

    count = 0;

    prevInTime = -DBL_MAX;
    for(ti=-1; ti<Nspikes; ti++) {
        /* Let the neuron spike, if it needs to, to start */
        if(ti < 0) {
            nextIn = (Nspikes > 0) ? (inTimes[0]/taum) : (dur/taum);
        }
        /* Otherwise, process the next input spike */
        else {
            if(inTimes[ti] < prevInTime) {
                char errorText[100];
                sprintf(errorText,"Spike time %i (%f) < spike time %i (%f)",ti+1,inTimes[ti],ti,prevInTime);
                mexErrMsgTxt(errorText);
            }
            prevInTime = inTimes[ti];
                
            IFSC_IncomingSpike(inTimes[ti]/taum-Tl, weights[ti], &V, &g, &Es, Ese, Esi);
            nextIn = (ti == Nspikes-1) ? ((dur-inTimes[ti])/taum) : ((inTimes[ti+1]-inTimes[ti])/taum);
            Tl = inTimes[ti]/taum;
        }

        /* Calculate the time of the next spike following the ARP */
        nextOut = SpikeTimingARP(V,g,Es,Ese,Esi,arp,Tl-lastOut);
        spikingTime = nextOut;

        while(nextOut > 0 && spikingTime < nextIn && (Tl+nextOut) <= dur/taum) {
            /* A valid spike has occurred: store output and update times */
            Tl += nextOut;
            lastOut = Tl;
            outMat[count++] = Tl*taum;

            /* Reset values */
            g = IFSC_OutgoingSpike(nextOut, g);
            V = Vre;

            /* Recalculate next spike time */
            nextOut = SpikeTimingARP(V,g,Es,Ese,Esi,arp,Tl-lastOut);
            spikingTime += nextOut;

            /* Check to make sure we aren't exceeding the size of output */
            if(count >= maxOut) {
                spikingTime = nextIn;
                ti = Nspikes;
            }
        }
    }
    IFSC_Done();

    /* Trim the output array */
    if(count) {
        mxSetM(plhs[0],count);
        outMat = (double*)mxRealloc((void*)outMat,sizeof(double)*count);
    }
    else {
        plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
    }
}

double SpikeTimingARP(double V, double g, double Es, double Ese, double Esi, double arp, double lastOut) {
    double temp, dt, nextElig;
    
    dt = IFSC_SpikeTiming(V, g, Es);
    nextElig = arp-lastOut;

    /* If next calculated time is less than the next eligible time ... */
    if (dt > 0 && dt < nextElig) {
        /* Update spike variables to reflect the next eligible time */
        IFSC_IncomingSpike(nextElig, 0, &V, &g, &Es, Ese, Esi);

        /* If the neuron would be spiking @ ARP time, let it */
        if(V>=1) {
            dt = nextElig;
        }
        /* Otherwise, calculate the next eligible spike time */
        else {
            temp = IFSC_SpikeTiming(V, g, Es);
            if(temp > 0)
                dt = nextElig + temp;
            else
                dt = 0;
        }
    }
    /* Otherwise, return the next calculated time */
    return dt;
}

/* ******************************************************************************
/* Exact simulation of integrate-and-fire models with
/* exponential synaptic conductances
/*
/* Romain Brette
/* brette@di.ens.fr
/* Last updated: Feb 2007 - I thank Michiel D'Haene for pointing a couple of bugs
/* ******************************************************************************
/*
/* Insert #define IFSC_PREC 1e-4 in file IFSC.h to use precalculated tables
/* otherwise the original expressions are used
/*
/* All variables are normalized, i.e. V0=0, Vt=1, tau = 1
/*
/* N.B.: the code is not optimized */

/* #include "IFSC.h" */
#include <math.h>
/* #include <stdlib.h> */
/* #include <stdio.h> */

/* ************************************************************ */
/* CONSTANTS
/* ************************************************************ */

/* The neuron spikes when the membrane potential */
/* is within SPIKE_COMPUTATION_ERROR of the threshold */
#ifdef IFSC_PREC
#define SPIKE_COMPUTATION_ERROR	1e-10
#else
#define SPIKE_COMPUTATION_ERROR	1e-6
#endif

/* Relative error in calculating the incomplete Gamma integral */
#define MAX_GAMMA_ERROR 1e-15

/* Time constant of the synaptic conductances and its inverse */
/* (_taus is in units of the membrane time constant) */
double _taus;
double inv_taus;

double rho(double g);

#ifdef IFSC_PREC
/* ************************************************************
/* LOOK-UP TABLES
/* ************************************************************
/* 
/* The values are computed with the look-up tables 
/* using linear interpolation as follows:
/*
/* cell number:     n=(int)(x/dx)    (dx is the precision of the table)
/* remainder:       h=(x/dx)-n       (for the linear interpolation)
/* value:           y=table[n]+h*(table[n+1]-table[n])
/*
/* ----------------------------------------------------------
/*    Look-up table for the exponential (exp(-x))
/* ----------------------------------------------------------
/* */
/* Address */
double *expLookupTable;
/* Precision */
double expLookup_dx;
/* Inverse of precision */
double inv_expLookup_dx;
/* Size */
int expLookup_nmax;
/* 1./(_taus*expLookup_dx) */
double inv_taus_X_inv_expLookup_dx;

/* -------------------------------------------------- */
/*    Look-up table for the (modified) rho function
/*       rho(1-\tau_s,\tau_s g) * g * \tau_s
/* -------------------------------------------------- */
/* Address */
double *rhoLookupTable;
/* Precision */
double rhoLookup_dg;
/* Inverse of precision */
double inv_rhoLookup_dg;
/* Size */
int rhoLookup_nmax;

/* ----------------------------------------------- */
/* Functions for the exponential look-up table */
/* ----------------------------------------------- */
/* */
/* Build the look-up table for x in [0,xmax] with precision dx */
void makeExpLookupTable(double dx,double xmax) {
	double x;
	int n;

	expLookup_dx=dx;
	inv_expLookup_dx=1./dx;
	inv_taus_X_inv_expLookup_dx=inv_taus*inv_expLookup_dx;
	expLookup_nmax=(int)(xmax*inv_expLookup_dx);

	expLookupTable=(double *)
		malloc(sizeof(double)*(expLookup_nmax));

	x=0.;
	for(n=0;n<expLookup_nmax;n++) {
		expLookupTable[n]=exp(-x);
		x+=dx;
	}

	expLookup_nmax--;
}

/* Free the memory allocated for the table */
void freeExpLookupTable() {
	free((void *)expLookupTable);
}

/* tableExpM(x) returns exp(-x) from the look-up table */
/* (with linear interpolation) */
double tableExpM(double x) {
	double a,b;
	double *table;
	int n=(int)(x=x*inv_expLookup_dx);

	if ((n>=expLookup_nmax) || (n<0)) {
		return (exp(-x*expLookup_dx));
	}

	table=expLookupTable+n;
	a=*(table++);
	b=*table;

	return ( a + (x-n)*(b - a) );
}

/* ----------------------------------------------- */
/* Functions for the rho look-up table */
/* ----------------------------------------------- */
/*
/* Build the look-up table for g in [0,gmax] with precision dg */
void makeRhoLookupTable(double dg,double gmax) {
	double g;
	int n;

	rhoLookup_dg=dg;
	inv_rhoLookup_dg=1./dg;
	rhoLookup_nmax=(int)(gmax/dg);

	rhoLookupTable=(double *)
		malloc(sizeof(double)*(rhoLookup_nmax));

	g=0.;
	for(n=0;n<rhoLookup_nmax;n++) {
		rhoLookupTable[n]=rho(g)*g*_taus;
		g+=dg;
	}

	rhoLookup_nmax--;
}

/* Free the memory allocated for the table */
void freeRhoLookupTable() {
	free((void *)rhoLookupTable);
}

/* tableRho(g) returns rho(g) from the look-up table */
/* (with linear interpolation) */
double tableRho(double g) {
	double a,b;
	double *table;
	int n=(int)(g=g*inv_rhoLookup_dg);

	if ((n>=rhoLookup_nmax) || (n<0)) {
		return (rho(g*rhoLookup_dg));
	}

	table=rhoLookupTable+n;
	a=*(table++);
	b=*table;

	return ( a + (g-n)*(b - a) );
}

#endif

/* ************************** */
/* CONSTRUCTION & DESTRUCTION */
/* ************************** */

/* Initialize (create the tables) and */
/*   set the constants: */
/*     taus = synaptic time constants (in units of the membrane time constant) */
void IFSC_Init(double taus){
	_taus=taus;
	inv_taus=1./_taus;

	#ifdef IFSC_PREC
	makeExpLookupTable(IFSC_PREC,1.);
	makeRhoLookupTable(IFSC_PREC,1.);
	#endif
}

/* Delete the tables */
void IFSC_Done() {
	#ifdef IFSC_PREC
	freeRhoLookupTable();
	freeExpLookupTable();
	#endif
}

/* ************************************************************ */
/* RHO FUNCTION (based on incomplete gamma integral - see text) */
/* ************************************************************ */
/* rho(g) = \rho(1-\tau_s,\tau_s * g) */
/* (see text) */
/* */
/* We use the power series expansion of the incomplete gamma integral */
double rho(double g) {
    double sum, del, ap;
	double x=_taus*g;

	/* Note: all numbers are always positive */
	ap = 1.-_taus;
    del = sum = 1.0 / (1.-_taus);
	do {
		++ap;
        del *= x / ap;
        sum += del;
    } while (del >= sum * MAX_GAMMA_ERROR);

	return sum;
}

/* **************************************** */
/* The three functions for exact simulation */
/* **************************************** */

/* 1) Update the variables V, g and Es */
/*      after an incoming spike at time t */
/*      (relative to the time of the last update tl) */
void IFSC_IncomingSpike(double t,double w, double *V, double *g, double *Es, double Ep, double Em) {
	double loc_Es = *Es;
	double gt=IFSC_OutgoingSpike(t,*g);

	#ifdef IFSC_PREC
	*V=-_taus*loc_Es*gt*rho(gt)+
		tableExpM(t-_taus*(gt-*g))*(*V+_taus*loc_Es**g*tableRho(*g));
	#else
	*V=-_taus*loc_Es*gt*rho(gt)+
		exp(-t+_taus*(gt-*g))*(*V+_taus*loc_Es**g*rho(*g));
	#endif

	if (w>0) {
		*Es=(gt*(loc_Es)+w*Ep)/(gt+w);
    	*g=gt+w;
    }
	else if (w<0) {
		*Es=(gt*(loc_Es)-w*Em)/(gt-w);
    	*g=gt-w;
    }
}

/* 2) Returns the updated value of variable g */
/*      after an outgoing spike at time t */
/*      (relative to the time of the last update tl) */
/*    The membrane potential V is not reset here, */
/*      therefore one must add the line V=Vr after calling */
/*      this function */
double IFSC_OutgoingSpike(double t, double g) {
	#ifdef IFSC_PREC
	return(g*tableExpM(t*inv_taus));
	#else
	return(g*exp(-t*inv_taus));
	#endif
}

/* 3) Calculate the time of next spike */
double IFSC_SpikeTiming(double V, double g, double Es) {
	if (IFSC_SpikeTest(V,g,Es))
	{
		/* Newton-Raphson method */
		double T=0.;
        double Tl=0.;

		while(1.-V>SPIKE_COMPUTATION_ERROR) {
			T+=(1.-V)/(-V+g*(Es-V));
			IFSC_IncomingSpike(T-Tl,0., &V, &g, &Es, 0., 0.);
            Tl=T;
		}

		return T;
	} else {
		return HUGE_VAL;
	}
}

/* The spike test (mostly for internal use) */
/*   positive => there is a spike */
/*   zero     => there is no spike */
bool IFSC_SpikeTest(double V, double g, double Es) {
	double gstar;

	/* Quick test */
	if (Es<=1.)
		return false;
	
	gstar=1./(Es-1.);
	if (g<=gstar)
		return false;

	/* Full test */
	/*     Calculate V(gstar) */
	/*		(N.B.: we should use a table for the log as well) */
	IFSC_IncomingSpike(-_taus*log(gstar/g),0., &V, &g, &Es, 0., 0.);
	if (V<=1.)
		return false;
	else
		return true;
}
