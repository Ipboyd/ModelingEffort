#include "mex.h"
#include "math.h"
#include "float.h"

extern double _taus;
void IFSC_Init(double taus, double dx); /* Set constants */
void IFSC_IncomingSpike(double t, double w, double *V, double *g, double *Es, double Ep, double Em); /* 1) Update {V,g,Es} after incoming spike @ time t */
double IFSC_OutgoingSpike(double t, double g); /* 2) Return updated value of variable g after a spike @ t */
double IFSC_SpikeTiming(double V, double g, double Es); /* 3) Calculate the time of next spike */
bool IFSC_SpikeTest(double V, double g, double Es); /* The spike test (mostly for internal use) */
double SpikeTimingARP(double V, double g, double Es, double Ese, double Esi, double arp, double dtLastOutput);
double *expLookupTable, *rhoLookupTable, *logLookupTable;
int expLookup_nmax, rhoLookup_nmax, logLookup_nmax;

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] ) {
    double *inTimes, *weights, *outMat, scaler, taus, taum;
    double arp, dur, Ese, Esi, Vth, Vre, El, V, g, Es, Tl;
    double nextIn, lastOut, nextOut, spikingTime, prevInTime;
    int ti, count, Nspikes;
    mwSize maxOut;

    /* Check for proper number of arguments. */
    if(nrhs<14) {
        mexErrMsgTxt("14, 15, or 16 inputs required.");
    }
    if(nrhs>16) {
        mexErrMsgTxt("14, 15, or 16 inputs required.");
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
            !mxIsDouble(prhs[9]) || mxIsComplex(prhs[9]) || mxIsEmpty(prhs[9]) ||
            !mxIsDouble(prhs[10]) || mxIsComplex(prhs[10]) || mxIsEmpty(prhs[10]) ||
            !mxIsDouble(prhs[11]) || mxIsComplex(prhs[11]) || mxIsEmpty(prhs[11]) ||
            !mxIsDouble(prhs[12]) || mxIsComplex(prhs[12]) || mxIsEmpty(prhs[12]) ||
            !mxIsDouble(prhs[13]) || mxIsComplex(prhs[13]) || mxIsEmpty(prhs[13])
            ) {
        mexErrMsgTxt("Inputs must be nonempty real doubles!");
    }
    
    /* If input is empty, output is empty */
    if(mxIsEmpty(prhs[0]) || mxIsEmpty(prhs[1])) {
        plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
        return;
    }

    /* Spikes, Weights, arp, Ese, Esi, taum, taus, Vre, Vth, El (i.e. Vss), expLookup, rhoLookup, dx, [dur], [maxOut] */
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

    if(nrhs >= 15) {
        if(!mxIsDouble(prhs[14])|| mxIsComplex(prhs[14]) || mxIsEmpty(prhs[14]))
            mexErrMsgTxt("dur must be a nonempty real double!");
        else
            dur = mxGetScalar(prhs[14]);
    }
    else {
        if(Nspikes)
            dur = inTimes[Nspikes-1]+5*(taum+taus*taum); /* Go 10 time constants beyond last input */
        else
            dur = 0;
    }
    if(nrhs >= 16) {
        if(!mxIsDouble(prhs[15])|| mxIsComplex(prhs[15]) || mxIsEmpty(prhs[15]))
            mexErrMsgTxt("maxOut must be a nonempty real double!");
        else
            maxOut = mxGetScalar(prhs[15]);
    }
    else {
        maxOut = (int)(1000*dur); /* Save room for 1000 Hz firing rate response */
    }
    plhs[0] = mxCreateDoubleMatrix(maxOut, 1, mxREAL);
    outMat = mxGetPr(plhs[0]);
    IFSC_Init(taus,mxGetScalar(prhs[13]));
    
    expLookupTable = mxGetPr(prhs[10]);
    logLookupTable = mxGetPr(prhs[11]);
    rhoLookupTable = mxGetPr(prhs[12]);
    if(mxGetNumberOfElements(prhs[10]) != expLookup_nmax+1)
        mexErrMsgTxt("expTable must be the correct length!");
    if(mxGetNumberOfElements(prhs[11]) != logLookup_nmax+1)
        mexErrMsgTxt("logTable must be the correct length!");
    if(mxGetNumberOfElements(prhs[12]) != rhoLookup_nmax+1)
        mexErrMsgTxt("rhoTable must be the correct length!");

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

#define SPIKE_COMPUTATION_ERROR	1e-10
#define MAX_GAMMA_ERROR 1e-15
double rho(double g);
double _taus, expLookup_dx, logLookup_dx, rhoLookup_dg;
double inv_taus, inv_expLookup_dx, inv_logLookup_dx, inv_rhoLookup_dg;

/* tableExpM(x) returns exp(-x) from the look-up table */
double tableExpM(double x) {
	double a, *table;
	int n=(int)(x=x/expLookup_dx);

	if ((n>=expLookup_nmax) || (n<0))
		return (exp(-x*expLookup_dx));

	table=expLookupTable+n;
	a=*(table++);
	return ( a + (x-n)*(*table - a) );
}

double tableLog(double x) {
	double a, *table;
	int n=(int)(x=x*inv_logLookup_dx);

	if ((n>=logLookup_nmax) || (n<0))
		return (exp(x*logLookup_dx));

	table=logLookupTable+n;
	a=*(table++);
	return ( a + (x-n)*(*table - a) );
}

/* tableRhoTausG(g) returns _taus*g*rho(g) from the look-up table */
double tableRhoTausG(double g) {
	double a, *table;
	int n=(int)(g=g*inv_rhoLookup_dg);

	if ((n>=rhoLookup_nmax) || (n<0))
		return (_taus*g*rhoLookup_dg*rho(g*rhoLookup_dg));

	table=rhoLookupTable+n;
	a=*(table++);
	return ( a + (g-n)*(*table - a) );
}

/* ************************** */
/* CONSTRUCTION & DESTRUCTION */
/* ************************** */

/* Set the constants */
void IFSC_Init(double taus, double dx){
	_taus=taus;
	inv_taus=1./_taus;

    expLookup_dx=dx;
    inv_expLookup_dx=1./dx;
    expLookup_nmax=(int)(2*inv_expLookup_dx)-1;

    logLookup_dx=dx;
    inv_logLookup_dx=1./dx;
    logLookup_nmax=(int)(2*inv_logLookup_dx)-1;

    rhoLookup_dg=dx;
    inv_rhoLookup_dg=1./dx;
    rhoLookup_nmax=(int)(2*inv_rhoLookup_dg)-1;
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

	*V=-loc_Es*tableRhoTausG(gt)+
		tableExpM(t-_taus*(gt-*g))*(*V+loc_Es*tableRhoTausG(*g));

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
	return(g*tableExpM(t*inv_taus));
}

/* 3) Calculate the time of next spike */
double IFSC_SpikeTiming(double V, double g, double Es) {
	if (IFSC_SpikeTest(V,g,Es))
	{
		/* Newton-Raphson method */
		double T=0.;
        double Tl=0.;
        int count=0;

		while(1.-V>SPIKE_COMPUTATION_ERROR && count++<1e6) {
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

	/* Full test: Calculate V(gstar) */
	IFSC_IncomingSpike(-_taus*tableLog(gstar/g),0., &V, &g, &Es, 0., 0.);
	if (V<=1.)
		return false;
	else
		return true;
}
