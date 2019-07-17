#include "mex.h"
#include "float.h"
#include "math.h"
#include "alignedmalloc.h"

#ifdef _WIN32
#include <windows.h>
#include <process.h>
#else
#include "pthread.h"
#endif

#define USE_SINGLES      /* Change to USE_DOUBLES to take and use doubles */
#define nMaxProcessors 4 /* Set to the max number the algorithm should use */

#ifdef USE_DOUBLES
#include "xmmintrin.h"
#define SSETYPE __m128d
#define SSEFLOAT double
#define SSEWIDTH 2
#define SSEFLT_MAX DBL_MAX
#define SSE_SETZERO _mm_setzero_pd
#define SSE_STORE _mm_storeu_pd
#define SSE_LOADU _mm_loadu_pd
#define SSE_MUL _mm_mul_pd
#define SSE_ADD _mm_add_pd
#define SSE_SUB _mm_sub_pd
#define mxSSEFUN mxIsDouble
#else
#include "emmintrin.h"
#define SSETYPE __m128
#define SSEFLOAT float
#define SSEWIDTH 4
#define SSEFLT_MAX FLT_MAX
#define SSE_SETZERO _mm_setzero_ps
#define SSE_STORE _mm_storeu_ps
#define SSE_LOADU _mm_loadu_ps
#define SSE_MUL _mm_mul_ps
#define SSE_ADD _mm_add_ps
#define SSE_SUB _mm_sub_ps
#define mxSSEFUN mxIsSingle
#endif

struct thread_data{
    SSEFLOAT *mat;
    SSEFLOAT *vec0;
    int nCol;
    int nRow;
    int inc;
    int bInd;
    SSEFLOAT bError;
    int sgn;
};

#ifdef _WIN32
unsigned __stdcall minRMSE(void *threadarg)
#else
void *minRMSE(void *threadarg)
#endif
{
    struct thread_data *td = (struct thread_data *)threadarg;
    SSEFLOAT *mat = td->mat;
    SSEFLOAT *vec0 = td->vec0;
    SSEFLOAT mp1[SSEWIDTH];
    SSEFLOAT mp2[SSEWIDTH];
    int nCol = td->nCol;
    int nRow = td->nRow;
    int inc = td->inc;
    int bInd = 0;
    SSEFLOAT bError = SSEFLT_MAX;
    int sgn = 0;
    
    SSETYPE *mvp, *mmp, mm, mtemp1, mtemp2, e1acc, e2acc;
    int nRes = nRow%SSEWIDTH;
    int n4Row = (nRow-nRes)/SSEWIDTH;

    SSEFLOAT *vec;
    SSEFLOAT temp1, temp2, err1, err2;
    int ii, jj;
    inc--; /* One increment of column occurs during the loop */
    for(ii=0; ii<nCol; ii++) {
        /* Use SSE to speed up computations (equiv. to below) */
        e1acc = SSE_SETZERO();
        e2acc = SSE_SETZERO();
        mvp = (SSETYPE*)vec0;
        /* If it's unaligned, do it the slow way */
        if(((size_t)mat) & 0xF) {
            for(jj=0; jj<n4Row; jj++) {
                mm = SSE_LOADU(mat);
                mtemp1 = SSE_SUB(*mvp, mm);
                mtemp2 = SSE_ADD(*mvp++, mm);
                e1acc = SSE_ADD(e1acc,SSE_MUL(mtemp1, mtemp1));
                e2acc = SSE_ADD(e2acc,SSE_MUL(mtemp2, mtemp2));
                mat += SSEWIDTH;
            }
        }
        else {
            mmp = (SSETYPE*)mat;
            for(jj=0; jj<n4Row; jj++) {
                mtemp1 = SSE_SUB(*mvp, *mmp);
                mtemp2 = SSE_ADD(*mvp++, *mmp++);
                e1acc = SSE_ADD(e1acc,SSE_MUL(mtemp1, mtemp1));
                e2acc = SSE_ADD(e2acc,SSE_MUL(mtemp2, mtemp2));
            }
            mat += SSEWIDTH*n4Row;
        }
        vec = vec0 + SSEWIDTH*n4Row;
        SSE_STORE(mp1, e1acc);
        SSE_STORE(mp2, e2acc);
        err1 = 0;
        err2 = 0;
        for (jj=0; jj<SSEWIDTH; jj++) {
	        err1 += mp1[jj];
	        err2 += mp2[jj];
        }

        /* Take care of remaining elements */
        for(jj=0; jj<nRes; jj++) {
            temp1 = (*vec - *mat);
            temp2 = (*vec++ + *mat++);
            err1 += temp1*temp1;
            err2 += temp2*temp2;
        }

        if(err1 < bError) {
            bError = err1;
            bInd = ii;
            sgn = 1;
        }
        if(err2 < bError) {
            bError = err2;
            bInd = ii;
            sgn = -1;
        }
        mat += nRow*inc;
    }
    td->bError = bError;
    td->bInd = bInd;
    td->sgn = sgn;
    #ifdef _WIN32
    _endthreadex( 0 );
    #else
    pthread_exit(NULL);
    #endif
}

struct thread_data myDatas[nMaxProcessors];

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
    SSEFLOAT *Se, *rmSh;
    SSEFLOAT bError = SSEFLT_MAX;
    mwSize tr, tc;
    int ii;
    int bInd = 0;
    int sgn = 0;
    struct thread_data *td;
    bool needToFree = false;
    int nProcessors = 1; /* Default to single thread */
    mxArray *trhs[1];
    
    /* Deal with thread junk */
    #ifdef _WIN32
    HANDLE thread[nMaxProcessors];
    #else
    pthread_t thread[nMaxProcessors];
    int rc;
    void *status;
    pthread_attr_t attr;
    
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    #endif
    
    /* Check for proper number of arguments. */
    if(nrhs != 2 && nrhs != 3) {
        mexErrMsgTxt("Exactly two or three outputs required.");
    } else if(nlhs!=2) {
        mexErrMsgTxt("Exactly two outputs required.");
    }

    /* The inputs must be noncomplex SSEFLOAT row vectors. */
    tr = mxGetM(prhs[0]);
    tc = mxGetN(prhs[0]);
    if( (nrhs != 2 && nrhs != 3) || !mxSSEFUN(prhs[0]) || !mxSSEFUN(prhs[1]) || mxIsComplex(prhs[1]) || mxGetM(prhs[1]) != mxGetM(prhs[0]) ) {
        mexErrMsgTxt("Incorrect inputs!\nN.B. Both inputs must be floats, try passing single(var)");
    }
    if( nrhs > 2 ) {
        nProcessors = (int)(mxGetScalar(prhs[2])+0.5);
        if(nProcessors > nMaxProcessors)
            nProcessors = nMaxProcessors;
        else if(nProcessors < 1)
            nProcessors = 1;
    }

    /* Set max lengths */
    Se = (SSEFLOAT*)mxGetData(prhs[0]);
    rmSh = (SSEFLOAT*)mxGetData(prhs[1]);
    plhs[0] = mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS,mxREAL);
    plhs[1] = mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS,mxREAL);
    
    /* Make sure rmSh is aligned for SSE instructions */
    if(((long int)rmSh) & 0xF) {
        needToFree = true;
        rmSh = (SSEFLOAT*)AlignedMalloc(tr*sizeof(SSEFLOAT), 16);
        memcpy((void*)rmSh,(void*)mxGetData(prhs[1]),tr*sizeof(SSEFLOAT));
    }

    /* Partition data into separable threads */
    for(ii=0; ii<nProcessors; ii++) {
        td = &(myDatas[ii]);
        td->mat = Se+(ii*tr);
        td->vec0 = rmSh;
        td->nCol = (tc-(ii+1))/nProcessors+1;
        td->nRow = tr;
        td->inc = nProcessors;
        #ifdef _WIN32
        thread[ii] = (HANDLE)_beginthreadex( NULL, 0, &minRMSE, &myDatas[ii], 0, NULL );
        #else
        rc = pthread_create(&thread[ii], &attr, minRMSE, (void *)&myDatas[ii]);
        if (rc) {
            mexErrMsgTxt("ERROR in pthread_create()\n");
        }
        #endif
    }
    #ifndef _WIN32
    pthread_attr_destroy(&attr);
    #endif

    /* Wait for threads to complete and check against best values*/
    bInd = bInd*nProcessors;
    for(ii=0; ii<nProcessors; ii++) {
        td = &(myDatas[ii]);
        #ifdef _WIN32
        WaitForSingleObject(thread[ii], INFINITE);
        CloseHandle(thread[ii]);
        #else
        rc = pthread_join(thread[ii], &status);
        if (rc)
            mexErrMsgTxt("ERROR return code from pthread_join()\n");
        #endif
        if(td->bError < bError) {
            bError = td->bError;
            sgn = td->sgn;
            bInd = td->bInd*nProcessors+ii;
        }

    }

    *((double*)mxGetData(plhs[0])) = bInd+1;
    *((double*)mxGetData(plhs[1])) = sgn;
    if(needToFree)
        AlignedFree(rmSh);
}
