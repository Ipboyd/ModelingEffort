#include "mex.h"
#include "string.h" /*needed for memset declaration*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	double prevSorted, nextSorted, newVal, *sorted, *x, *y, *tempSort;
	mwIndex xInd, changeInd, colInd, prevSortInds, nextSortInds, dimInd, *sortInds;
	mwSize winLen, len, nCol, nDimensions;
	const mwSize *dimensions;
    mxArray *tlhs[2], *trhs[1];
    
    /* Check for proper number of arguments. */
    if(nrhs < 2) {
        mexErrMsgTxt("Two inputs required.");
    } else if(nlhs > 1) {
        mexErrMsgTxt("Too many output arguments");
    }
    
    /* The inputs must be noncomplex double row vectors. */
	x = mxGetPr(prhs[0]);
	winLen = mxGetScalar(prhs[1]);
    len = mxGetM(prhs[0]);
	if(winLen > len) {
		winLen = len;
	}

    nCol = mxGetN(prhs[0]);
    if(nrhs < 2 || !mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) || !mxIsScalar(prhs[1])) {
        mexErrMsgTxt("Incorrect inputs!\nN.B. Both inputs must be doubles, try passing double(var)");
    }
    	
	/* Special case for row vector */
	nDimensions = mxGetNumberOfDimensions(prhs[0]);
	dimensions = (mwSize *) mxMalloc(nDimensions*sizeof(mwSize));
 	dimensions = mxGetDimensions(prhs[0]);
	if(len == 1) {
		len = nCol;
		nCol = 1;
		plhs[0] = mxCreateDoubleMatrix(1, len, mxREAL);
	} else {
		/*plhs[0] = mxCreateDoubleMatrix(len, 1, mxREAL);*/
		plhs[0] = (mxArray *) mxCreateNumericArray(nDimensions, dimensions, mxDOUBLE_CLASS, mxREAL);
		nCol = 1;
		for(dimInd = 1; dimInd < nDimensions; dimInd++) { /*yes, this is supposed to start at the second element of dimensions*/
			nCol *= dimensions[dimInd];
		}
	}
	
	if(len > 0) {
		y = mxGetPr(plhs[0]);
		sorted = (double *) mxMalloc(winLen*sizeof(double));
		sortInds = (mwIndex *) mxMalloc(winLen*sizeof(mwIndex));
		for(changeInd = 0; changeInd < winLen; changeInd++) {
			sortInds[changeInd] = changeInd;
		}
		
		/* Now find the medians */
		for(colInd = 0; colInd < nCol; colInd++) {
			if(colInd) {
				x += len;
			}
			memset(sorted, 0, winLen*sizeof(double));
			for(xInd = 0; xInd < len + (winLen - 1)/2; xInd++) {/*for xInd = 1:len + winLen/2 - 1*/
				if(xInd < len) {/*if xInd <= len*/
					newVal = x[xInd];/*newVal = x(xInd);*/
				} else {
					newVal = 0;
				}
				
				changeInd = 0;/*changeInd = 1;*/
				while(changeInd < winLen - 1 && newVal > sorted[changeInd] && sortInds[changeInd] != 0) {/*while changeInd < winLen && newVal > sorted(changeInd) && sortInds(changeInd) ~= 1*/
					sortInds[changeInd++]--;/*sortInds(changeInd) = sortInds(changeInd) - 1;changeInd = changeInd + 1;*/
				}
				if(sortInds[changeInd] == 0) { /*if sortInds(changeInd) == 1 /* popInd < putInd*/
					while(changeInd < winLen - 1 && newVal > sorted[changeInd + 1]) {/*while changeInd < winLen && newVal > sorted(changeInd + 1)*/
						sorted[changeInd] = sorted[changeInd + 1];/*sorted(changeInd) = sorted(changeInd + 1);*/
						sortInds[changeInd] = sortInds[changeInd + 1] - 1;
						changeInd++;
						/*sortInds(changeInd) = sortInds(changeInd + 1) - 1;changeInd = changeInd + 1;*/
					}
					sorted[changeInd] = newVal;/*sorted(changeInd) = newVal;*/
					sortInds[changeInd] = winLen - 1;/*sortInds(changeInd) = winLen;*/
				} else { /* putInd < popInd*/
					prevSorted = sorted[changeInd];/*prevSorted = sorted(changeInd);*/
					prevSortInds = sortInds[changeInd];/*prevSortInds = sortInds(changeInd);*/
					sorted[changeInd] = newVal;/*sorted(changeInd) = newVal;*/
					sortInds[changeInd++] = winLen - 1;/*sortInds(changeInd) = winLen;changeInd = changeInd + 1;*/
					if(changeInd < winLen) {
						while(changeInd < winLen - 1 && sortInds[changeInd] != 0) {/*while changeInd < winLen && sortInds(changeInd) ~= 1*/
							nextSorted = sorted[changeInd];/*nextSorted = sorted(changeInd);*/
							nextSortInds = sortInds[changeInd];/*nextSortInds = sortInds(changeInd);*/
							sorted[changeInd] = prevSorted;/*sorted(changeInd) = prevSorted;*/
							sortInds[changeInd++] = prevSortInds - 1;/*sortInds(changeInd) = prevSortInds - 1;changeInd = changeInd + 1;*/
							prevSorted = nextSorted;
							prevSortInds = nextSortInds;
						}
						sorted[changeInd] = prevSorted;/*sorted(changeInd) = prevSorted;*/
						sortInds[changeInd] = prevSortInds - 1;/*sortInds(changeInd) = prevSortInds - 1;*/
					}
				}
				
				for(changeInd++; changeInd < winLen; changeInd++) {/*for changeInd = changeInd + 1:winLen*/
					sortInds[changeInd]--;/*sortInds(changeInd) = sortInds(changeInd) - 1;*/
				}
				if(xInd >= (winLen + 1)/2 - 1) {/*if xInd >= winLen/2*/
					if(winLen%2) {/*if mod(winLen, 2)*/
						*y++ = sorted[(winLen - 1)/2];/*med(xInd - floor(winLen/2)) = sorted((winLen + 1)/2);*/
					} else {
						*y++ = (sorted[winLen/2 - 1] + sorted[winLen/2])/2;/*med(xInd - winLen/2 + 1) = (sorted(winLen/2) + sorted(winLen/2 + 1))/2;*/
					}
				}
			}
		}
		mxFree(sorted);
		mxFree(sortInds);
	}
	/*mxFree(dimensions);*/
}