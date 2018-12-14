#include "mex.h"
#include "matrix.h"
#include "math.h"
#define SQR(x) (x)*(x)

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{    
    
    double *X, *Y, *designMatrix;
    double temp,theta;
	int nx, ny, dim,i,j,k;
    
	/* Input variable*/
    X = mxGetPr(prhs[0]);        /* randperm position of the array */
    Y = mxGetPr(prhs[1]);        /* randperm position of the array */
    theta = mxGetScalar(prhs[2]);        /* randperm position of the array */

    nx = mxGetM(prhs[0]);
    ny = mxGetM(prhs[1]);
    dim = mxGetN(prhs[0]);
    
    /* Output Variable*/
	plhs[0] = mxCreateDoubleMatrix(nx,ny,mxREAL);
	designMatrix = mxGetPr(plhs[0]);
		
    /* Design Matrix H*/
	for(i=0;i<nx;i++)
	{
		for(j=0;j<ny;j++)
		{
			temp = 0;
			for(k=0;k<dim;k++)
			{
				temp += SQR(X[i+ k*nx] - Y[k*ny + j]);
			}
            designMatrix[i + j*nx] = exp(-temp*theta*theta);
		}
	}
}

