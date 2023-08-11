#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <cstdio>
#include <ctime>
#include <time.h>
#include <stdlib.h>
#include <complex>
#include <vector>
#include <string>
#include <algorithm>
#include <string>
#include <gsl/gsl_sf_legendre.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_trig.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_math.h>
#include <fftw3.h>
using namespace std;
typedef complex<double> dcomplex;
#define PI 3.14159265358979323846264338327950288419716939937510
#define rc reinterpret_cast<fftw_complex*>
#define im dcomplex(0,1)
void Smear(double* p, double sigma, int N);

//parameters
double os = 2; // oversampling
const double beta = .5;
const int Ncycle = 20;	// total number of cycles
const int NHIO = 100;	// number of HIO per cycle
const int NER = 100;	// number of ER per cycle
double tol = .1;		// Shrinkwrap tolerance
double newtontol = 1e-14; // Newton bisection tolerance
int maxiter = 100;		 // Newton bisection max iterations
double sigdata = 100000 * 16 / os / os / os / os; // noise std dev added to data
double sigPN = 100000 * 16 / os / os / os / os; // noise std dev expected by constraint


// Helper Functions
double Sqr(double x) { return x * x; }
double Cub(double x) { return x * x * x; }

void vcopy(double* x, double* y, int N)
{
	for (int i = 0; i < N; ++i) y[i] = x[i];
}

int min(int i, int j)
{
	if (i < j)
		return i;

	return j;
}

// store 2D FFT of X in Xhat
// X should be K * K
void FFT2D(double* X, dcomplex* Xhat, int K, int N)
{
	int Karr[2];
	Karr[0] = K, Karr[1] = K;

	fftw_plan pfw = fftw_plan_many_dft_r2c(2, Karr, N, X, NULL, 1, K * K, rc(Xhat), NULL, 1, K * (K / 2 + 1), FFTW_ESTIMATE);
	fftw_execute(pfw);
	fftw_destroy_plan(pfw);

	for (int i = 0; i < N * K * (K / 2 + 1); ++i) Xhat[i] /= K * K;
}


// store 2D IFFT of Xhat in X
// Destroys input array!!!!!!!!!!!!
void IFFT2D(dcomplex* Xhat, double* X, int K, int N)
{
	int Karr[2];
	Karr[0] = K, Karr[1] = K;

	fftw_plan pbw = fftw_plan_many_dft_c2r(2, Karr, N, rc(Xhat), NULL, 1, K * (K / 2 + 1), X, NULL, 1, K * K, FFTW_ESTIMATE);
	fftw_execute(pbw);
	fftw_destroy_plan(pbw);
}

// plot density p to 2Dp(num).vtr, for given (num)
void Plot2D(double* p, int N, int num)
{
	stringstream str(stringstream::out);
	double cmin = -1;
	double cmax = 1;


	str << "<VTKFile type =\"RectilinearGrid\">" << '\n';
	str << "<RectilinearGrid WholeExtent=\"" << 1 << " " << N << " " << 1 << " " << N << " " << 1 << " " << 1 << "\">" << '\n';
	str << "<Piece Extent=\"" << 1 << " " << N << " " << 1 << " " << N << " " << 1 << " " << 1 << "\">" << '\n';
	str << "<PointData>" << '\n';
	str << "<DataArray type=\"Float64\" Name=\"p" << "\" NumberOfComponents=\"1\" format=\"ascii\">" << '\n';
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			str << p[(i + 0 * N / 2) % N + ((j + 0 * N / 2) % N) * N] << endl;
		}
	}
	str << "</DataArray>" << '\n';
	str << "<DataArray type=\"Float64\" Name=\"Laplacian" << "\" NumberOfComponents=\"1\" format=\"ascii\">" << '\n';
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			double lap = 0;
			int ii = (i + N / 2) % N;
			int jj = (j + N / 2) % N;
			lap = p[(ii - 1 + N) % N + jj * N] - 4 * p[ii + jj * N] + p[(ii + 1 + N) % N + jj * N]
				+ p[ii + ((jj - 1 + N) % N) * N] + p[ii + ((jj + 1 + N) % N) * N];
			str << lap << endl;
		}
	}
	str << "</DataArray>" << '\n';

	str << "</PointData>" << '\n';
	str << "<CellData>" << '\n';
	str << "</CellData>" << '\n';
	str << "<Coordinates>" << '\n';
	str << "<DataArray type=\"Float64\" Name=\"X\" NumberOfComponents=\"1\" format=\"ascii\">" << '\n';
	for (int i = 0; i < N; ++i)
	{
		str << cmin + i * ((cmax - cmin) / (1.0 * N - 1)) << endl;
	}
	str << "</DataArray>" << '\n';
	str << "<DataArray type=\"Float64\" Name=\"Y\" NumberOfComponents=\"1\" format=\"ascii\">" << '\n';
	for (int i = 0; i < N; ++i)
	{
		str << cmin + i * ((cmax - cmin) / (1.0 * N - 1)) << endl;
	}
	str << "</DataArray>" << '\n';
	str << "<DataArray type=\"Float64\" Name=\"Z\" NumberOfComponents=\"1\" format=\"ascii\">" << '\n';

	str << 0 << endl;

	str << "</DataArray>" << '\n';
	str << "</Coordinates>" << '\n';
	str << "</Piece>" << '\n';
	str << "</RectilinearGrid>" << '\n';
	str << "</VTKFile>" << '\n';


	ofstream myfile;
	char file[255];
	char snum[255];
	strcpy(file, "2Dp");
	sprintf(snum, "%d", num);
	strcat(file, snum);
	strcat(file, ".vtr");
	myfile.open(file);
	myfile.write(str.str().c_str(), str.str().length());
	myfile.close();
}

// plot intensity a to Amplitudes(num).vtr, for given (num)
void Plota(double* a, int N, int num)
{
	stringstream str(stringstream::out);
	double cmin = -1;
	double cmax = 1;
	int N2 = N / 2 + 1;

	str << "<VTKFile type =\"RectilinearGrid\">" << '\n';
	str << "<RectilinearGrid WholeExtent=\"" << 1 << " " << N << " " << 1 << " " << N2 << " " << 1 << " " << 1 << "\">" << '\n';
	str << "<Piece Extent=\"" << 1 << " " << N << " " << 1 << " " << N2 << " " << 1 << " " << 1 << "\">" << '\n';
	str << "<PointData>" << '\n';
	str << "<DataArray type=\"Float64\" Name=\"a" << "\" NumberOfComponents=\"1\" format=\"ascii\">" << '\n';
	for (int j = 0; j < N2; ++j)
	{
		for (int i = 0; i < N; ++i)
		{
			str << a[j + ((N + i - N / 2) % N) * N2] << endl;
		}
	}
	str << "</DataArray>" << '\n';
	str << "</PointData>" << '\n';
	str << "<CellData>" << '\n';
	str << "</CellData>" << '\n';
	str << "<Coordinates>" << '\n';
	str << "<DataArray type=\"Float64\" Name=\"X\" NumberOfComponents=\"1\" format=\"ascii\">" << '\n';
	for (int i = 0; i < N; ++i)
	{
		str << cmin + i * ((cmax - cmin) / (1.0 * N - 1)) << endl;
	}
	str << "</DataArray>" << '\n';
	str << "<DataArray type=\"Float64\" Name=\"Y\" NumberOfComponents=\"1\" format=\"ascii\">" << '\n';
	for (int i = 0; i < N; ++i)
	{
		str << cmin + i * ((cmax - cmin) / (1.0 * N - 1)) << endl;
	}
	str << "</DataArray>" << '\n';
	str << "<DataArray type=\"Float64\" Name=\"Z\" NumberOfComponents=\"1\" format=\"ascii\">" << '\n';

	str << 0 << endl;

	str << "</DataArray>" << '\n';
	str << "</Coordinates>" << '\n';
	str << "</Piece>" << '\n';
	str << "</RectilinearGrid>" << '\n';
	str << "</VTKFile>" << '\n';


	ofstream myfile;
	char file[255];
	char snum[255];
	strcpy(file, "Amplitudes");
	sprintf(snum, "%d", num);
	strcat(file, snum);
	strcat(file, ".vtr");
	myfile.open(file);
	myfile.write(str.str().c_str(), str.str().length());
	myfile.close();
}


// create face image and store to p
// create corresponding square support and store to S
// p is N * N, S is N * N
void Makep(double* p, int N, int* S)
{
	double x, y;
	//	double sigma = .2;
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			x = i, y = j;
			x /= N, y /= N;
			x -= .5, y -= .5;
			x *= os, y *= os;


			if (max(fabs(x), fabs(y)) < .41) S[j + i * N] = 1;

			if (Sqr(x) + Sqr(y) < Sqr(.4) && (Sqr(x - .15) + Sqr(y - .15) > Sqr(.1)) && Sqr(x + .15) + Sqr(y - .15) > Sqr(.1) && Sqr(x) + 12 * Sqr(y + .15) > Sqr(.3))
			{

				p[j + i * N] = 1e6;
			}
			else
			{
				p[j + i * N] = 0;
			}
		}
	}

	Smear(p, 1.0 / N, N);
	for (int i = 0; i < N * N; ++i) p[i] *= S[i];
}


// magnitude projector
// stores projection of p in PMp, where mag is intensity data
void PM(double* p, double* PMp, double* mag, int N)
{
	int N2 = N / 2 + 1;

	vector<dcomplex> phat(N * N2);

	FFT2D(p, phat.data(), N, 1);

	for (int i = 0; i < N * N2; ++i)
	{
		if (mag[i] > 0)
		{
			phat[i] = sqrt(mag[i]) * exp(im * arg(phat[i]));
		}
	}
	IFFT2D(phat.data(), PMp, N, 1);
}

// magnitude projector, operating in Fourier space
void PMhat(dcomplex* phat, dcomplex* PMphat, double* mag, int D)
{
	for (int i = 0; i < D; ++i)
	{
		if (mag[i] > 0)
		{
			PMphat[i] = sqrt(mag[i]) * exp(im * arg(phat[i]));
		}
		else
		{
			PMphat[i] = phat[i];
		}
	}
}

// noise projector, assuming constant Gaussian noise
// given intensity I and data mag, stores projection in PNI
void PN(double* I, double* PNI, double* mag, int N)
{
	int N2 = N / 2 + 1;

	double Ierr = 0.0;
	for (int i = 0; i < N * N2; ++i)
	{
		Ierr += Sqr(I[i] - mag[i]);
	}

	double lambda = 1 - sqrt(Ierr / N / N2 / Sqr(sigPN));

	// if lambda is negative, PNI is on the constraint, so project
	if (lambda < 0)
	{
		for (int i = 0; i < N * N2; ++i)
		{
			PNI[i] = (I[i] - lambda * mag[i]) / (1 - lambda);
		}
	}
	// otherwise the constraint is already satisfied, so PNI = I
	else
	{
		for (int i = 0; i < N * N2; ++i)
		{
			PNI[i] = I[i];
		}
	}
}

// Bisection Bounded Newton Iteration for variable noise projector

// Function Evaluation -- f(-lambda)
void ComputeLF(double& F, double neglambda, double* I, double* Id, double* sigma, int N)
{
	int N2 = N / 2 + 1;
	F = 0;

	for (int i = 0; i < N * N2; ++i)
	{
		F += Sqr(I[i] - Id[i]) / Sqr(1 + neglambda / Sqr(sigma[i])) / Sqr(sigma[i]);
	}
	F = F / N / N2 - 1;
}

// Derivative Evaluation -- f'(-lambda)
void ComputeLFp(double& Fp, double neglambda, double* I, double* Id, double* sigma, int N)
{
	int N2 = N / 2 + 1;
	Fp = 0;

	for (int i = 0; i < N * N2; ++i)
	{
		Fp += - Sqr(sigma[i]) * Sqr(I[i] - Id[i]) / Cub(Sqr(sigma[i]) + neglambda);
	}
	Fp = 2 * Fp / N / N2;
}

// Bisection Bounded Newton to solve f(-lambda) = 0
void Tikhonov(double* sigma, double* I, double* Id, int N, double& lambda)
{
	double F, Fp;
	double lb = 0, ub = 1e100;			// initial bounds
	double neglambda = - lambda;

	double neglambdaold = 0;
	int iter = 0;

	// for debugging
	/*
	double flambda = 0;
	for (int i = -100; i < 101; ++i)
	{
		ComputeLF(flambda, -pow(10,i), I, Id, sigma, N);
		cout << "lambda = 1e" << i << "		flambda = " << flambda << endl;
	}
	*/

	while (iter < maxiter)
	{
		ComputeLF(F, neglambda, I, Id, sigma, N); // Function Evaluation
		ComputeLFp(Fp, neglambda, I, Id, sigma, N);	// Derivative Evaluation


		if (fabs(F) < newtontol) break;
		// Newton Step if within bounds and update step doesn't blow up
		if (neglambda - F / Fp > lb && neglambda - F / Fp < ub && fabs(F / Fp) < 10 * neglambda)
		{
			neglambda -= F / Fp;
			//cout << "Newton step: -lambda = " << neglambda << endl;
		}
		else // Bisection Step otherwise
		{
			if (F > 0 || Fp < 0)
			{
				if (lb > 0)
				{
					ub = neglambda;
					neglambda = (lb + neglambda) / 2;
				}
				else
				{
					neglambda /= 10;
				}
			}
			else
			{
				lb = neglambda;
				if (ub < 1e100)
				{
					neglambda = (ub + neglambda) / 2;
				}
				else
				{
					neglambda *= 10;
				}
			}
			//cout << "Bisection step: -lambda = " << neglambda << endl;
		}
		if (neglambda > 1e100 || neglambda < 1e-100)
		{
			cout << "bad lambda" << " " << neglambda << endl;
			break;
		}
		iter++;
		neglambdaold = neglambda;
	}

	lambda = - neglambda;
}

// noise projector for variable noise
void PNvarnoise(double* I, double* PNI, double* mag, double* sigPNs, int N)
{
	int N2 = N / 2 + 1;
	double lambda = -1e90; // initialization
	// solve for Lagrange multiplier using Newton
	Tikhonov(sigPNs, I, mag, N, lambda);
	//cout << "Lambda:     " << lambda << endl;
	
	// if lambda is negative, PNI is on the constraint, so project
	if (lambda < 0)
	{
		for (int i = 0; i < N * N2; ++i)
		{
			PNI[i] = (I[i] - lambda * mag[i] / Sqr(sigPNs[i])) / (1 - lambda / Sqr(sigPNs[i]));
		}
	}
	// otherwise the constraint is already satisfied, so PNI = I
	else
	{
		for (int i = 0; i < N * N2; ++i)
		{
			PNI[i] = I[i];
		}
	}
}

// intensity filtering for HIO
void G(double* I, double* GI, double* p, int* S, int N)
{
	int N2 = N / 2 + 1;

	// apply magnitude projector
	vector<double> PMp(N * N);
	//vcopy(p, PMp.data(), N);
	PM(p, PMp.data(), I, N);

	// apply support projector
	for (int i = 0; i < N * N; ++i)
	{
		if (S[i] == 0)
		{
			PMp[i] = 0.0;
		}
	}

	// compute new intensity
	vector<dcomplex> PMphat(N * N2);
	FFT2D(PMp.data(), PMphat.data(), N, 1);
	for (int i = 0; i < N * N2; ++i) GI[i] = Sqr(norm(PMphat[i]));
}


void HIO(double* p, double* mag, int* S, double* sigPNs, int N)
{
	int N2 = N / 2 + 1;
	// compute intensity of p
	vector<dcomplex> phat(N * N2);
	FFT2D(p, phat.data(), N, 1);
	vector<double> I(N * N2);
	for (int i = 0; i < N * N2; ++i) I[i] = Sqr(norm(phat[i]));

	// intensity filtering
	vector <double> GI(N * N2);
	G(I.data(), GI.data(), p, S, N);
	//vcopy(I.data(), GI.data(), N * N2);			// no intensity filtering

	// apply noise projector
	vector<double> PNIp(N * N2);
	//PNvarnoise(GI.data(), PNIp.data(), mag, sigPNs, N);
    PN(GI.data(), PNIp.data(), mag, N);			// constant noise projector
	//vcopy(GI.data(), PNIp.data(), N);				// no noise projector
	
	// apply magnitude projector
	vector<double> PMp(N * N);
	PM(p, PMp.data(), PNIp.data(), N);

	for (int i = 0; i < N * N; ++i)
	{
		if (S[i])
		{
			p[i] = PMp[i];
		}
		else
		{
			if (S[i])
			{
				p[i] = p[i] - beta * (PMp[i] - min(max(PMp[i], 0.0), 1.0));
			}
			else
			{
				p[i] = p[i] - beta * PMp[i];
			}
		}
	}

}


void ER(double* p, double* mag, int* S, double* sigPNs, int N)
{
	int N2 = N / 2 + 1;
	// compute intensity of p
	vector<dcomplex> phat(N * N2);
	FFT2D(p, phat.data(), N, 1);
	vector<double> I(N * N2);
	for (int i = 0; i < N * N2; ++i) I[i] = Sqr(norm(phat[i]));

	// intensity filtering
	vector <double> GI(N * N2);
	G(I.data(), GI.data(), p, S, N);
	//vcopy(I.data(), GI.data(), N * N2);			// no intensity filtering

	// apply noise projector
	vector<double> PNIp(N * N2);
	//PNvarnoise(GI.data(), PNIp.data(), mag, sigPNs, N);
	PN(GI.data(), PNIp.data(), mag, N);			// constant noise projector
	//vcopy(GI.data(), PNIp.data(), N);				// no noise projector

	// apply magnitude projector
	vector<double> PMp(N * N);
	PM(p, PMp.data(), PNIp.data(), N);

	// apply support projector
	for (int i = 0; i < N * N; ++i)
	{
		if (S[i])
		{
			p[i] = PMp[i];
		}
		else
		{
			p[i] = 0;
		}
	}

}

void Smear(double* p, double sigma, int N)
{
	double x, y;

	vector<dcomplex> phat(N * (N / 2 + 1));

	FFT2D(p, phat.data(), N, 1);

	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N / 2 + 1; ++j)
		{
			x = min(i, N - i);
			y = j;
			phat[j + i * (N / 2 + 1)] *= exp(-(x * x + y * y) * PI * PI * sigma * sigma * 2);
		}
	}

	IFFT2D(phat.data(), p, N, 1);
}

void Shrinkwrap(double* p, int* S, int N, int& counter)
{
	vector<double> psmear(N * N);
	vcopy(p, psmear.data(), N * N);

	Smear(psmear.data(), 1.0 / N, N);

	double maxp = -1e100;
	for (int i = 0; i < N * N; ++i)
	{
		if (psmear[i] > maxp)
		{
			maxp = psmear[i];
		}
	}

	for (int i = 0; i < N * N; ++i)
	{
		if (psmear[i] < tol*maxp)
		{
			S[i] = 0;
			p[i] = 0;
		}
		else
		{
			S[i] = 1;
		}
	}

}


void MakeData(double* a, int* S, int N)
{
	int N2 = N / 2 + 1;

	vector<double> p(N * N);
	vector<dcomplex> phat(N * N2);
	Makep(p.data(), N, S);
	FFT2D(p.data(), phat.data(), N, 1);

	for (int i = 0; i < N * N2; ++i) a[i] = Sqr(abs(phat[i]));
}
 
// root mean square error between two densities
// where translation is unknown
double RMS_translation(double* p, double* ptrue, int N)
{
	int N2 = N / 2 + 1;

	vector<dcomplex> phat(N * N2);
	vector<dcomplex> ptruehat(N * N2);
	vector<dcomplex> convhat(N * N2);
	vector<double> conv(N * N);

	// convolve p with ptrue
	FFT2D(p, phat.data(), N, 1);
	FFT2D(ptrue, ptruehat.data(), N, 1);

	for (int i = 0; i < N * N2; ++i)
	{
		convhat[i] = phat[i] * ptruehat[i];
	}

	IFFT2D(convhat.data(), conv.data(), N, 1);

	// max of convolution
	double err = -1e100;
	double pnormsq = 0;
	double ptruenormsq = 0;
	for (int i = 0; i < N * N; ++i)
	{
		if (conv[i] > err)
		{
			err = conv[i];
		}
		pnormsq += Sqr(p[i]);
		ptruenormsq += Sqr(ptrue[i]);
	}

	// return rms error
	err = sqrt((pnormsq + ptruenormsq) / N / N - 2 * err);

	return err;
}

// root mean square error between two densities
// where translation and vertical reflection is unknown
double RMS(double* p, double* ptrue, int N)
{
	// reflection of density
	vector<double> pflip(N * N);
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			pflip[N - j + i * N] = p[j + i * N];
		}
	}

	// return min error of two possible reflections
	double err = RMS_translation(p, ptrue, N);
	double errflip = RMS_translation(pflip.data(), ptrue, N);
	cout << err << "		" << errflip << endl;

	return min(err, errflip);
}


int main()
{
	int N = os * 121 + 1;
	int N2 = N / 2 + 1;
	vector<double> p(N * N);
	vector<dcomplex> phat(N * N2);
	vector<int> S(N * N, 0);
	vector<double> a(N * N2);
	vector<double> sigdatas(N * N2);
	vector<double> sigPNs(N * N2);

	// non-uniform Gaussian noise
	// variance is radial Gaussian centered at (0,0)
	/*
	double R = 0.2; // stdev of radial Gaussian
	double x; double y; double r;
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N2; ++j)
		{
			x = min(i, N - i); y = j;
			x /= N2; y /= N2;
			r = sqrt(Sqr(x) + Sqr(y));
			sigdatas[j + i * N2] = sigdata * exp(-Sqr(r / R) / 2);
			sigPNs[j + i * N2] = sigPN * exp(-Sqr(r / R) / 2);
		}
	}
	*/

	// uniform Gaussian noise
	//for (int i = 0; i < N * N2; ++i) sigdatas[i] = sigdata;
	//for (int i = 0; i < N * N2; ++i) sigPNs[i] = sigPN;

	MakeData(a.data(), S.data(), N);
	Plota(a.data(), N, 0);
	for (int i = 0; i < N * N2; ++i)
	{
		sigdatas[i] = sqrt(a[i]);
		sigPNs[i] = sqrt(a[i]);
	}

	// add noise to data
	gsl_rng* rng = gsl_rng_alloc(gsl_rng_ranlxs0);
	gsl_rng_set(rng, 12);
	for (int i = 0; i < N * N2; ++i)
	{
		a[i] += gsl_ran_gaussian(rng, sigdatas[i]);
		if (a[i] < 0)
		{
			a[i] = 0;
		}
	}

	// print noise and intensity values on (0,0) to (0,1) line
	for (int j = 0; j < N2; ++j)
	{
		cout << "y: " << j * 1.0 / N2 << "		sigdata: " << sigdatas[j] << "		A: " << a[j] << endl;
	}

	// Initialization
	//gsl_rng* rng = gsl_rng_alloc(gsl_rng_ranlxs0);
	//gsl_rng_set(rng, 12);
	for (int i = 0; i < N * N; ++i)
	{
		if (S[i])
		{
			p[i] = 1e6*gsl_rng_uniform(rng);
		}
	}

	int counter = 0;
	Plot2D(p.data(), N, counter++);

	// Reconstruction 
	for (int c = 0; c < Ncycle; ++c)
	{
		cout << c << endl;
		for (int i = 0; i < NHIO; ++i)
		{
			HIO(p.data(), a.data(), S.data(), sigPNs.data(), N);
		}
		Plot2D(p.data(), N, counter++);

		for (int i = 0; i < NER; ++i)
		{
			ER(p.data(), a.data(), S.data(), sigPNs.data(), N);
		}
		Plot2D(p.data(), N, counter++);

		Shrinkwrap(p.data(), S.data(), N, counter);
		for (int i = 0; i < NER; ++i)
		{
			ER(p.data(), a.data(), S.data(), sigPNs.data(), N);
		}
		//ER(p.data(), a.data(), S.data(), N);

		// plot max and min of p for debugging
		double minp = 1e100;
		double maxp = -1e100;
		for (int i = 0; i < N * N; ++i)
		{
			if (p[i] > maxp)
			{
				maxp = p[i];
			}
			if (p[i] < minp)
			{
				minp = p[i];
			}
		}
		cout << "Max of p:	" << maxp << "		Min of p:  " << minp << endl;
	}
	Plot2D(p.data(), N, counter);

	// RMS error
	vector<double> ptrue(N * N);
	Makep(ptrue.data(), N, S.data());

	double rms = RMS(p.data(), ptrue.data(), N);
	cout << "RMS: " << rms << endl;

	return 1;
}
