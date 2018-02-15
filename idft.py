import math as m
import numpy as np

def idft(F, scale=False):
	"""
	Title:
		idft
	Description:
		Performs an Inverse Discrete Fourier Transform on given 1D signal
	Attributes:
		f - 1D ndarray signal to be transformed
		scale - if true, divide by length of signal
	Author:
		Molly Hill, mmh5847@rit.edu
	"""
	#error-checking
	if type(F) != np.ndarray:
		raise TypeError("Source signal must be ndarray.")
	if type(scale) != bool:
		scale = True
		print("Scale var must be Boolean. Defaulting to True.")	
	if F.dtype != np.complex128:
		F = F.astype(np.complex128)
		print("Converting signal to dtype complex128.")
	
	#initializes variables
	M = len(F)
	u = np.indices((M,M))
	ux = u[0]*u[1]
	
	#handles scale factor
	if scale == True:
		sc = M
	else:
		sc = 1
	
	#computes inverse transform	
	ftrans = np.sum(np.multiply(F,np.exp((2*m.pi*1j*ux)/M)),axis=0)/sc
	
	return ftrans

if __name__ == '__main__':
	import numerical
	import numpy
	import time

	N = 2**12
	F = numpy.zeros(N, dtype=numpy.complex128)
	F[0] = 1

	repeats = 10
	print('Repetitions = {0}'.format(repeats))

	startTime = time.clock()
	for repeat in range(repeats):
		f = numerical.idft(F)
	string = 'Average time per transform = {0:.8f} [s] ({1}-point iDFT)'
	print(string.format((time.clock() - startTime)/repeats, len(F)))

	startTime = time.clock()
	for repeat in range(repeats):
		f = numpy.fft.ifft(F)
	string = 'Average time per transform = {0:.8f} [s] ({1}-point iFFT)'
	print(string.format((time.clock() - startTime)/repeats, len(F)))
