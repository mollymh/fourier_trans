import math as m
import numpy as np

def dft(f, scale=True):
	"""
	Title:
		dft
	Description:
		Performs a Discrete Fourier Transform on given 1D signal
	Attributes:
		f - 1D ndarray signal to be transformed
		scale - if true, divide by length of signal
	Author:
		Molly Hill, mmh5847@rit.edu
	"""
	#error-checking
	if type(f) != np.ndarray:
		raise TypeError("Source signal must be ndarray.")
	if type(scale) != bool:
		scale = True
		print("Scale var must be Boolean. Defaulting to True.")	
	if f.dtype != np.complex128:
		f = f.astype(np.complex128)
		print("Converting signal to dtype complex128.")
	
	#initializes variables
	M = len(f)
	u = np.indices((M,M))
	ux = u[0]*u[1]

	#handles scale factor
	if scale == True:
		sc = M
	else:
		sc = 1

	#computes transform
	Ftrans = np.sum(np.multiply(f,np.exp((-2*m.pi*1j*ux)/M)),axis=0)/sc
	
	return Ftrans

if __name__ == '__main__':
	import numerical
	import numpy
	import time

	N = 2**12
	f = numpy.ones(N, dtype=numpy.complex128)

	repeats = 10
	print('Repetitions = {0}'.format(repeats))

	startTime = time.clock()
	for repeat in range(repeats):
		F = numerical.dft(f)
	string = 'Average time per transform = {0:.8f} [s] ({1}-point DFT)'
	print(string.format((time.clock() - startTime)/repeats, len(f)))

	startTime = time.clock()
	for repeat in range(repeats):
		F = numpy.fft.fft(f)
	string = 'Average time per transform = {0:.8f} [s] ({1}-point FFT)'
	print(string.format((time.clock() - startTime)/repeats, len(f)))
