import numpy as np
import ipcv

def dft2(f, scale=True):
	"""
	Title:
		dft2
	Description:
		Performs a Discrete Fourier Transform on given 2D array
	Attributes:
		f - 2D ndarray to be transformed
		scale - if true, divide by length of signal
	Author:
		Molly Hill, mmh5847@rit.edu
	"""
	#error-checking
	if type(f) != np.ndarray:
		raise TypeError("Source must be ndarray.")
	if type(scale) != bool:
		scale = True
		print("Scale var must be Boolean. Defaulting to True.")	
	if f.dtype != np.complex128:
		f = f.astype(np.complex128)
		print("Converting signal to dtype complex128.")

	Fr = np.array([])
	Fc = np.array([])

	for r in np.arange(f.shape[0]): #transform one dimension
		Fr = np.append(Fr,ipcv.dft(f[r],scale))
	Fr = np.reshape(Fr,f.shape)
	for c in np.arange(f.shape[1]): #transform other dimension
		Fc = np.append(Fc,ipcv.dft(Fr[:,c],scale))
	Ftrans = np.reshape(Fc,f.shape)

	return Ftrans

if __name__ == '__main__':
	import numerical
	import numpy
	import time

	M = 2**5
	N = 2**5
	f = numpy.ones((M,N), dtype=numpy.complex128)

	repeats = 10
	print('Repetitions = {0}'.format(repeats))

	startTime = time.clock()
	for repeat in range(repeats):
		F = numerical.dft2(f)
	string = 'Average time per transform = {0:.8f} [s] '
	string += '({1}x{2}-point DFT2)'
	print(string.format((time.clock() - startTime)/repeats, M, N))

	startTime = time.clock()
	for repeat in range(repeats):
		F = numpy.fft.fft2(f)
	string = 'Average time per transform = {0:.8f} [s] '
	string += '({1}x{2}-point FFT2)'
	print(string.format((time.clock() - startTime)/repeats, M, N))
