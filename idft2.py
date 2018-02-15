import numpy as np
import ipcv

def idft2(F, scale=False):
	"""
	Title:
		idft2
	Description:
		Performs an Inverse Discrete Fourier Transform on given 2D array
	Attributes:
		f - 2D ndarray to be transformed
		scale - if true, divide by length of signal
	Author:
		Molly Hill, mmh5847@rit.edu
	"""
	#error-checking
	if type(F) != np.ndarray:
		raise TypeError("Source must be ndarray.")
	if type(scale) != bool:
		scale = True
		print("Scale var must be Boolean. Defaulting to True.")	
	if F.dtype != np.complex128:
		F = F.astype(np.complex128)
		print("Converting signal to dtype complex128.")

	fr = np.array([])
	fc = np.array([])

	for r in np.arange(F.shape[0]): #transform one dimension
		fr = np.append(fr,ipcv.idft(F[r],scale))
	fr = np.reshape(fr,F.shape)
	for c in np.arange(F.shape[1]): #transform other dimension
		fc = np.append(fc,ipcv.idft(fr[:,c],scale))
	ftrans = np.reshape(fc,F.shape)

	return ftrans

if __name__ == '__main__':
	import numerical
	import numpy
	import time

	M = 2**5
	N = 2**5
	F = numpy.zeros((M,N), dtype=numpy.complex128)
	F[0,0] = 1

	repeats = 10
	print('Repetitions = {0}'.format(repeats))

	startTime = time.clock()
	for repeat in range(repeats):
		f = numerical.idft2(F)
	string = 'Average time per transform = {0:.8f} [s] '
	string += '({1}x{2}-point iDFT2)'
	print(string.format((time.clock() - startTime)/repeats, M, N))

	startTime = time.clock()
	for repeat in range(repeats):
		f = numpy.fft.ifft2(F)
	string = 'Average time per transform = {0:.8f} [s] '
	string += '({1}x{2}-point iFFT2)'
	print(string.format((time.clock() - startTime)/repeats, M, N))
