import matplotlib.pyplot as plt
from scipy.io import wavfile # get the api
from scipy.fftpack import fft
from pylab import *

def f(filename):
    # song files are in ogg... we need it to be in wav.
    fs, data = wavfile.read(filename) 
    
    # songs have multiple channels, but we only need one channel
    a = data.T[0]
    
    # this is 8-bit track, b is now normalized on [-1,1)
    #b=[(ele/2**16)*2-1 for ele in a] 

    # create a list of complex number
    c = fft(a)

    # only need half of the fft list (because the internet says so)
    d = len(c)//2 

    #bam, it is plotted and saved. 
    #plt.plot(abs(c[:(d-1)]),'r')
    #savefig(filename+'.png',bbox_inches='tight')
	
    return c

guitar = f("auldlangguitar.wav")
violin = f("auldlangviolin.wav")
harmon = f("auldlangharmonica.wav")
combine= f("combined.wav")
cut = combine[:-14]
combined2 = guitar + violin

plt.plot(np.abs(guitar), 'r')
#plt.show()
savefig('guitarplot.png',bbox_inches='tight')

gc = np.dot(guitar, combined2)
vc = np.dot(violin, combined2)
hc = np.dot(harmon, combined2)

ng = guitar - (np.dot(guitar, violin) * (violin / len(violin))) - (np.dot(guitar, violin) * (harmon / len(harmon))) #/ np.linalg.norm(guitar)
nv = violin - (np.dot(violin, guitar) * (guitar / len(guitar))) - (np.dot(violin, harmon) * (harmon / len(harmon)))#/ np.linalg.norm(violin)
nh = harmon - (np.dot(harmon, guitar) * (guitar / len(guitar))) - (np.dot(harmon, violin) * (violin / len(violin)))#/ np.linalg.norm(harmon)
nc = combined2 #/ np.linalg.norm(cut)
 
a = np.column_stack((ng, nv, nh))
a2 = np.column_stack((ng, nv))
x, res, rank, s = np.linalg.lstsq(a, nc)
x2, res2, rank2, s2 = np.linalg.lstsq(a2, nc)
plt.plot(np.abs(nv * x2[0]), 'r')
#plt.show()
savefig('decompguitarplot.png',bbox_inches='tight')
decompGuitar = np.fft.ifft(ng * x[0] + nv * x[1] + nh * x[2])
print("X\n")
print(x)

print("decomp real")
print(np.real(decompGuitar))

decompreal = np.real(decompGuitar)
decompreal = decompreal #/ np.min(np.abs(decompreal[np.nonzero(decompreal)]))

b = np.column_stack((decompreal.astype(uint8), decompreal.astype(uint8)))

origfs, origdata = wavfile.read("auldlangguitar.wav")
wavfile.write("decompguitar.wav", origfs, b)
np.savetxt("guitar.csv", guitar, delimiter= ",")
np.savetxt("combined.csv", combine, delimiter= ",")
np.savetxt("channel2.csv", b, delimiter= ",")
print("decomp orig")
print(np.min(decompreal[np.nonzero(decompreal)]))
