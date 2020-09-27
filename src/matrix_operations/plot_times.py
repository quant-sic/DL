#!/usr/bin/env python3
#
# Plot time measurements.
#

import matplotlib
matplotlib.use('tkagg')

import argparse
import numpy as np
import matplotlib.pyplot as plt

def plot(filename,savefile):
    plt.ion()

    # import data
    data = np.genfromtxt(filename,delimiter='\t',dtype=float,skip_header=1,skip_footer=0)
    N=data[:,0]
    # plot functions, time
    plt.figure(1)
    plt.clf()
    plt.plot(N,data[:,1],label='MM1')
    plt.plot(N,data[:,2],label='MM2')
    plt.plot(N,data[:,3],label='SM')
    plt.plot(N,data[:,4],label='SM COA')
    plt.plot(N,data[:,5],label='cuBlas')

    plt.ylabel('runtime [s]')
    plt.xlabel('# elements')
    plt.xscale('log',basex=2)
    plt.yscale('log',basey=2)
    plt.legend()
    plt.title('Matrix Multiplication functions runtime')
    if savefile != '':
        plt.savefig(savefile + '_func_time.pdf')

    # plot functions, performance
    plt.figure(2)
    plt.clf()
    plt.plot(N,2*N**3/data[:,1]/1e9,label='MM1')
    plt.plot(N,2*N**3/data[:,2]/1e9,label='MM2')
    plt.plot(N,2*N**3/data[:,3]/1e9,label='SM')
    plt.plot(N,2*N**3/data[:,4]/1e9,label='SM COA')
    plt.plot(N,2*N**3/data[:,5]/1e9,label='cuBlas')
    plt.ylabel('performance [Gflops]')
    plt.xlabel('# elements')
    plt.legend()
    plt.title('Matrix Multiplication functions performance')
    if savefile != '':
        plt.savefig(savefile + '_func_perf.pdf')

    plt.show()
    input("Press 'Enter' to continue...")

def main():
    parser = argparse.ArgumentParser(description='Plot time measurements.')
    parser.add_argument('-f',dest='filename',default='times.txt',help='Datei mit Zeitmessung (defaul: time.log)')
    args = parser.parse_args()

    plot(args.filename,'time')

if __name__ == '__main__':
    main()
