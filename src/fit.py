import matplotlib.pyplot as plt
import traceback
import pandas as pd 
import datetime
import numpy as np
import itertools
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.stats import linregress
from skimage.measure import block_reduce
import argparse
nua_min = 1e-9*1e9
nua_max = 1e-9*7e11
num_min = 1e-9*4e8
num_max = 1e-9*3.2e13
nuc_min = 1e-9*1.0e12
nuc_max = 1e-9*1.2e22
trigger = datetime.datetime(2024, 2, 5, 22, 13, 6, 00)
t0 = 1
nuc_0 = 1.08e9
plotdata = pd.read_csv("grbmeas.csv")
plotdata = plotdata.sort_values(by='freq').copy(deep=True)
# plotdata = plotdata[plotdata['obsdate'] > 2e-3]
print(plotdata)
otherdata = pd.read_csv("otherdata.txt")
# otherdata = otherdata[(otherdata['freq'] < 4.2e5) | (otherdata['freq'] > 1e9)].copy(deep=True)
# otherdata = otherdata[(otherdata['obsdate'] < 11)].copy(deep=True)
parser = argparse.ArgumentParser()
parser.add_argument("--forwardOnly",action="store_true")
parser.add_argument("--noerrors",action="store_true")
parser.add_argument("--k", type=int, default=2, help="Value for k, (use 0 or 2)")
args = parser.parse_args()
margin = 0.1
def powerlaw(t,a,k):
    return a*t**k
def wrap_sbpl(t,amp, tb, a1, a2, d):
    f = sbpl(amplitude=amp, x_break=tb, alpha_1=a1, alpha_2=a2, delta=d)
    return f(t)
def double_sbpl(x,amp,x_break1,a1,n1,a2,n2,x_break2):
    return amp*x_break1
xrtdata = []
batdata = []
# keV
PLANCK = 4.135667696e-15 # eV / Hz
xrtfreq = ((10e3+0.3e3)/2)/PLANCK/1e9
batfreq = ((150e3+15e3)/2)/PLANCK/1e9
# eV / (eV * (1/Hz)) / (Hz/GHz) -> GHz
xrttimeincluded = []
battimeincluded = []
with open("batxrt.txt","r") as f:
    lines = f.readlines()
    isXRT = False
    for l in lines:
        if "xrt" in l:
            isXRT=True
         
        cols = l.replace("\n","").split("\t")
        if len(cols)==6:
            if isXRT:
                if (float(cols[0])/60/60/24) not in xrttimeincluded:
                    xrtdata.append((float(cols[0])/60/60/24,xrtfreq,1e6*float(cols[3]),1e6*float(cols[4]),1e6*0.1*float(cols[3])))
                    xrttimeincluded.append(float(cols[0])/60/60/24)
            else:
                if (float(cols[0])/60/60/24) not in battimeincluded:
                    batdata.append((float(cols[0])/60/60/24,batfreq,1e6*float(cols[3]),1e6*float(cols[4]),1e6*0.1*float(cols[3])))
                    battimeincluded.append(float(cols[0])/60/60/24)
xrtdata.append((otherdata['obsdate'][0],otherdata['freq'][0],otherdata['flux'][0],otherdata['err'][0],otherdata['rms'][0]))
# xrt = np.array(xrtdata,dtype=[('obsdate','f8'),('freq','f8'),('flux','f8'),('err','f8'),('rms','f8')])
# print(xrt)
# xrtcopy = xrt[(xrt['obsdate'] > 1e-2) & (xrt['obsdate'] < 8)].copy()
# tvals = xrtcopy['obsdate']
# fluxvals = xrtcopy['flux']
# timesort = np.argsort(tvals)
# tvals = tvals[timesort]
# fluxvals = fluxvals[timesort]
# BLOCK = 4
# tbin = []
# fbin = [] 
# for i in range(0,len(tvals),BLOCK): #ate(zip(tvals[::BLOCK],fluxvals[::BLOCK])):
#     # print(tvals[i:(i+BLOCK)],np.mean(tvals[i:(i+BLOCK)]),fluxvals[i:(i+BLOCK)],np.mean(fluxvals[i:(i+BLOCK)]))
#     if tvals[i] < 1e-2:
#         continue
#     if ((tvals[i:(i+BLOCK)].max()-tvals[i:(i+BLOCK)].min())/np.mean(tvals[i:(i+BLOCK)])) < 1.5:
#         tbin.append(np.mean(tvals[i:(i+BLOCK)]))
#         fbin.append(np.mean(fluxvals[i:(i+BLOCK)]))
# xrtcopy = xrt[ (xrt['obsdate'] >= 8)].copy()
# tvals = xrtcopy['obsdate']
# fluxvals = xrtcopy['flux']
# timesort = np.argsort(tvals)
# tvals = tvals[timesort]
# fluxvals = fluxvals[timesort]
# BLOCK = 1
# for i in range(0,len(tvals),BLOCK): #ate(zip(tvals[::BLOCK],fluxvals[::BLOCK])):
#         tbin.append(tvals[i])
#         fbin.append(fluxvals[i])
# tbin = np.array(tbin)[12:]
# fbin = np.array(fbin)[12:]
# xrtdata = []
# for t,f in zip(tbin,fbin):
#     xrtdata.append((t,xrtfreq,f,0.1*f,0))

# with open("xrtdata.txt","r") as f:
#     lines = f.readlines()
#     for l in lines:
#         cols = l.replace("\n","").split("\t")
#         if len(cols)==6:
#             xrtdata.append((float(cols[0])/60/60/24,xrtfreq,1e6*float(cols[3])/1e-23/bw,1e6*float(cols[4])/1e-23/bw,1e6*0.1*float(cols[3])/1e-23/bw))
# xrtdata.append((otherdata['obsdate'][0],otherdata['freq'][0],otherdata['flux'][0],otherdata['err'][0],otherdata['rms'][0]))
xrt = pd.DataFrame(xrtdata, columns=["obsdate","freq","flux","err","rms"])
xrt = xrt[xrt['obsdate'] >0.01]
bat = pd.DataFrame(batdata, columns=["obsdate","freq","flux","err","rms"])
otherdata = pd.concat([plotdata,otherdata,xrt,bat])
fitdata = otherdata[(otherdata['freq'] < 4.2e5) | ((otherdata['freq'] > 1e9) & (otherdata['freq'] <1.5e9)) ]
# otherdata = otherdata[(otherdata['freq'] < 4.2e5)].copy(deep=True)
# fitdata = otherdata[(otherdata['freq'] < 1.5e9)]
freqbin = [(plotdata['freq'][0] - plotdata['freq'][0]*margin,plotdata['freq'][0] + plotdata['freq'][0]*margin)]
for freq in np.unique(plotdata['freq']):
    addbin = True
    for fb in freqbin:
        if (freq > fb[0]) and (freq < fb[1]):
            addbin = False
    if addbin:
        freqbin.append((freq - freq*margin,freq + freq*margin))
    
for freq in np.unique(otherdata['freq']):
    addbin = True
    for fb in freqbin:
        if (freq > fb[0]) and (freq < fb[1]):
            addbin = False
    if addbin:
        freqbin.append((freq - freq*margin,freq + freq*margin))
print(len(freqbin),"freq bins")
# timedata = otherdata['obsdate'].to_numpy()
# fluxdata = otherdata['flux'].to_numpy()
# BLOCK = 2
# remainder = len(timedata) % BLOCK
# timedata = timedata[:-remainder]
# fluxdata = fluxdata[:-remainder]
# bintime = block_reduce(timedata,block_size=BLOCK,func=np.mean)
# binflux = block_reduce(fluxdata,block_size=BLOCK,func=np.mean)
# 
# datalist = []
# for t,f in zip(bintime,binflux):
#     datalist.append((t,1245264.45995191,f,0.1*f,0))
# print(datalist)
# for row in otherdata[otherdata['freq'] < 1e6].to_numpy():
#     datalist.append(tuple(row))
# otherdata = pd.DataFrame(datalist,columns=["obsdate","freq","flux","err","rms"])
print(len(freqbin),"freq bins")
def powerlaw(t,a,k):
    return a*t**k
def wrap_sbpl(t,amp, tb, a1, a2, d):
    f = sbpl(amplitude=amp, x_break=tb, alpha_1=a1, alpha_2=a2, delta=d)
    return f(t)
def double_sbpl(x,amp,x_break1,a1,n1,a2,n2,x_break2):
    return amp*x_break1
xrtdata = []
# bw = 0.7e3/4.135667696e-15
# xrtfreq = ((10+0.3)/2)/4.135667696e-15/1e9
# with open("xrtdata.txt","r") as f:
#     lines = f.readlines()
#     for l in lines:
#         cols = l.replace("\n","").split("\t")
#         if len(cols)==6:
#             xrtdata.append((float(cols[0])/60/60/24,xrtfreq,1e6*float(cols[3])/1e-23/bw,1e6*float(cols[4])/1e-23/bw,1e6*0.1*float(cols[3])/1e-23/bw))
# xrtdata.append((otherdata['obsdate'][0],otherdata['freq'][0],otherdata['flux'][0],otherdata['err'][0],otherdata['rms'][0]))
# xrt = np.array(xrtdata,dtype=[('obsdate','f8'),('freq','f8'),('flux','f8'),('err','f8'),('rms','f8')])
# with open("xrtout.csv", "w") as xrtf:
#     xrtf.write("obsdate,freq,flux,err,rms\n")
#     for x in xrtdata:
#         xrtf.write(f"{x[0]},{x[1]},{x[2]},{x[3]},{x[4]}\n")
# exit()




if args.forwardOnly:
    initial_guess = [1e-3,15,200,nuc_0,10,2.2]
    bounds = [(1e-5,10),(nua_min,nua_max),(num_min,num_max),(nuc_min,nuc_max),(6,15),(2,3)]
    bounds0 = tuple([b[0] for b in bounds])
    bounds1 = tuple([b[1] for b in bounds])
    bounds = [bounds0,bounds1]
    curdata = fitdata
    tdata = curdata['obsdate']
    nudata = curdata['freq']
    xdata = (tdata,nudata)
    ydata = curdata['flux']*1e-6
    yerr = np.sqrt(curdata['err']**2 + curdata['rms']**2)*1e-6
    popt, pcov = curve_fit(wrap_forward, xdata, ydata, p0=initial_guess,bounds=bounds,sigma=yerr,maxfev=100000)
    varnames = ["nua_0","num_0","nuc_0","jet_break","p"]
    text = f"f0={popt[0]}+/-{np.absolute(pcov[0][0])**0.5}"
    print(text)
    for ind,var in enumerate(varnames):
        vnum = ind+1
        text = f" {var}={popt[vnum]}+/-{np.absolute(pcov[vnum][vnum])**0.5}"
        print(text)
    print("d=0.2")
    bigpopt = popt
    bigpcov = pcov
    bigsigma = np.sqrt(np.diagonal(bigpcov))
    def getminmax(ivar, bigpopt, bigsigma):
        minarray = []
        maxarray = []
        for i,tval in tqdm(enumerate(ivar[0]),total=len(ivar[0])):
            meas = []
            for f0try in [bigpopt[0]+bigsigma[0],bigpopt[0]-bigsigma[0]]:
                for nu01try in [bigpopt[1]+bigsigma[1],bigpopt[1]-bigsigma[1]]:
                    for nu02try in [bigpopt[2]+bigsigma[2],bigpopt[2]-bigsigma[2]]:
                        fully = wrap_bigsbpl(ivar,f0try,nu01try,nu02try)[i]
                        meas.append(fully)
            minarray.append(np.amin(meas))
            maxarray.append(np.amax(meas))
         
        return np.array(minarray), np.array(maxarray)
else:
    def getminmax(ivar, bigpopt, bigsigma):
        minarray = []
        maxarray = []
        for i,tval in tqdm(enumerate(ivar[0]),total=len(ivar[0])):
            meas = []
            for f0try in [bigpopt[0]+bigsigma[0],bigpopt[0]-bigsigma[0]]:
                for frevtry in [bigpopt[1]+bigsigma[1],bigpopt[1]-bigsigma[1]]:
                    for nu01revtry in [bigpopt[2]+bigsigma[2],bigpopt[2]-bigsigma[2]]:
                        for nu01try in [bigpopt[3]+bigsigma[3],bigpopt[3]-bigsigma[3]]:
                            for nu02try in [bigpopt[4]+bigsigma[4],bigpopt[4]-bigsigma[4]]:
                                fully = wrap_bigsbpl(ivar,f0try,frevtry,nu01revtry,nu01try,nu02try)[i]
                                meas.append(fully)
            minarray.append(np.amin(meas))
            maxarray.append(np.amax(meas))
         
        return np.array(minarray), np.array(maxarray)
    def wrap_bigsbpl(ivar, f0, f0rev, nu01rev,nu01,nu02,nu03):
        
        s = 5
        d = 0.2
        k=args.k
        t, nu = ivar
        res = []
        frev = reverse_shock(ivar, f0rev, nu01rev,k)
        f = theory_bigsbpl(ivar, f0, nu01, nu02,nu03, k)
        return frev + f
    initial_guess = [1e-3,5e-5,10,13,100,8e8]
    bounds = [(1e-6,1),(3e-5,1e5),(0.1,1e6),(6,15),(33,1e4),(1e8,2e9)]
    bounds0 = tuple([b[0] for b in bounds])
    bounds1 = tuple([b[1] for b in bounds])
    bounds = [bounds0,bounds1]
    curdata = fitdata
    tdata = curdata['obsdate']
    nudata = curdata['freq']
    xdata = (tdata,nudata)
    ydata = curdata['flux']*1e-6
    yerr = np.sqrt(curdata['err']**2 + curdata['rms']**2)*1e-6
    popt, pcov = curve_fit(wrap_bigsbpl, xdata, ydata, p0=initial_guess,bounds=bounds,sigma=yerr)
    varnames = ["f0_rev","nua0_rev","nua_0","num_0","nuc_0"]
    text = f"f0={popt[0]}+/-{np.absolute(pcov[0][0])**0.5}"
    print(text)
    for ind,var in enumerate(varnames):
        vnum = ind+1
        text = f" {var}={popt[vnum]}+/-{np.absolute(pcov[vnum][vnum])**0.5}"
        print(text)
    print("d=0.2")
    bigpopt = popt
    bigpcov = pcov
    bigsigma = np.sqrt(np.diagonal(bigpcov))
    

    # Non-relativistic Rev. Shock
    def nonrel_reverse_shock(ivar, f0, nu0_1,k):
        t, nu = ivar
        s=10
        res = []
        p=2
        t0=0.05
        nu0_2 = 100
        nu0_3 = 1e9
        if k==0:
            g = ((7/2) + (3/2) )/2
        else:
            g = ((3/2) + (1/2) )/2
        a1 = -(11*g+12)/(7*(2*g+1))
        b1 = -(3*(11*g+12))/(35*(2*g+1))
        b2 = -(3*(5*g+8))/(7*(2*g+1))
        b3 = -(3*(5*g+8))/(7*(2*g+1))
        for tval,nuval in zip(t,nu):
            fnu_m = f0*(tval/t0)**a1
            nua = nu0_1*(tval/t0)**b1
            num = nu0_2*(tval/t0)**b2
            nuc = nu0_3*(tval/t0)**b3
            if nuval < nua:
                c1 = 2 
                c2 = 1/3
                c3 = -(p-1)/2
                fpk = fnu_m*(nua/num)**(1/3)
                result = dsbpl(nuval,fpk,nua,c1,c2,num,c3,s)
            else:
                c1 = 1/3
                c2 = -(p-1)/2
                c3 = -(p-1)/2 - 0.1
                fpk = fnu_m
                result = dsbpl(nuval,fpk,num,c1,c2,nuc,c3,s)
            res.append(result)
        return np.array(res)
    def forwardshock(ivar, f0,nu01,nu02,nu03):
        
        s = 5
        d = 0.2
        k=args.k
        t, nu = ivar
        res = []
        f = theory_bigsbpl(ivar, f0, nu01, nu02, nu03,k)
        return f
        
    initial_guess = [1e-3,13,100,1.2e9]
    bounds = [(1e-5,10),(6,15),(75,1e4),(0.9e9,2e9)]
    bounds0 = tuple([b[0] for b in bounds])
    bounds1 = tuple([b[1] for b in bounds])
    bounds = [bounds0,bounds1]
    curdata = fitdata
    tdata = curdata['obsdate']
    nudata = curdata['freq']
    xdata = (tdata,nudata)
    ydata = curdata['flux']*1e-6
    yerr = np.sqrt(curdata['err']**2 + curdata['rms']**2)*1e-6
    popt, pcov = curve_fit(forwardshock, xdata, ydata, p0=initial_guess,bounds=bounds,sigma=yerr)
    varnames = ["nua_0","num_0","nuc_0"]
    text = f"f0={popt[0]}+/-{np.absolute(pcov[0][0])**0.5}"
    print(text)
    for ind,var in enumerate(varnames):
        vnum = ind+1
        text = f" {var}={popt[vnum]}+/-{np.absolute(pcov[vnum][vnum])**0.5}"
        print(text)
    print("d=0.2")
    forwardpopt = popt
    forwardpcov = pcov
    forwardsigma = np.sqrt(np.diagonal(pcov))


   #  def wrap_thinbigsbpl(ivar, f0, frev, nu0rev,nu01,nu02):
   #      
   #      s = 5
   #      d = 0.2
   #      k=args.k
   #      t, nu = ivar
   #      res = []
   #      frev = nonrel_reverse_shock(ivar, frev, nu0rev,k)
   #      f = theory_bigsbpl(ivar, f0, nu01, nu02, k)
   #      return frev + f
   #  initial_guess = [1e-3,5e-5, 10,10,50]
   #  bounds = [(1e-6,1),(3e-5,2),(1,100),(1,100),(15,1e5)]
   #  bounds0 = tuple([b[0] for b in bounds])
   #  bounds1 = tuple([b[1] for b in bounds])
   #  bounds = [bounds0,bounds1]
   #  curdata = otherdata[otherdata['band']!="X1"]
   #  tdata = curdata['obsdate']
   #  nudata = curdata['freq']
   #  xdata = (tdata,nudata)
   #  ydata = curdata['flux']*1e-6
   #  yerr = np.sqrt(curdata['err']**2 + curdata['rms']**2)*1e-6
   #  popt, pcov = curve_fit(wrap_thinbigsbpl, xdata, ydata, p0=initial_guess,bounds=bounds,sigma=yerr)
   #  varnames = ["f0_rev","nua0_rev","nua_0","num_0"]
   #  text = f"f0={popt[0]}+/-{np.absolute(pcov[0][0])**0.5}"
   #  print(text)
   #  for ind,var in enumerate(varnames):
   #      vnum = ind+1
   #      text = f" {var}={popt[vnum]}+/-{np.absolute(pcov[vnum][vnum])**0.5}"
   #      print(text)
   #  print("d=0.2")
   #  thinpopt = popt

def simnu_c(x,f0):
    p=2.2
    return f0*(x/t0)**((2-3*p)/4)
def simnu_m(x,f0):
    p=2.2
    return f0*(x/t0)**((1-3*p)/4)
def simjet(x,f0):
    p=2.2
    return f0*(x/t0)**(-p)
if len(freqbin) == 1:
    fig,axs = plt.subplots(int(np.ceil(len(freqbin)/2)),1,figsize=(15,8),sharex=True,sharey=False)
    flatax = [axs]
else:
    fig,axs = plt.subplots(int(np.ceil(len(freqbin)/2)),2,figsize=(25,35),sharex=True,sharey=False)
    flatax = axs.flatten()
ufreq = np.sort(np.unique(otherdata['freq']))
for ind,ax in enumerate(flatax):
    if ind < len(freqbin):
        freq = np.average(freqbin[ind])
        curdata = otherdata[(otherdata['freq']>freqbin[ind][0]) & (otherdata['freq']<freqbin[ind][1])]
        if curdata.size==0:
            curdata = otherdata[(otherdata['freq']>freqbin[ind][0]) & (otherdata['freq']<freqbin[ind][1])]
        marker = itertools.cycle((',', '+', '.', 'o', '*'))
        linestyle = itertools.cycle(('-', ':', '-.', '--'))
        for freq in np.sort(np.unique(curdata['freq'])):
            xdata = curdata['obsdate']
            ydata = curdata['flux']*1e-6
            yerr = np.sqrt(curdata['err']**2 + curdata['rms']**2)*1e-6
            xline = np.geomspace(otherdata['obsdate'].min(),365,num=100)
            subcurdata = curdata[(curdata['freq']==freq) & (curdata['err']!=-1)]
            subxdata = subcurdata['obsdate']
            subydata = subcurdata['flux']
            subyerr = np.sqrt(subcurdata['err']**2 + subcurdata['rms']**2)
            ax.errorbar(subxdata,subydata,yerr=subyerr,fmt=' ',color='black')
            if freq < 1000:
                labeltext = f'{freq} GHz'
            else:
                labeltext = f'{1e9*freq:5.2e} Hz'
            ax.scatter(subxdata,subydata,label=labeltext,marker=next(marker),color='black')
        # ax.plot(xdata,simnu_c(xdata,8),label="try nu_c",color='red')
        # ax.plot(xdata,simnu_m(xdata,6),label="try nu_m",color='blue',ls=':')
        # ax.plot(xdata,simjet(xdata,1),label="try jet",color='green',ls='--')
        # ax.plot(xdata,powerlaw(xdata/10,0.1,-4),label="try pl",color='black',ls='-.')
        # ax.set_ylim(1e-4,1e5)
        nu = np.array([freq for f in xline])
        yline = wrap_bigsbpl((xline,nu), *bigpopt)*1e6
        ax.plot(xline,yline,color='navy',ls=next(linestyle))
       #  nu = np.array([freq for f in xline])
       #  f0= 5e-3
       #  nu01=25
       #  nu02=200
       #  nu03 = 1.08e9
       #  yline = wrap_bigsbpl((xline,nu),f0, nu01, nu02,nu03 )*1e6
       #  ax.plot(xline,yline,alpha=0.5,color='red',ls=next(linestyle),label="try params")
        if args.forwardOnly:
            tcobs = t0*(freq/bigpopt[3])**2
        else:
            tcobs = t0*(freq/bigpopt[5])**2
        ax.axvline(tcobs,label="nu_c passing through")
        if not args.forwardOnly:
            f0, f0rev, nu01rev, nu01, nu02, nu03 =  tuple(bigpopt)
            yline = reverse_shock((xline,nu), f0rev, nu01rev,args.k)*1e6
           #  ax.plot(xline,yline,alpha=0.5,color='red',ls=next(linestyle))
            yline = theory_bigsbpl((xline,nu), f0, nu01, nu02,nu03, args.k)*1e6
               #  ax.plot(xline,yline,alpha=0.5,color='red',ls=next(linestyle))
            # upper_param = np.array([bigpopt[0]+bigsigma[0],bigpopt[1]+bigsigma[1],bigpopt[2]-bigsigma[2],bigpopt[3]+bigsigma[3],bigpopt[4]+bigsigma[4]])
            # lower_param = np.array([bigpopt[0]-bigsigma[0],bigpopt[1]-bigsigma[1],bigpopt[2]+bigsigma[2],bigpopt[3]-bigsigma[3],bigpopt[4]-bigsigma[4]])
       #  elif ind==(len(ufreq)+1):
       #      freq = otherdata['freq'][0]
       #      curdata = otherdata
        limitdata = curdata[(curdata['freq']==freq) & (curdata['err']==-1)]
        limitxdata = limitdata['obsdate']
        limitydata = limitdata['rms']*3
        if limitdata.size >0:
            ax.scatter(limitxdata,limitydata,marker='v',color='black',label=f"{labeltext} 3$\sigma$ limit")
        if not args.noerrors:
            bound_lower = getminmax((xline,nu),bigpopt,bigsigma)[0]*1e6
            bound_upper = getminmax((xline,nu),bigpopt,bigsigma)[1]*1e6

            ax.fill_between(xline,bound_lower,bound_upper,color='navy',alpha=0.15)
        if not args.forwardOnly:
            yline = forwardshock((xline,nu), *forwardpopt)*1e6
        #     ax.plot(xline,yline,alpha=0.5,color='black',ls=next(linestyle))
       #  bound_upper = forwardshock((xline,nu),*(forwardpopt+forwardsigma))*1e6
       #  bound_lower = forwardshock((xline,nu),*(forwardpopt-forwardsigma))*1e6
       #  ax.fill_between(xline,bound_lower,bound_upper,color='black',alpha=0.15)
       #  yline = wrap_bigsbpl((xline,nu), *bigpopt)
       #  ax.plot(xline,yline,alpha=0.5,color='black')
 #        trypopt = [1.8e-3, 355e-6, 9.5, 50, 400]
 #        yline = wrap_bigsbpl((xline,nu), *trypopt)
 #        ax.plot(xline,yline,alpha=0.5,color='black',label=f'tryfit',ls=':')
       #      
       #  
       #  popt, pcov = curve_fit(wrap_sbpl, xdata, ydata, p0=initial_guess,bounds=bounds)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel("Flux Density ($\mu Jy$)")
        ax.set_xlim(otherdata['obsdate'].min(),365)
        # ax.set_ylim(1e-3,3000)
        # test_theory = [1e-3, 18.5, 60.3 , 2]
        # print(test_theory)
        # ax.plot(xline,yline,color='black',alpha=0.5,ls=':',label='theory')
        # ax.plot(xline,yline,color='black',alpha=0.5,ls='-')


#         test_theory = [1.e-3, 58 , 95 , 0]
#         print(test_theory)
      #   if args.forwardOnly: 
      #       ax.axvline(get_tbreak((xline,nu),*popt),ls=':')
#         yline = theory_bigsbpl((xline,nu), *test_theory)
#         ax.plot(xline,yline,color='black',alpha=0.5,ls=':',label='ISM')
        # yline = wrap_latebigsbpl((xline,nu), *latebigpopt)
#         ax.plot(xline,yline,color='black',ls='--',alpha=0.5,label='nonrel')
 #        if freq==9.0:
#             yline = wrap_earlybigsbpl(xline, *earlybigpopt)
         #    ax.plot(xline,yline,color='black',ls='--',alpha=0.5,label='Reverse Shock')
# d    ef wrap_latebigsbpl(ivar, f0, nu0, a1, b1, c1, c2):
        if len(np.unique(curdata['freq'])) >1:
            freq1 = curdata['freq'].min()
            if freq1 < 1000:
                mintext = f'{freq1} GHz'
            else:
                mintext = f'{1e9*freq1:5.2e} Hz'
            freq2 = curdata['freq'].max()
            if freq2 < 1000:
                maxtext = f'{freq2} GHz'
            else:
                maxtext = f'{1e9*freq2:5.2e} Hz'
            title=f'{mintext} to {maxtext}'
            ax.set_title(title)
        else:
            title=labeltext
            ax.set_title(title)
        ax.legend()
       #  if (freq < 16) and (freq >2):
       #      tpk.append(popt[1])
       #      nupk.append(np.average(curdata['freq']))
# plt.legend()
ax.set_xlabel("Days post-trigger")
if args.k==2:
    fig.suptitle(f"Stellar Wind Profile")
elif args.k==0:
    fig.suptitle(f"ISM Profile")
plt.tight_layout()
plt.savefig("tryfit.png")
plt.close()
if args.forwardOnly:
    forwardpopt = bigpopt     
    forwardshock = wrap_bigsbpl
chisq = 0
measdata = fitdata[fitdata['flux'] > 0 ]
dof = len(measdata) - len(forwardpopt)
for d,nu,f,ferr,rms in measdata[['obsdate','freq','flux','err','rms']].to_numpy():
    errtot = np.sqrt(ferr**2 + rms**2)*1e-6
    model = yline = forwardshock((np.array([d]),np.array([nu])), *forwardpopt)
    chisq += ((f*1e-6 - model) / errtot)**2 / dof
print("FORWARD ONLY red. chisq:",chisq,"dof:",dof,"fitting indices")
if not args.forwardOnly:
    chisq = 0
    measdata = fitdata[fitdata['flux'] > 0 ]
    dof = len(measdata) - len(bigpopt)
    for d,nu,f,ferr,rms in measdata[['obsdate','freq','flux','err','rms']].to_numpy():
        errtot = np.sqrt(ferr**2 + rms**2)*1e-6
        model = yline = wrap_bigsbpl((np.array([d]),np.array([nu])), *bigpopt)
        chisq += ((f*1e-6 - model) / errtot)**2 / dof
    print("FORWARD + REVERSE red. chisq:",chisq,"dof:",dof,"fitting indices")
    
if len(freqbin) == 1:
    fig,axs = plt.subplots(int(np.ceil(len(freqbin)/2)),1,figsize=(15,8),sharex=True,sharey=False)
    flatax = [axs]
else:
    fig,axs = plt.subplots(int(np.ceil(len(freqbin)/2)),2,figsize=(25,35),sharex=True,sharey=False)
    flatax = axs.flatten()
ufreq = np.sort(np.unique(otherdata['freq']))
for ind,ax in enumerate(flatax):
    if ind < len(freqbin):
        freq = np.average(freqbin[ind])
        curdata = otherdata[(otherdata['freq']>freqbin[ind][0]) & (otherdata['freq']<freqbin[ind][1])]
        if curdata.size==0:
            curdata = otherdata[(otherdata['freq']>freqbin[ind][0]) & (otherdata['freq']<freqbin[ind][1])]
        marker = itertools.cycle((',', '+', '.', 'o', '*'))
        linestyle = itertools.cycle(('-', ':', '-.', '--'))
        for freq in np.sort(np.unique(curdata['freq'])):
            xdata = curdata['obsdate']
            ydata = curdata['flux']*1e-6
            yerr = np.sqrt(curdata['err']**2 + curdata['rms']**2)*1e-6
            xline = np.geomspace(otherdata['obsdate'].min(),365,num=100)
            subcurdata = curdata[(curdata['freq']==freq) & (curdata['err']!=-1)]
            subxdata = subcurdata['obsdate']
            subydata = subcurdata['flux']
            subyerr = np.sqrt(subcurdata['err']**2 + subcurdata['rms']**2)
            nu = np.array([freq for f in subxdata])
            ymodel = wrap_bigsbpl((subxdata,nu), *bigpopt)*1e6
            if freq < 1000:
                labeltext = f'{freq} GHz'
            else:
                labeltext = f'{1e9*freq:5.2e} Hz'
            ax.scatter(subxdata,subydata - ymodel,label=labeltext,marker=next(marker),color='black')
        # ax.plot(xdata,simnu_c(xdata,8),label="try nu_c",color='red')
        # ax.plot(xdata,simnu_m(xdata,6),label="try nu_m",color='blue',ls=':')
        # ax.plot(xdata,simjet(xdata,1),label="try jet",color='green',ls='--')
        # ax.plot(xdata,powerlaw(xdata/10,0.1,-4),label="try pl",color='black',ls='-.')
        # ax.set_ylim(1e-4,1e5)
       #  nu = np.array([freq for f in xline])
       #  f0= 5e-3
       #  nu01=25
       #  nu02=200
       #  nu03 = 1.08e9
       #  yline = wrap_bigsbpl((xline,nu),f0, nu01, nu02,nu03 )*1e6
       #  ax.plot(xline,yline,alpha=0.5,color='red',ls=next(linestyle),label="try params")
        if args.forwardOnly:
            tcobs = t0*(freq/bigpopt[3])**2
        else:
            tcobs = t0*(freq/bigpopt[5])**2
        ax.axvline(tcobs,label="nu_c passing through")
        ax.set_xscale('log')
        ax.set_yscale('symlog')
        ax.set_ylabel("Flux Density ($\mu Jy$)")
        ax.set_xlim(otherdata['obsdate'].min(),365)
        ax.set_ylim(-1e3,1e3)
        # test_theory = [1e-3, 18.5, 60.3 , 2]
        # print(test_theory)
        # ax.plot(xline,yline,color='black',alpha=0.5,ls=':',label='theory')
        # ax.plot(xline,yline,color='black',alpha=0.5,ls='-')


#         test_theory = [1.e-3, 58 , 95 , 0]
#         print(test_theory)
      #   if args.forwardOnly: 
      #       ax.axvline(get_tbreak((xline,nu),*popt),ls=':')
#         yline = theory_bigsbpl((xline,nu), *test_theory)
#         ax.plot(xline,yline,color='black',alpha=0.5,ls=':',label='ISM')
        # yline = wrap_latebigsbpl((xline,nu), *latebigpopt)
#         ax.plot(xline,yline,color='black',ls='--',alpha=0.5,label='nonrel')
 #        if freq==9.0:
#             yline = wrap_earlybigsbpl(xline, *earlybigpopt)
         #    ax.plot(xline,yline,color='black',ls='--',alpha=0.5,label='Reverse Shock')
# d    ef wrap_latebigsbpl(ivar, f0, nu0, a1, b1, c1, c2):
        if len(np.unique(curdata['freq'])) >1:
            freq1 = curdata['freq'].min()
            if freq1 < 1000:
                mintext = f'{freq1} GHz'
            else:
                mintext = f'{1e9*freq1:5.2e} Hz'
            freq2 = curdata['freq'].max()
            if freq2 < 1000:
                maxtext = f'{freq2} GHz'
            else:
                maxtext = f'{1e9*freq2:5.2e} Hz'
            title=f'{mintext} to {maxtext}'
            ax.set_title(title)
        else:
            title=labeltext
            ax.set_title(title)
        ax.legend()
       #  if (freq < 16) and (freq >2):
       #      tpk.append(popt[1])
       #      nupk.append(np.average(curdata['freq']))
# plt.legend()
ax.set_xlabel("Days post-trigger")
if args.k==2:
    fig.suptitle(f"Stellar Wind Profile")
elif args.k==0:
    fig.suptitle(f"ISM Profile")
plt.tight_layout()
plt.savefig("residuals.png")
plt.close()
