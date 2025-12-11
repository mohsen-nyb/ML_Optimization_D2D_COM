from scipy.stats import *
from scipy.integrate import quad
import numpy as np
import math
import random

F = 14 #Number of sub-channels
Tslt = 0.005 #Time slot duration
Tth = 0.08 #Time threshold
SINRTh = 10 #SINR threshold
Omega = 2 #Rayleigh Fading parameter
c = 3 * 1e8
d0 = 10 #Reference distance
f = 2.4 * 1e9 #Operating frequency
Lambdan = 100 #Incoming packet rate
normalizedBuffer = 50 #Normalized buffer size
Pt = 0.2 #Transmit power
GaNoise = 1.38 * 1e-23 * 290 * 1e8 #Noise power
xMin = -50
xMax = 50
yMin = -50
yMax = 50
zMin = 0
zMax = 2
xDelta = xMax - xMin
yDelta = yMax - yMin
zDelta = zMax - zMin
areaTotal = xDelta * yDelta * zDelta
lambda0 = 0.0005 #Newtork density
size = np.random.poisson(lambda0 * areaTotal) #Number of nodes
size += size % 2 #Make sure size is even
x = xDelta * np.random.uniform(0, 1, size) + xMin
y = yDelta * np.random.uniform(0, 1, size) + yMin
z = zDelta * np.random.uniform(0, 1, size - 2) + zMin
z = np.insert(z, 0, 60) #UAV 1 altitude
z = np.insert(z, 0, 50) #UAV 2 altitude
mp = [i for i in range(size)]


def PLoS(s, n, zet=20, v=3 * 1e-4, mi=0.5): #Line-of-sight probability
    if z[s] == z[n]:
        print("Same Height")
        return (1 - np.exp(-(z[s] ** 2) / (2 * (zet ** 2)))) ** (
                math.dist((x[s], y[s], z[s]), (x[n], y[n], z[n])) * np.sqrt(v * mi))
    else:
        return (1 - (((np.sqrt(2 * np.pi) * zet) / math.dist((0, 0, z[s]), (0, 0, z[n]))) * np.abs(
            (1 - norm.cdf(z[s] / zet)) - (1 - norm.cdf(z[n] / zet))))) ** (
                math.dist((x[s], y[s], 0), (x[n], y[n], 0)) * np.sqrt(v * mi))


def b(s, n):
    return np.sqrt(2 * np.exp(2.708 * PLoS(s, n) ** 2))


def RorR(s, n):
    if PLoS(s, n) < 0.5:
        return 1
    else:
        return 1


def Miun(betan, s, n):
    if RorR(s, n) == 1:
        return 1 - ((rice.cdf(betan, b(s, n))) ** F)
    else:
        return 1 - ((rayleigh.cdf(betan)) ** F)


def Pdly(s, n): #Time threshold probability
    return np.exp(-Tth * ((Miun(Beta[s], s, n) / Tslt) - Lambdan))


def rho(s, n):
    return Lambdan * Tslt / (Miun(Beta[s], s, n))


def Pov(s, n): #Buffer overflow probability
    return ((1 - rho(s, n)) * np.exp(-normalizedBuffer * (1 - rho(s, n)))) / (
            1 - rho(s, n) * np.exp(-normalizedBuffer * (1 - rho(s, n))))


def hn(s, n): # Path loss model
    return (c / (4 * np.pi * f)) * np.sqrt((d0 ** ((-1.5 * PLoS(s, n) + 3.5) - 2)) / ((math.dist((x[s], y[s], z[s]), (x[n], y[n], z[n]))) ** (-1.5 * PLoS(s, n) + 3.5)))


def EIfunc(x, RorR, b):
    if RorR == 0:
        return (x ** 2) * (rayleigh.pdf(x))
    else:
        return (x ** 2) * (rice.pdf(x, b))


def EI(m, s, n):
    sum = 0
    for i in range(len(m)):
        sum += Pt * (hn(m[i], n) ** 2) * (
            (quad(lambda x: EIfunc(x, RorR(m[i], n), b(m[i], n)), Beta[m[i]], 10))[
                0]) * (Miun(Beta[m[i]], s, n) / F)
    return sum


def DIfunc(x, RorR, b):
    if RorR == 0:
        return (x ** 4) * (rayleigh.pdf(x))
    else:
        return (x ** 4) * (rice.pdf(x, b))


def DI(m, s, n):
    sum = 0
    sum1 = 0
    for i in range(len(m)):
        j = 0
        sum += (Pt ** 2) * (hn(m[i], n) ** 4) * (
            (quad(lambda x: DIfunc(x, RorR(m[i], n), b(m[i], n)), Beta[m[i]], 10))[
                0]) * ((Miun(Beta[m[i]], s, n) / F) ** 2)
        while j < i:
            sum1 += 2 * Pt * (hn(m[i], n) ** 2) * (
                (quad(lambda x: EIfunc(x, RorR(m[i], n), b(m[i], n)), Beta[m[i]], 10))[
                    0]) * (
                            Miun(Beta[m[j]], s, n) / F) * Pt * (hn(m[j], n) ** 2) * (
                        (quad(lambda x: EIfunc(x, RorR(m[j], n), b(m[j], n)), Beta[m[j]], 10))[0]) * (
                            Miun(Beta[m[j]], s, n) / F)
            j += 1
    return sum + sum1 - (EI(m, s, n) ** 2)


def locmu(m, s, n): #Interference log-normal model parameters
    return np.log(EI(m, s, n)) - np.log(1 + (DI(m, s, n) / (EI(m, s, n) ** 2))) / 2


def sclsigma(m, s, n): #Interference log-normal model parameters
    return np.sqrt(np.log(1 + (DI(m, s, n) / (EI(m, s, n) ** 2))))


def Perrfunc(x, m, s, n):
    if RorR(s, n) == 0:
        return rayleigh.pdf(x) * (0.5 - (0.5 * math.erf((np.log(
            (Pt * (hn(s, n) ** 2) * (x ** 2) / SINRTh) - GaNoise) - locmu(m, s, n)) / (
                                                                np.sqrt(2) * sclsigma(m, s, n)))))
    else:
        return rice.pdf(x, b(s, n)) * (0.5 - (0.5 * math.erf((np.log(
            (Pt * (hn(s, n) ** 2) * (x ** 2) / SINRTh) - GaNoise) - locmu(m, s, n)) / (
                                                                     np.sqrt(2) * sclsigma(m, s, n)))))


def Perr(m, s, n): #Transmission error probability
    return quad(lambda x: Perrfunc(x, m, s, n), Beta[s], 10)[0]


def BetaUpRayleigh(s): # Maximum transmission threshold for Rayleigh fading
    return np.sqrt(- Omega * np.log(1 - (1 - Tslt * Lambdan) ** (1 / F)))


def BetaUpRician(s, n, betan):
    if rice.cdf(betan, b(s, n)) <= ((1 - Lambdan * Tslt) ** (1 / F)):
        return betan
    else:
        return 0


def BetamUpRician(s): # Maximum transmission threshold for Rician fading
    ini_betam = 2
    while BetaUpRician(s, size - (s + 1), ini_betam) != 0:
        ini_betam += 0.01
    return ini_betam


def Ploss(m, s, n): #Overall packet loss probability
    ov = Pov(s, n)
    dly = Pdly(s, n)
    err = Perr(m, s, n)
    plss = ov + ((1 - ov) * dly) + ((1 - ov) * (1 - dly) * err)
    return plss if plss < 1 else 1


def Rn(m, s, n): #Effective throughput
    return Lambdan * (1 - Ploss(m, s, n))


def LCS(bst_betan, bst_rn, j, m, stpdiv=0.5, stpini=0.01, stpth=0.01): 
    stp = stpini * stpdiv
    flag = 0
    k = 0
    while flag != 4:
        while flag == 0:
            stp /= stpdiv
            if (RorR(mp[j], size - (mp[j] + 1)) == 1 and BetaUpRician(mp[j], size - (mp[j] + 1),
                                                                      bst_betan + stp) != 0) or (
                    RorR(mp[j], size - (mp[j] + 1)) == 0 and bst_betan + stp < BetaUpRayleigh(mp[j])):
                Beta[mp[j]] = bst_betan + stp
            rn_can = Rn(m, mp[j], size - (mp[j] + 1))
            print('Current Beta0+:', Beta[mp[j]], 'Current Step0+:', stp, 'Rn Candidate0+:', rn_can)
            if rn_can > bst_rn:
                bst_rn = rn_can
                bst_betan = Beta[mp[j]]
                k += 1
            elif k != 0 and stpdiv != 1:
                flag = 1
            elif k != 0 and stpdiv == 1:
                flag = 4
            elif k == 0:
                flag = 2
                stp *= stpdiv
        while flag == 1:
            stp *= stpdiv
            if stp < stpth:
                flag = 4
                break
            if (RorR(mp[j], size - (mp[j] + 1)) == 1 and BetaUpRician(mp[j], size - (mp[j] + 1),
                                                                      bst_betan + stp) != 0) or (
                    RorR(mp[j], size - (mp[j] + 1)) == 0 and bst_betan + stp < BetaUpRayleigh(mp[j])):
                Beta[mp[j]] = bst_betan + stp
            rn_can = Rn(m, mp[j], size - (mp[j] + 1))
            print('Current Beta1+:', Beta[mp[j]], 'Current Step1+:', stp, 'Rn Candidate1+:', rn_can)
            if rn_can > bst_rn:
                bst_rn = rn_can
                bst_betan = Beta[mp[j]]
                if stp == stpth:
                    flag = 4
            elif stp == stpth:
                flag = 2
                stp *= stpdiv
        while flag == 2:
            stp /= stpdiv
            if bst_betan - stp > 0:
                Beta[mp[j]] = bst_betan - stp
            rn_can = Rn(m, mp[j], size - (mp[j] + 1))
            print('Current Beta2-:', Beta[mp[j]], 'Current Step2-:', stp, 'Rn Candidate2-:', rn_can)
            if rn_can > bst_rn:
                bst_rn = rn_can
                bst_betan = Beta[mp[j]]
            elif stpdiv != 1:
                flag = 3
            elif stpdiv == 1:
                flag = 4
        while flag == 3:
            stp *= stpdiv
            if stp < stpth:
                flag = 4
                break
            if bst_betan - stp > 0:
                Beta[mp[j]] = bst_betan - stp
            rn_can = Rn(m, mp[j], size - (mp[j] + 1))
            print('Current Beta3-:', Beta[mp[j]], 'Current Step3-:', stp, 'Rn Candidate3-:', rn_can)
            if rn_can > bst_rn:
                bst_rn = rn_can
                bst_betan = Beta[mp[j]]
                if stp == stpth:
                    flag = 4
            elif stp == stpth:
                flag = 0
                stp *= stpdiv
    return bst_betan, bst_rn


def FODTO(mp, itr=100):
    global Beta
    m = mp.copy()
    betan_star = [BetamUpRician(i) - 0.01 if RorR(i, size - (i + 1)) == 1 else BetaUpRayleigh(i) - 0.01 for i in
                  range(size)]
    Beta = betan_star.copy()
    rn_bst = betan_star.copy()
    print('Beta:', Beta)
    for i in range(itr):
        prv_Betan = betan_star.copy()
        for j in range(len(mp)):
            m = [x for x in m if x not in [mp[j], size - (mp[j] + 1)]]
            Beta = prv_Betan.copy()
            bst_rn = Rn(m, mp[j], size - (mp[j] + 1))
            print('Interferers:', m, 'Transmitter:', mp[j], 'Receiver:', size - (mp[j] + 1), 'Throughput:', bst_rn)
            betan_star[mp[j]], rn_bst[mp[j]] = LCS(Beta[mp[j]], bst_rn, j, m)
            m = mp.copy()
        print("Optimal Transmission Threshold:", betan_star)
        print("Optimal Throughput:", rn_bst)
        if betan_star == prv_Betan:
            print("Number of Iterations:", i + 1)
            break
    return betan_star, rn_bst


def PODTO(mp, psi=0.5, itr=100):
    global Beta
    updated_last_iter = [False for _ in range(len(mp))]
    updated_this_iter = [False for _ in range(len(mp))]
    m = mp.copy()
    betan_star = [BetamUpRician(i) - 0.01 if RorR(i, size - (i + 1)) == 1 else BetaUpRayleigh(i) - 0.01 for i in
                  range(size)]
    Beta = betan_star.copy()
    rn_bst = betan_star.copy()
    print('Beta:', Beta)
    for i in range(itr):
        prv_Betan = betan_star.copy()
        for j in range(len(mp)):
            if random.random() < psi:
                updated_this_iter[mp[j]] = False
                print('Dropped Node:', mp[j])
                continue
            updated_this_iter[mp[j]] = True
            m = [x for x in m if x not in [mp[j], size - (mp[j] + 1)]]
            Beta = prv_Betan.copy()
            bst_rn = Rn(m, mp[j], size - (mp[j] + 1))
            print('Interferers:', m, 'Transmitter:', mp[j], 'Receiver:', size - (mp[j] + 1), 'Throughput:', bst_rn)
            betan_star[mp[j]], rn_bst[mp[j]] = LCS(Beta[mp[j]], bst_rn, j, m)
            m = mp.copy()
        print("Optimal Transmission Threshold:", betan_star)
        print("Optimal Throughput:", rn_bst)
        if (betan_star == prv_Betan) and (not any((not updated_last_iter[n]) and (not updated_this_iter[n]) for n in range(len(mp)))):
            print("Number of Iterations:", i + 1)
            break
        updated_last_iter = updated_this_iter.copy()
    return betan_star, rn_bst


def LODTO(mp, psi=0.5, itr=100):
    global Beta
    m = mp.copy()
    betan_star = [BetamUpRician(i) - 0.01 if RorR(i, size - (i + 1)) == 1 else BetaUpRayleigh(i) - 0.01 for i in
                  range(size)]
    Beta = betan_star.copy()
    rn_bst = betan_star.copy()
    betam = [betan_star.copy() for _ in range(len(mp))]
    print('Beta:', Beta)
    for i in range(itr):
        prv_Betan = betan_star.copy()
        for j in range(len(mp)):
            m = [x for x in m if x not in [mp[j], size - (mp[j] + 1)]]
            for k in range(len(mp)):
                if random.random() >= psi or j == k:
                    betam[mp[j]][mp[k]] = prv_Betan[mp[k]]
                else:
                    print(f"Node {mp[k]} Dropped for Beta[{mp[j]}][{mp[k]}]")
            Beta = betam[mp[j]].copy()
            print('Beta',[mp[j]],':', betam[mp[j]])
            bst_rn = Rn(m, mp[j], size - (mp[j] + 1))
            print('Interferers:', m, 'Transmitter:', mp[j], 'Receiver:', size - (mp[j] + 1), 'Throughput:', bst_rn)
            betan_star[mp[j]], rn_bst[mp[j]] = LCS(Beta[mp[j]], bst_rn, j, m)
            m = mp.copy()
        print("Optimal Transmission Threshold:", betan_star)
        print("Optimal Throughput:", rn_bst)
        if betan_star == prv_Betan:
            print("Number of Iterations:", i + 1)
            break
    return betan_star, rn_bst


print('LODTO:', LODTO(mp))