#!/usr/bin/env python
'''
this script takes in raw or smeared MC files in event-block format
and generates files of physics-relevant features in event-line format
'''

import sys
import math
from array import array
import random
import numpy as np
from ROOT import TVector3, TLorentzVector

## defining vectors
p3g     = TVector3()
p3k     = TVector3()
p3L     = TVector3()
p3p     = TVector3()
p3pi    = TVector3()
p3mu    = TVector3()
p3nu    = TVector3()
p3miss  = TVector3()
capA    = TVector3()
capB    = TVector3()
capD    = TVector3()
capE    = TVector3()
c_vec   = TVector3()
a_vec   = TVector3()
b_vec   = TVector3()

v1_gen  = TVector3()
v2_gen  = TVector3()
v3_gen  = TVector3()
v1_meas = TVector3()
v2_meas = TVector3()
v3_meas = TVector3()

p4g     = TLorentzVector()
p4targ  = TLorentzVector()
p4k     = TLorentzVector()
p4L     = TLorentzVector()
p4p     = TLorentzVector()
p4pi    = TLorentzVector()
p4mu    = TLorentzVector()
p4nu    = TLorentzVector()
p4miss  = TLorentzVector()

files = sys.argv[1:]

## some masses in GeV/c^2
m_gamma    = 0.0
m_target   = 0.938272
m_kplus    = 0.493677
m_lam      = 1.115683
m_pion     = 0.13957
m_muon     = 0.105658
m_proton   = 0.938272
m_neutrino = 0.0
m_electron = 0.000511

for infil in files:
    # set event target class
    evt_targ = 0
    if "sl_mu" in infil:
        evt_targ = 1

    p3g.SetXYZ(0,0,0)
    p3k.SetXYZ(0,0,0)
    p3L.SetXYZ(0,0,0)
    p3p.SetXYZ(0,0,0)
    p3pi.SetXYZ(0,0,0)
    p3mu.SetXYZ(0,0,0)
    p3nu.SetXYZ(0,0,0)
    p3miss.SetXYZ(0,0,0)

    v1_gen.SetXYZ(0,0,0)
    v2_gen.SetXYZ(0,0,0)
    v3_gen.SetXYZ(0,0,0)
    v1_meas.SetXYZ(0,0,0)
    v2_meas.SetXYZ(0,0,0)
    v3_meas.SetXYZ(0,0,0)

    print("\n" + infil)
    outfil = './features_files/' + infil.split('/')[-1].replace('.ascii','.feats.ascii')
    print(outfil)

    fil = open(infil, 'r')
    ofil = open(outfil, 'w')
    lin = fil.readline()
    lin_list = lin.split(None)

    nevts = 0
    while len(lin) > 0: # and nevts < 1000:
        outfeats = [evt_targ]
        header = ['target']
        nevts += 1
        if lin_list[0] == "10000":
            p3g.SetXYZ(0, 0, float(lin_list[2]))

        lin = fil.readline()
        lin_list = lin.split(None)

        v3_found = 0
        while len(lin_list) > 5:
            if len(lin_list) == 8:
                if lin_list[1] == "11":
                    p3k.SetXYZ(float(lin_list[5]), float(lin_list[6]), float(lin_list[7]))
                elif lin_list[1] == "18":
                    p3L.SetXYZ(float(lin_list[5]), float(lin_list[6]), float(lin_list[7]))
                elif lin_list[1] == "14":
                    p3p.SetXYZ(float(lin_list[5]), float(lin_list[6]), float(lin_list[7]))
                elif lin_list[1] == "9":
                    p3pi.SetXYZ(float(lin_list[5]), float(lin_list[6]), float(lin_list[7]))
                elif lin_list[1] == "4":
                    p3nu.SetXYZ(float(lin_list[5]), float(lin_list[6]), float(lin_list[7]))
                elif lin_list[1] == "6":
                    p3mu.SetXYZ(float(lin_list[5]), float(lin_list[6]), float(lin_list[7]))
            elif len(lin_list) == 6:
                if lin_list[0] == "1":
                    v1_gen.SetXYZ(float(lin_list[3]), float(lin_list[4]), float(lin_list[5]))
                elif lin_list[0] == "2":
                    v2_gen.SetXYZ(float(lin_list[3]), float(lin_list[4]), float(lin_list[5]))
                elif lin_list[0] == "3":
                    v3_found = 1
                    v3_gen.SetXYZ(float(lin_list[3]), float(lin_list[4]), float(lin_list[5]))

            lin = fil.readline()
            lin_list = lin.split(None)

        p4g.SetVectM(p3g,0.0)
        p4targ.SetXYZT(0,0,0,m_proton)
        p4L.SetVectM(p3L,m_lam)
        p4k.SetVectM(p3k,m_kplus)
        p4p.SetVectM(p3p,m_proton)
        p4mu.SetVectM(p3mu,m_muon)

        p4nu = p4g + p4targ - p4k - p4p - p4mu
        p3nu = p4nu.Vect()
        p4tot_i = p4g + p4targ
        p4tot_f = p4k + p4p + p4mu + p4nu

        if v3_found == 0:
            v3_gen = v2_gen

        outfeats.append(p3g.Mag())
        header.append('p_gamma')
        outfeats.append(p3k.CosTheta())
        header.append('costheta_k')
        outfeats.append(p4k.E()) #e_k
        header.append('e_k')
        outfeats.append(p3k.Mag()) #p_k
        header.append('p_k')
        outfeats.append(p3L.Mag()) #p_l, found just using the components in the MC
        header.append('p_Lam')
        outfeats.append(p4L.E())
        header.append('e_Lam')
        outfeats.append(p3p.Mag()) #p_p
        header.append('p_prot')
        #outfeats.append(p3pi.Mag())
        outfeats.append(p3nu.Mag()) #p_n
        header.append('p_nu')
        outfeats.append(p3mu.Mag()) #p_m
        header.append('p_mu')

        pmut = p3mu - p3mu.Dot(p3L)/p3L.Mag()*p3L.Unit()
        pnut = p3nu - p3nu.Dot(p3L)/p3L.Mag()*p3L.Unit()
        ppt = p3p - p3p.Dot(p3L)/p3L.Mag()*p3L.Unit()
        outfeats.append(pmut.Mag())
        header.append('ptL_mu')
        outfeats.append(ppt.Mag())
        header.append('ptL_prot')
        t = pmut.Mag() - ppt.Mag()
        outfeats.append(t)
        header.append('t_prot_mu')
        outfeats.append((pmut.Unit().Dot(ppt.Unit())))
        header.append('ptL_mu_prot_dot')
        outfeats.append((pnut.Unit().Dot(ppt.Unit())))
        header.append('ptL_nu_prot_dot')
        outfeats.append((p4p + p4mu).M())
        header.append('inv_prot_mu')
        outfeats.append((p4g + p4targ - p4k - p4p).M())
        header.append('missing_mass')
        outfeats.append((p4g + p4targ - p4k).M())
        header.append('missing_mass_off_k')

        ### primary vertexing
        a_vec.SetXYZ(0,0,1.0)
        b_vec = p3k.Unit()
        capA = v1_gen #.SetXYZ(0,0,0)
        #capA.SetXYZ(v1_gen.X(),v1_gen.Y(),0)
        capB = v1_gen
        c_vec = capB - capA
        capD = a_vec * (-1.0*a_vec.Dot(b_vec) * b_vec.Dot(c_vec)
                        + a_vec.Dot(c_vec) * b_vec.Dot(b_vec))
        capD *= 1.0/((a_vec.Dot(a_vec) * b_vec.Dot(b_vec)
                      - a_vec.Dot(b_vec) * a_vec.Dot(b_vec)))
        capD += capA

        capE = b_vec * (a_vec.Dot(b_vec) * a_vec.Dot(c_vec)
                        - b_vec.Dot(c_vec) * a_vec.Dot(a_vec))
        capE *= 1.0/((a_vec.Dot(a_vec) * b_vec.Dot(b_vec)
                      - a_vec.Dot(b_vec) * a_vec.Dot(b_vec)))
        capE += capB

        v1_meas = 0.5*(capD + capE)
        #outfeats.append((v1_meas - v1_gen).Mag())

        ### Lambda vertexing
        a_vec = p3mu.Unit()
        b_vec = p3p.Unit()
        capA = v2_gen
        capB = v3_gen
        c_vec = capB - capA
        capD = a_vec * (-1.0*a_vec.Dot(b_vec) * b_vec.Dot(c_vec)
                        + a_vec.Dot(c_vec) * b_vec.Dot(b_vec))
        capD *= 1.0/((a_vec.Dot(a_vec) * b_vec.Dot(b_vec)
                      - a_vec.Dot(b_vec) * a_vec.Dot(b_vec)))
        capD += capA

        capE = b_vec * (a_vec.Dot(b_vec) * a_vec.Dot(c_vec)
                        - b_vec.Dot(c_vec) * a_vec.Dot(a_vec))
        capE *= 1.0/((a_vec.Dot(a_vec) * b_vec.Dot(b_vec)
                      - a_vec.Dot(b_vec) * a_vec.Dot(b_vec)))
        capE += capB

        v2_meas = 0.5*(capD + capE)
        #outfeats.append((v2_meas - v2_gen).Mag())
        #outfeats.append((v2_gen - v1_gen).Mag())
        #outfeats.append((v3_gen - v2_gen).Mag())
        outfeats.append((v2_meas - v1_meas).Mag())
        header.append('Lam_flight_len')
        #print(v2_gen.X(),v2_gen.Y(),v2_gen.Z())
        #print(v2_meas.X(),v2_meas.Y(),v2_meas.Z())
        #print()

        ### distance of v2 off lambda flight path
        offlam = ((v1_meas - v2_meas) - (p3L.Unit().Dot((v1_meas - v2_meas)))*p3L.Unit()).Mag()
        outfeats.append(offlam)
        header.append('Lam_vert_discrep')

        for i in range(len(outfeats)):
            if (abs(outfeats[i]) < 0.01 or abs(outfeats[i]) > 100) and outfeats[i] != 0:
                outfeats[i] = str('{:0.4e}'.format(outfeats[i]))
            else:
                outfeats[i] = str(round(outfeats[i],5))

        #outfeats = [str(round(x,5)) for x in outfeats]
        #outfeats = [str('{:0.5e}'.format(x)) for x in outfeats]

        outstr = ','.join(outfeats)
        headstr = ','.join(header)

        if nevts == 1:
            ofil.write(headstr + '\n')

        ofil.write(outstr + '\n')
        #print(outstr)

    fil.close()
    ofil.close()

# Histogram Canvas

# st_egamma = THStack('st_egamma','st_egamma')
# st_ek = THStack('st_ek','st_ek')
# st_el = THStack('st_el','st_el')
# st_ktheta = THStack('st_ktheta','st_ktheta')
# for i in range(n_tags):
#     h_egamma[i].SetFillColorAlpha(hist_cols[i],0.30)
#     st_egamma.Add(h_egamma[i])
#
#     h_ek[i].SetFillColorAlpha(hist_cols[i],0.30)
#     st_ek.Add(h_ek[i])
#
#     h_el[i].SetFillColorAlpha(hist_cols[i],0.30)
#     st_el.Add(h_el[i])
#
#     h_ktheta[i].SetFillColorAlpha(hist_cols[i],0.30)
#     st_ktheta.Add(h_ktheta[i])
#
# energy = TCanvas('energy','energy',10,10,700,650)
# energy.Divide(2,2)
# energy.cd(1)
# st_egamma.Draw('nostack')
# energy.cd(2)
# st_ek.Draw('nostack')
# energy.cd(3)
# st_el.Draw('nostack')
# energy.cd(1)
# st_ktheta.Draw('nostack')
# gPad.Update()
#
# ####################
# st_momk = THStack("st_momk","st_momk")
# st_moml = THStack("st_moml","st_moml")
# for i in range(n_tags):
#     h_momk[i].SetFillColorAlpha(hist_cols[i],0.30)
#     st_momk.Add(h_momk[i])
#     h_moml[i].SetFillColorAlpha(hist_cols[i],0.30)
#     st_moml.Add(h_moml[i])
#
# momentum_same = TCanvas('momentum_same','momentum_same',30,30,700,650)
# momentum_same.Divide(1,2)
# momentum_same.cd(1)
# st_momk.Draw('nostack')
# momentum_same.cd(2)
# st_moml.Draw('nostack')
# gPad.Update()
#
# ####################
# st_mompi = THStack('st_mompi','st_mompi')
# st_momn = THStack('st_momn','st_momn')
# st_momm = THStack('st_momm','st_momm')
# st_momp = THStack('st_momp','st_momp')
# st_misspoffkp = THStack('st_misspoffkp','st_misspoffkp')
# leg = TLegend(0.50,0.60,.88,.88)
# for i in range(n_tags):
#     h_mompi[i].SetFillColorAlpha(hist_cols[i],0.30)
#     st_mompi.Add(h_mompi[i])
#     h_momn[i].SetFillColorAlpha(hist_cols[i],0.30)
#     st_momn.Add(h_momn[i])
#     h_momm[i].SetFillColorAlpha(hist_cols[i],0.30)
#     if i == 0:
#         leg.AddEntry(h_momn[i],'SLD signal')
#     elif i == 1:
#         leg.AddEntry(h_momn[i],'HADT background')
#     st_momm.Add(h_momm[i])
#     h_momp[i].SetFillColorAlpha(hist_cols[i],0.30)
#     st_momp.Add(h_momp[i])
#     h_misspoffkp[i].SetFillColorAlpha(hist_cols[i],0.30)
#     st_misspoffkp.Add(h_misspoffkp[i])
#
# momentum_diff = TCanvas('momentum_diff','momentum_diff',40,40,700,650)
# momentum_diff.Divide(2,3)
# momentum_diff.cd(1)
# st_mompi.Draw('nostack')
# #momentum_diff.cd(2)
# #st_momn.Draw('nostack')
# momentum_diff.cd(3)
# st_momm.Draw('nostack')
# momentum_diff.cd(4)
# st_momp.Draw('nostack')
# momentum_diff.cd(5)
# st_misspoffkp.Draw('nostack')
# gPad.Update()
#
# #######################
# st_pmut = THStack('st_pmut','st_pmut')
# st_ppt = THStack('st_ppt','st_ppt')
# st_t = THStack('st_t','st_t')
# st_invpmu = THStack('st_invpmu','st_invpmu')
# for i in range(n_tags):
#     h_pmut[i].SetFillColorAlpha(hist_cols[i],0.30)
#     st_pmut.Add(h_pmut[i])
#     h_ppt[i].SetFillColorAlpha(hist_cols[i],0.30)
#     st_ppt.Add(h_ppt[i])
#     h_t[i].SetFillColorAlpha(hist_cols[i],0.30)
#     st_t.Add(h_t[i])
#     h_invpmu[i].SetFillColorAlpha(hist_cols[i],0.30)
#     st_invpmu.Add(h_invpmu[i])
#
# pt = TCanvas('pt','pt',50,50,700,650)
# pt.Divide(2,2)
# pt.cd(1)
# st_pmut.Draw('nostack')
# pt.cd(2)
# st_ppt.Draw('nostack')
# #pt.cd(3)
# #st_t.Draw('nostack')
# gPad.Update()
#
# p2 = TCanvas('p2','p2',60,60,1000,400)
# p2.Divide(n_tags,1,0,0)
# for i in range(n_tags):
#     p2.cd(i+1)
#     h2_t_pnu[i].GetYaxis().SetTitle("t (GeV/c)")
#     h2_t_pnu[i].GetXaxis().SetTitle("neutrino momentum (GeV/c)")
#     h2_t_pnu[i].Draw('colz')
# gPad.Update()
#
# v2 = TCanvas('v2','v2',70,70,700,650)
# v2.Divide(1,n_tags,0,0)
# for i in range(n_tags):
#     v2.cd(i+1)
#     h2_v2rhoz[i].GetXaxis().SetTitle("z_{2} (cm)")
#     h2_v2rhoz[i].GetYaxis().SetTitle("#rho_{2} (cm)")
#     h2_v2rhoz[i].Draw('colz')
# gPad.Update()
#
#
# st_v2pocadiff = THStack('st_v2pocadiff','st_v2pocadiff')
# st_v1pocadiff = THStack('st_v1pocadiff','st_v1pocadiff')
# st_decaylen = THStack('st_decaylen','st_decaylen')
# st_decaylenmeas = THStack('st_decaylenmeas','st_decaylenmeas')
# st_pidecaylen = THStack('st_pidecaylen','st_pidecaylen')
# st_v2offlam = THStack('st_v2offlam','st_v2offlam')
# for i in range(n_tags):
#     h_v1pocadiff[i].SetFillColorAlpha(hist_cols[i],0.30)
#     st_v1pocadiff.Add(h_v1pocadiff[i])
#     h_v2pocadiff[i].SetFillColorAlpha(hist_cols[i],0.30)
#     st_v2pocadiff.Add(h_v2pocadiff[i])
#     h_decaylen[i].SetFillColorAlpha(hist_cols[i],0.30)
#     st_decaylen.Add(h_decaylen[i])
#     h_decaylenmeas[i].SetFillColorAlpha(hist_cols[i],0.30)
#     st_decaylenmeas.Add(h_decaylenmeas[i])
#     h_pidecaylen[i].SetFillColorAlpha(hist_cols[i],0.30)
#     st_pidecaylen.Add(h_pidecaylen[i])
#     h_v2offlam[i].SetFillColorAlpha(hist_cols[i],0.30)
#     st_v2offlam.Add(h_v2offlam[i])
#
# pvert = TCanvas('pvert','pvert',80,80,700,650)
# pvert.Divide(2,3)
# pvert.cd(1)
# #gPad.SetLogx()
# st_v1pocadiff.Draw('nostack')
# pvert.cd(2)
# #gPad.SetLogx()
# st_v2pocadiff.Draw('nostack')
# pvert.cd(3)
# #gPad.SetLogx()
# st_decaylen.Draw('nostack')
# pvert.cd(4)
# #gPad.SetLogx()
# # f_decaylen = TF1("f_decaylen", "[0]*exp(-[1]*x)", 0.0, 100)
# # f_decaylen.SetLineColor(2)
# # f_decaylen.SetLineWidth(1)
# # f_decaylen.SetParameter(0,5000)
# # f_decaylen.SetParameter(1,0.1)
# # h_decaylenmeas[0].Fit("f_decaylen")
# ####st_decaylenmeas.Draw('nostack')
# # f_decaylen.Draw('same')
# pvert.cd(5)
# gPad.SetLogy()
# st_pidecaylen.Draw('nostack')
# pvert.cd(6)
# #gPad.SetLogx()
# #st_v2offlam.Draw('nostack')
#
# gPad.Update()
#
# poster = TCanvas('poster','poster',90,90,700,325)
# poster.Divide(2,1)
# poster.cd(1)
# st_momn.Draw('nostack')
# st_momn.SetTitle()
# st_momn.GetXaxis().SetTitle("neutrino momentum (GeV/c)")
# leg.Draw('same')
# poster.Modified()
# poster.cd(2)
# st_t.Draw('nostack')
# st_t.GetXaxis().SetTitle("t (GeV/c)")
# st_t.SetTitle()
# poster.Modified()
# gPad.Update()
#
# poster2 = TCanvas('poster2','poster2',100,100,700,325)
# poster2.Divide(2,1)
# poster2.cd(1)
# st_decaylenmeas.Draw('nostack')
# st_decaylenmeas.GetXaxis().SetTitle("reocn. #Lambda flight length (cm)")
# st_decaylenmeas.SetTitle()
# poster2.Modified()
# poster2.cd(2)
# gPad.SetLogy()
# st_v2offlam.Draw('nostack')
# st_v2offlam.GetXaxis().SetTitle("#Lambda vertex discrepancy (cm)")
# st_v2offlam.SetTitle()
# poster2.Modified()
# gPad.Update()
#
# if __name__ == '__main__':
#   rep = ''
#   while not rep in [ 'q', 'Q' ]:
#     rep = input( 'enter "q" to quit: ' )
#     if 1 < len(rep):
#       rep = rep[0]
