#!/usr/bin/env python
# coding: utf-8
# python from_root_to_h5.py <root_path> <output_path> <sample_type>
# python from_root_to_h5.py ../../Software/pythia8307/HVmodel/test_100k.root ./HVmodel/DA/signal.h5 1

import sys
import ROOT
import h5py
import numpy as np

from tqdm import tqdm

ROOT.gROOT.ProcessLine('.include /usr/local/Delphes-3.4.2/')
ROOT.gROOT.ProcessLine('.include /usr/local/Delphes-3.4.2/external/')
ROOT.gInterpreter.Declare('#include "/usr/local/Delphes-3.4.2/classes/DelphesClasses.h"')
ROOT.gInterpreter.Declare('#include "/usr/local/Delphes-3.4.2/external/ExRootAnalysis/ExRootTreeReader.h"')
ROOT.gInterpreter.Declare('#include "/usr/local/Delphes-3.4.2/external/ExRootAnalysis/ExRootConfReader.h"')
ROOT.gInterpreter.Declare('#include "/usr/local/Delphes-3.4.2/external/ExRootAnalysis/ExRootTask.h"')
ROOT.gSystem.Load("/usr/local/Delphes-3.4.2/install/lib/libDelphes")

def create_dataset(f, nevent, MAX_JETS):

    f.create_dataset('J1/MASK', (nevent, MAX_JETS), maxshape=(None, MAX_JETS), dtype='|b1')
    f.create_dataset('J1/pt', (nevent, MAX_JETS), maxshape=(None, MAX_JETS), dtype='<f4')
    f.create_dataset('J1/eta', (nevent, MAX_JETS), maxshape=(None, MAX_JETS), dtype='<f4')
    f.create_dataset('J1/phi', (nevent, MAX_JETS), maxshape=(None, MAX_JETS), dtype='<f4')

    f.create_dataset('J2/MASK', (nevent, MAX_JETS), maxshape=(None, MAX_JETS), dtype='|b1')
    f.create_dataset('J2/pt', (nevent, MAX_JETS), maxshape=(None, MAX_JETS), dtype='<f4')
    f.create_dataset('J2/eta', (nevent, MAX_JETS), maxshape=(None, MAX_JETS), dtype='<f4')
    f.create_dataset('J2/phi', (nevent, MAX_JETS), maxshape=(None, MAX_JETS), dtype='<f4')

    f.create_dataset('EVENT/Mjj', (nevent,), maxshape=(None,), dtype='<f4')
    f.create_dataset('EVENT/signal', (nevent,), maxshape=(None,), dtype='<i8')

def write_dataset(file, index, data: dict):

    for key, value in data.items():
        file[key][index] = value

def get_dataset_keys(f):
    keys = []
    f.visit(lambda key : keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
    return keys

def resize_h5(file_path, nevent):

    with h5py.File(file_path,'r+') as f:
        keys = get_dataset_keys(f)
        for key in keys:
            shape = list(f[key].shape)
            shape[0] = nevent
            f[key].resize(shape)
    print(f'{file_path} resize to {nevent}')

def Mjets(*arg):
    # arg: list of jets
    # return: invariant mass of jets
    e_tot, px_tot, py_tot, pz_tot = 0, 0, 0, 0
    
    for jet in arg:
        pt, eta, phi, m = jet[0], jet[1], jet[2], jet[3]
        
        px, py, pz = pt*np.cos(phi), pt*np.sin(phi), pt*np.sinh(eta)
        e = np.sqrt(m**2 + px**2 + py**2 + pz**2)
        
        px_tot += px
        py_tot += py
        pz_tot += pz
        e_tot += e
    
    return np.sqrt(e_tot**2 - px_tot**2 - py_tot**2 - pz_tot**2)

def get_pt_eta_phi(constituents):
    pts, etas, phis  = [], [], []
    
    for consti in constituents:
        try:
            pts.append(consti.PT)
            etas.append(consti.Eta)
            phis.append(consti.Phi)

        except:
            pts.append(consti.ET)
            etas.append(consti.Eta)
            phis.append(consti.Phi)
            
    pts = np.array(pts)
    etas = np.array(etas)
    phis = np.array(phis)

    return pts, etas, phis

def from_root_to_h5(tree, output_path, nevent, signal):
    # Select events and save the jets information to h5 file

    # Hidden Valley model selection
    # 1. 2 jets
    # 2. pT > 750 GeV
    # 3. |eta| < 2.0
    # 4. 4300 < Mjj < 5900 GeV

    with h5py.File(output_path, 'w') as f_out:

        MAX_JETS = 300
        create_dataset(f_out, nevent, MAX_JETS)

        event_index = 0
        for event_id, event in tqdm(enumerate(tree)):

            if event.Jet_size < 2:
                continue

            if event.Jet[1].PT < 750:
                continue

            if abs(event.Jet[0].Eta) > 2.0 or abs(event.Jet[1].Eta) > 2.0:
                continue

            jets = [[event.Jet[i].PT, event.Jet[i].Eta, event.Jet[i].Phi, event.Jet[i].Mass] for i in range(2)]
            mjj = Mjets(*jets)

            if mjj < 4300 or mjj > 5900:
                continue
            
            # get jet constituents
            constituents = [consti for consti in event.Jet[0].Constituents if consti != 0]
            n_consti_1 = len(constituents)
            PT1, Eta1, Phi1 = get_pt_eta_phi(constituents)

            constituents = [consti for consti in event.Jet[1].Constituents if consti != 0]
            n_consti_2 = len(constituents)
            PT2, Eta2, Phi2 = get_pt_eta_phi(constituents)

            if n_consti_1 < 5 or n_consti_2 < 5:
                continue
             
            # 準備寫入資料
            data_dict = {
                'J1/MASK': np.arange(MAX_JETS)<n_consti_1,
                'J1/pt': PT1[:MAX_JETS] if n_consti_1>MAX_JETS else np.pad(PT1, (0,MAX_JETS-n_consti_1)),
                'J1/eta': Eta1[:MAX_JETS] if n_consti_1>MAX_JETS else np.pad(Eta1, (0,MAX_JETS-n_consti_1)),
                'J1/phi': Phi1[:MAX_JETS] if n_consti_1>MAX_JETS else np.pad(Phi1, (0,MAX_JETS-n_consti_1)),

                'J2/MASK': np.arange(MAX_JETS)<n_consti_2,
                'J2/pt': PT2[:MAX_JETS] if n_consti_2>MAX_JETS else np.pad(PT2, (0,MAX_JETS-n_consti_2)),
                'J2/eta': Eta2[:MAX_JETS] if n_consti_2>MAX_JETS else np.pad(Eta2, (0,MAX_JETS-n_consti_2)),
                'J2/phi': Phi2[:MAX_JETS] if n_consti_2>MAX_JETS else np.pad(Phi2, (0,MAX_JETS-n_consti_2)),
            
                'EVENT/Mjj': mjj,
                'EVENT/signal': signal,
            }

            write_dataset(f_out, event_index, data_dict)
            event_index += 1

    resize_h5(output_path, event_index)

def main():

    root_file = sys.argv[1]
    output_path = sys.argv[2]
    sample_type = int(sys.argv[3])

    f = ROOT.TFile(root_file)
    tree_s = f.Get("Delphes")

    # Get number of events
    nevent = tree_s.GetEntries()
    print('Number of events: ', nevent)

    from_root_to_h5(tree_s, output_path, nevent, sample_type)

if __name__ == '__main__':
    main()