import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pretty_midi
import os
import pypianoroll as pproll



def midfile_to_piano_roll(filepath):
    pm = pretty_midi.PrettyMIDI(filepath)
    pr=pm.get_piano_roll()
    df = pd.DataFrame(pr)
    df.to_csv(filepath[:-3]+"csv")
    return filepath[:-3]+"csv"
    
    
def midfile_to_piano_roll_ins(filepath,instrument_n=0):
    pm = pretty_midi.PrettyMIDI(filepath)
    pr=pm.instruments[instrument_n].get_piano_roll()
    df = pd.DataFrame(pr)
    df.to_csv(filepath[:-3]+str(instrument_n)+".csv")
    return filepath[:-3]+str(instrument_n)+".csv"

def load_all_dataset(dirpath):
    return [pd.read_csv(os.path.join(dirpath, file)).values for file in os.listdir(dirpath) if file.endswith(".csv")]

def load_all_dataset_names(dirpath):
    return [file.split('_')[0] for file in os.listdir(dirpath) if file.endswith(".csv")]

def get_max_length(dataset):
    return np.max([x.shape[1] for x in dataset])

def get_numkeys(dataset):
    return np.unique([x.shape[0] for x in dataset])

def visualize_piano_roll(pianoroll_matrix):
    if(pianoroll_matrix.shape[0]==128):
        pianoroll_matrix=pianoroll_matrix.T
    track = pproll.Track(pianoroll=pianoroll_matrix, program=0, is_drum=False, name='piano roll')   
    # Plot the piano-roll
    fig, ax = track.plot()
    plt.show()
    
    