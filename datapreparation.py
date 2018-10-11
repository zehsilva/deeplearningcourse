from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pretty_midi
import os
import pypianoroll as pproll
import sys
import argparse


def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.

    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.

    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.

    '''
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm



def midfile_to_piano_roll(filepath):
    """ convert a mid file to a piano roll matrix and saves it in a csv file
        input: path to mid file
        output: path to piano_roll csv file
    """
    pm = pretty_midi.PrettyMIDI(filepath)
    pr=pm.get_piano_roll()
    df = pd.DataFrame(pr)
    df.to_csv(filepath[:-3]+"csv")
    return filepath[:-3]+"csv"

def piano_roll_to_mid_file(pianoroll_matrix,fname):
    """ input: piano roll matrix with shape (number of notes, time steps)
        output: string with path to mid file
    """
    piano_roll_to_pretty_midi(pianoroll_matrix).write(fname)
    return os.path.join(os.getcwd(),fname)
    
    
def midfile_to_piano_roll_ins(filepath,instrument_n=0):
    """ convert mid file to piano_roll csv matrix, but selecting a specific instrument in the mid file
        input: path to mid file, intrument to select in midfile
        output: path to piano_roll csv file
    """
    pm = pretty_midi.PrettyMIDI(filepath)
    pr=pm.instruments[instrument_n].get_piano_roll()
    df = pd.DataFrame(pr)
    df.to_csv(filepath[:-3]+str(instrument_n)+".csv")
    return filepath[:-3]+str(instrument_n)+".csv"

def load_all_dataset(dirpath):
    """ given a diretory finds all the csv in the diretory an load them as numpy arrays
        input: path a diretory
        output: list of numpy arrays
    """
    return [pd.read_csv(os.path.join(dirpath, file)).values for file in os.listdir(dirpath) if file.endswith(".csv")]

def load_all_dataset_names(dirpath):
    """ given a diretory finds all the csv in the diretory an split the first part
        of the name of the file to return as a tag for the associated numpy array
        input: path a diretory
        output: list of strings
    """
    return [file.split('_')[0] for file in os.listdir(dirpath) if file.endswith(".csv")]

def get_max_length(dataset):
    """ find the maximum length of piano roll matrices in a list of matrices
        input: list of numpy 2d arrays
        output: maximun shape[1] of the list of arrays
        
    """
    return np.max([x.shape[1] for x in dataset])

def get_numkeys(dataset):
    """ return all the number of keys present in the piano roll matrices 
    (typically it should all have the same number of keys and it should be 128)
    input: list of numpy 2d arrays
    output: unique shape[0] of the list of arrays
        
    """
    return np.unique([x.shape[0] for x in dataset])

def visualize_piano_roll(pianoroll_matrix):
    """ input: piano roll matrix with shape (number of notes, time steps)
        effect: generates a nice graph with the piano roll visualization
    """
    if(pianoroll_matrix.shape[0]==128):
        pianoroll_matrix=pianoroll_matrix.T
    track = pproll.Track(pianoroll=pianoroll_matrix, program=0, is_drum=False, name='piano roll')   
    # Plot the piano-roll
    fig, ax = track.plot()
    plt.show()
    
    