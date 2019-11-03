
# coding: utf-8

# In[1]:


import sys 
sys.path.append('../../RenderMan/Builds/LinuxMakefile/build/')
sys.path.append('../../dexed/Builds/Linux/build/')


# In[2]:


import librenderman as rm
import numpy as np
import scipy.optimize as optimize
import librosa


# In[3]:


# Important settings. These are good general ones.
sampleRate = 44100
bufferSize = 512
fftSize = 512

# This will host a VST. It will render the features and audio we need.
engine = rm.RenderEngine(sampleRate, bufferSize, fftSize)

# Load the VST into the RenderEngine.
path = "/mnt/f/Repos/SHD19/Synthesizer/anysynth-master/notebooks/amsynth_vst.so"
engine.load_plugin(path)

# Create a patch generator. We can initialise it to generate the correct
# patches for a given synth by passing it a RenderEngine which has
# loaded a instance of the synthesiser. 
generator = rm.PatchGenerator(engine)

# We can also get a string of information about the
# available parameters.
#print engine.get_plugin_parameters_description()
paramsDict = engine.get_plugin_parameters_description()
paramsDict = {p.split(':')[1].strip():int(p.split(':')[0].strip()) for p in paramsDict.split('\n')[:-1]}
print paramsDict


# In[51]:


# Settings to play a note and extract data from the synth.
midiNote = 80
midiVelocity = 127
noteLength = 0.1
renderLength = 0.1
n_mfcc=20

fixedParams={'filter_vel_sens': 0, 'amp_decay': 0, 'osc2_range': 0,
             'filter_kbd_track': 0, 'filter_env_amount': 1, 'amp_release': 0,
             'lfo_waveform': 0, 'filter_sustain': 1, 'filter_mod_amount': 1,
             'portamento_time': 1, 'filter_cutoff': 0, 'portamento_mode': 1,
             'reverb_damp': 0, 'osc2_detune': 0, 'osc_mix': 0.5, 'osc2_pulsewidth': 0,
             'lfo_freq': 0, 'osc_mix_mode': 1, 'filter_slope': 1, 'distortion_crunch': 0,
             'osc1_pulsewidth': 0, 'amp_sustain': 1, 'osc2_pitch': 0, 'keyboard_mode': 1,
             'filter_type': 0, 'freq_mod_amount': 0, 'reverb_width': 0, 'freq_mod_osc': 0,
             'filter_release': 0, 'reverb_roomsize': 0, 'master_vol': 1, 'osc1_waveform': 0,
             'reverb_wet': 0, 'amp_mod_amount': 0, 'osc2_waveform': 0, 'amp_attack': 0,
             'amp_vel_sens': 0, 'filter_resonance': 0, 'filter_attack': 0,
             'filter_decay': 0, 'osc2_sync': 0}

dynParams=['osc_mix','osc2_pitch','osc2_range','filter_cutoff','osc1_waveform','osc2_waveform','filter_resonance','osc2_sync']

def get_mfccs(audio):
    S = librosa.feature.melspectrogram(y=np.array(audio), sr=sampleRate, n_mels=n_mfcc)
    return S[:,:]


def wrapSynth(xParams, getAudio=False):
    
    for key,value in fixedParams.items():
        engine.override_plugin_parameter(paramsDict[key], value)
        
    for key,value in zip(dynParams,xParams):
        engine.override_plugin_parameter(paramsDict[key], value)
    
    engine.render_patch(midiNote, midiVelocity, noteLength, renderLength)

    # Get the data. Note the audio is automattically made mono, no
    # matter what channel size for ease of use.
    if getAudio:
        audio = engine.get_audio_frames()
        return audio
    
    mfccs = np.mean(engine.get_mfcc_frames(),axis=0)
    return mfccs



# In[70]:


np.random.seed(100)
parameters=np.arange(1,21)
for i in range(0,1000):
    #np.savetxt(X=parameters,newline=",",delimiter=",",fname="/home/bla/synthyZeug/parameters_"+str(i)+".csv")
    tParams = np.random.uniform(size=len(dynParams))
    np.savetxt(fname="/home/bla/synthyZeug/parameters_"+str(i)+".csv",X=tParams)
    target = wrapSynth(tParams, getAudio=True)
    targetMfccs = get_mfccs(target)#wrapSynth(tParams, getAudio=False)
    np.savetxt(X=targetMfccs,fname="/home/bla/synthyZeug/feature_"+str(i)+".csv")


# In[71]:


def metric(xParams):
    test = wrapSynth(xParams)
    return np.linalg.norm(test-targetMfccs)
