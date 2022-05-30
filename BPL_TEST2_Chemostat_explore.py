# Figure - Simulation of chemostat reactor
#          with functions added to facilitate explorative simulation work
#
# Author: Jan Peter Axelsson
#------------------------------------------------------------------------------------------------------------------
# 2020-01-28 - Use Docker JModelica 2.4 compiled FMU and run in Python3 with PyFMI
#            - Import platform and locale (for later use with OpenModelica FMU)
#            - Fix print() 
#            - Fix np.nan for Jupyter notebook
#            - Move plt.show() from newplot() to simu()
#            - Need to eliminate use of Trajectory since imported from pyjmi
#            - Change newplot and simu using objectoriented diagrams
# 2020-01-28 - Added system print of system information
# 2020-01-29 - Improved check of platform to adapt code for Windows/Linux in dialog
# 2020-02-02 - Now only for Python 3, i.e. special Python 2 script for compilation
# 2020-02-07 - Polish code and moved the two function that describe framework up
# 2020-02-12 - Tested with JModelica 2.14 and seems ok
# 2020-03-05 - Updated name for FMU
# 2020-03-16 - Indluced in system_info() information if FMU is ME or CS
#------------------------------------------------------------------------------------------------------------------
# 2020-05-02 - Adapt for Jupyter
#            - Change name and type of FMU
#            - Suppress printing of run time statistics - opts
#            - Suppress plt.show in simu()
# 2020-05-10 - Corrected plot of phaseplane in simu()
# 2020-07-01 - Updated opts for Windows to make it silent
#------------------------------------------------------------------------------------------------------------------
# 2021-05-08 - Adapted for BPL ver 2.0.5 and updated BPL interface
# 2021-05-19 - Default value given for disp()
# 2021-05-21 - Adapted for Linux also 
# 2021-05-25 - Change to parDict and parLocation and use par() for parDict.update()
# 2021-06-05 - Updated init() to handle also strings
#------------------------------------------------------------------------------------------------------------------
# 2022-04-08 - Updated for BPL ver 2.0.7 and FMU_explore 0.9.0
# 2022-04-20 - Added a new plot
# 2022-05-29 - Updated to FMU-explore 0.9.1 - describe_general() to handle boolean parameters
# 2022-05-30 - Introduced legends for ax3
# 2022-05-30 - Updated describe() with handling of cstrProdMax doc-string
#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------
#  Framework
#------------------------------------------------------------------------------------------------------------------

# Setup framework
import sys
import platform
import locale
import numpy as np 
import matplotlib.pyplot as plt 
from pyfmi import load_fmu
from pyfmi.fmi import FMUException
from itertools import cycle
from importlib_metadata import version   # included in future Python 3.8

# Set the environment - for Linux a JSON-file in the FMU is read
if platform.system() == 'Linux': locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

#------------------------------------------------------------------------------------------------------------------
#  Setup application FMU
#------------------------------------------------------------------------------------------------------------------

# Define model file name and class name              
#model_name = 'BPL_TEST2.Batch' 
#application_file = 'BPL_TEST2.mo'
#library_file = 'W:/BPL/package.mo'

# Provde the right FMU and load for different platforms in user dialogue:
global fmu_model, model, opts
if platform.system() == 'Windows':
   print('Windows - run FMU pre-compiled JModelica 2.14')
   fmu_model ='BPL_TEST2_Chemostat_windows_jm_cs.fmu'        
   model = load_fmu(fmu_model, log_level=0)
   opts = model.simulate_options()
   opts['silent_mode'] = True
elif platform.system() == 'Linux':
   print('Linux - run FMU pre-compiled JModelica 2.4')
   fmu_model ='BPL_TEST2_Chemostat_linux_jm_cs.fmu'        
   model = load_fmu(fmu_model, log_level=0)
   opts = model.simulate_options()
   opts['silent_mode'] = True
else:    
   print('There is no FMU for this platform')

# Simulation time
global simulationTime; simulationTime = 60.0

# Dictionary of time discrete states
timeDiscreteStates = {} 

# Define a minimal compoent list of the model as a starting point for describe('parts')
component_list_minimum = ['bioreactor', 'bioreactor.culture']

#------------------------------------------------------------------------------------------------------------------
#  Specific application constructs: stateDict, parDict, diagrams, newplot(), describe()
#------------------------------------------------------------------------------------------------------------------

# Create stateDict that later will be used to store final state and used for initialization in 'cont':
#stateDict = model.get_states_list()
global stateDict

# Create dictionaries parDict and parLocation
global parDict; parDict = {}
parDict['V_0'] = 1.0
parDict['VX_0'] = 1.0
parDict['VS_0'] = 30.0

parDict['Y'] = 0.5
parDict['qSmax'] = 0.75
parDict['Ks'] = 0.1

parDict['S_in'] = 30.0
parDict['feedtank.V_0'] = 100.0

parDict['t0'] = 0.0
parDict['F0'] = 0.0
parDict['t1'] = 10.0
parDict['F1'] = 0.20
parDict['t2'] = 999.0
parDict['F2'] = 0.20
parDict['t3'] = 1000.0
parDict['F3'] = 0.20

global parLocation; parLocation = {}
parLocation['V_0'] = 'bioreactor.V_0'
parLocation['VX_0'] = 'bioreactor.m_0[1]' 
parLocation['VS_0'] = 'bioreactor.m_0[2]' 

parLocation['Y'] = 'bioreactor.culture.Y'
parLocation['qSmax'] = 'bioreactor.culture.qSmax'
parLocation['Ks'] = 'bioreactor.culture.Ks'

parLocation['S_in'] = 'feedtank.c_in[2]'
parLocation['feedtank.V_0'] = 'feedtank.V_0'
parLocation['t0'] = 'dosagescheme.table[1,1]'
parLocation['F0'] = 'dosagescheme.table[1,2]'
parLocation['t1'] = 'dosagescheme.table[2,1]'
parLocation['F1'] = 'dosagescheme.table[2,2]'
parLocation['t2'] = 'dosagescheme.table[3,1]'
parLocation['F2'] = 'dosagescheme.table[3,2]'
parLocation['t3'] = 'dosagescheme.table[4,1]'
parLocation['F3'] = 'dosagescheme.table[4,2]'

# Create list of diagrams to be plotted by simu()
global diagrams
diagrams = []

# Define standard diagrams
def newplot(title='Chemostat cultivation', plotType='TimeSeries'):
   """ Standard plot window, two possibilities:
         diagram = 'TimeSeries' default
         diagram = 'PhasePlane' 
       and plot main title
         title = '' """
       
   # Transfer of argument to global variable
   global ax1, ax2, ax3, ax4, ax5 
   global ax11, ax21, ax31, ax41, ax12, ax22

   # Reset pens
   setLines()
    
   # Plot diagram
   if plotType == 'TimeSeries':

      plt.figure()
      ax1 = plt.subplot(4,1,1)
      ax2 = plt.subplot(4,1,2)       
      ax3 = plt.subplot(4,1,3)
      ax4 = plt.subplot(4,1,4)

      ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('S [g/L]')
    
      ax2.grid()
      ax2.set_ylabel('X [g/L]')
 
      ax3.grid()
      ax3.set_ylabel('F*X [g/h]') 
 
      ax4.grid()
      ax4.set_ylabel('D=F/V, mu [1/h]')
      ax4.set_xlabel('Time [h]') 
      
      # List of commands to be executed by simu() after a simulation  
      diagrams.clear()
      diagrams.append("ax1.plot(t,sim_res['bioreactor.c[2]'],color='b',linestyle=linetype)")
      diagrams.append("ax2.plot(t,sim_res['bioreactor.c[1]'],color='b',linestyle=linetype)")   
      diagrams.append("ax3.plot(t,sim_res['bioreactor.inlet[1].F']*sim_res['bioreactor.c[1]'],color='b',linestyle=linetype)") 
      diagrams.append("ax3.plot([0, simulationTime], [cstrProdMax(model), cstrProdMax(model)], color='r',linestyle=linetype)")
      diagrams.append("ax3.legend(['FX','FX_max'])")   
      diagrams.append("ax4.plot(t,sim_res['D'],color='b',linestyle=linetype)") 
      diagrams.append("ax4.plot(t,sim_res['bioreactor.culture.q[1]'],color='r',linestyle=linetype)") 
      diagrams.append("ax4.legend(['D','mu'])")    
   
   elif plotType == 'TimeSeries2':

      plt.figure()
      ax1 = plt.subplot(5,1,1)
      ax2 = plt.subplot(5,1,2)       
      ax3 = plt.subplot(5,1,3)
      ax4 = plt.subplot(5,1,4)
      ax5 = plt.subplot(5,1,5)

      ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('S [g/L]')
      
      ax2.grid()
      ax2.set_ylabel('S [g/L]')
          
      ax3.grid()
      ax3.set_ylabel('X [g/L]')
 
      ax4.grid()
      ax4.set_ylabel('F*X [g/h]') 
 
      ax5.grid()
      ax5.set_ylabel('D=F/V, mu [1/h]')
      ax5.set_xlabel('Time [h]') 
      
      # List of commands to be executed by simu() after a simulation  
      diagrams.clear()
      diagrams.append("ax1.plot(t,sim_res['bioreactor.c[2]'],color='b',linestyle=linetype)")
      diagrams.append("ax2.plot(t,sim_res['bioreactor.c[2]'],color='b',linestyle=linetype)")
      diagrams.append("ax2.set_ylim([0,1])")
      diagrams.append("ax3.plot(t,sim_res['bioreactor.c[1]'],color='b',linestyle=linetype)")   
      diagrams.append("ax4.plot(t,sim_res['bioreactor.inlet[1].F']*sim_res['bioreactor.c[1]'],color='b',linestyle=linetype)") 
      diagrams.append("ax4.plot([0, simulationTime], [cstrProdMax(model), cstrProdMax(model)], color='r',linestyle=linetype)")
      diagrams.append("ax4.legend(['FX','FX_max'])")    
      diagrams.append("ax5.plot(t,sim_res['D'],color='b',linestyle=linetype)") 
      diagrams.append("ax5.plot(t,sim_res['bioreactor.culture.q[1]'],color='r',linestyle=linetype)") 
      diagrams.append("ax5.legend(['D','mu'])")    

   elif plotType == 'TimeSeries3':

      plt.figure()
      ax11 = plt.subplot(4,2,1)
      ax12 = plt.subplot(4,2,2)       
      ax21 = plt.subplot(4,2,3)
      ax22 = plt.subplot(4,2,4)
      ax31 = plt.subplot(4,2,5)
      ax41 = plt.subplot(4,2,7)

      ax11.set_title(title)
      ax11.grid()
      ax11.set_ylabel('S [g/L]')
    
      ax21.grid()
      ax21.set_ylabel('X [g/L]')
 
      ax31.grid()
      ax31.set_ylabel('F*X [g/h]') 
 
      ax41.grid()
      ax41.set_ylabel('D=F/V, mu [1/h]')
      ax41.set_xlabel('Time [h]') 
      
      ax12.set_title(' - microscopic world')
      ax12.grid()
      ax12.set_ylabel('qS [g/(g*h)]')    
      
      ax22.grid()
      ax22.set_ylabel('mu [1/h]')
      ax22.set_xlabel('Time [h]')   
      
      # List of commands to be executed by simu() after a simulation  
      diagrams.clear()
      diagrams.append("ax11.plot(t,sim_res['bioreactor.c[2]'],color='b',linestyle=linetype)")
      diagrams.append("ax21.plot(t,sim_res['bioreactor.c[1]'],color='b',linestyle=linetype)")   
      diagrams.append("ax31.plot(t,sim_res['bioreactor.inlet[1].F']*sim_res['bioreactor.c[1]'],color='b',linestyle=linetype)") 
      diagrams.append("ax31.plot([0, simulationTime], [cstrProdMax(model), cstrProdMax(model)], color='r',linestyle=linetype)")
      diagrams.append("ax31.legend(['FX','FX_max'])")   
      diagrams.append("ax41.plot(t,sim_res['D'],color='b',linestyle=linetype)") 
      diagrams.append("ax41.plot(t,sim_res['bioreactor.culture.q[1]'],color='r',linestyle=linetype)") 
      diagrams.append("ax41.legend(['D','mu'])") 
      
      diagrams.append("ax12.plot(t,-sim_res['bioreactor.culture.q[2]'],color='b',linestyle=linetype)")   
      diagrams.append("ax22.plot(t,sim_res['bioreactor.culture.q[1]'],color='b',linestyle=linetype)") 
      
   elif plotType == 'TimeSeries4':

      plt.figure()
      ax1 = plt.subplot(3,1,1)
      ax2 = plt.subplot(3,1,2)       
      ax3 = plt.subplot(3,1,3)

      ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('S [g/L]')
    
      ax2.grid()
      ax2.set_ylabel('X [g/L]')
  
      ax3.grid()
      ax3.set_ylabel('D=F/V, mu [1/h]')
      ax3.set_xlabel('Time [h]') 
      
      # List of commands to be executed by simu() after a simulation  
      diagrams.clear()
      diagrams.append("ax1.plot(t,sim_res['bioreactor.c[2]'],color='b',linestyle=linetype)")
      diagrams.append("ax2.plot(t,sim_res['bioreactor.c[1]'],color='b',linestyle=linetype)")   
      diagrams.append("ax3.plot(t,sim_res['D'],color='b',linestyle=linetype)") 
      diagrams.append("ax3.plot(t,sim_res['bioreactor.culture.q[1]'],color='r',linestyle=linetype)") 
      diagrams.append("ax3.legend(['D','mu'])")          

   elif plotType == 'TimeSeries5':

      plt.figure()
      ax1 = plt.subplot(2,1,1)
      ax2 = plt.subplot(2,1,2)       

      ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('S [g/L]')
    
      ax2.grid()
      ax2.set_ylabel('X [g/L]')
      ax2.set_xlabel('Time [h]') 
      
      # List of commands to be executed by simu() after a simulation  
      diagrams.clear()
      diagrams.append("ax1.plot(t,sim_res['bioreactor.c[2]'],color='b',linestyle=linetype)")
      diagrams.append("ax2.plot(t,sim_res['bioreactor.c[1]'],color='b',linestyle=linetype)")   
        
   elif plotType == 'TimeSeries6':

      plt.figure()
      ax11 = plt.subplot(4,2,1)
      ax21 = plt.subplot(4,2,3)       
      ax31 = plt.subplot(4,2,5)
      ax41 = plt.subplot(4,2,7)
      
      ax12 = plt.subplot(1,2,2)

      ax11.set_title(title)
      ax11.grid()
      ax11.set_ylabel('S [g/L]')
      
      ax21.grid()
      ax21.set_ylabel('S [g/L]')
    
      ax31.grid()
      ax31.set_ylabel('X [g/L]')
  
      ax41.grid()
      ax41.set_ylabel('D=F/V, mu [1/h]')
      ax41.set_xlabel('Time [h]') 
        
      ax12.grid()
      ax12.set_ylabel('mu=Y*qS [1/h]')
      ax12.set_xlabel('S [g/L]')
      
      # List of commands to be executed by simu() after a simulation  
      diagrams.clear()
      diagrams.append("ax11.plot(t,sim_res['bioreactor.c[2]'],color='b',linestyle=linetype)")
      diagrams.append("ax21.plot(t,sim_res['bioreactor.c[2]'],color='b',linestyle=linetype)")
      diagrams.append("ax21.set_ylim([0,1])")
      diagrams.append("ax31.plot(t,sim_res['bioreactor.c[1]'],color='b',linestyle=linetype)")   
      diagrams.append("ax41.plot(t,sim_res['D'],color='b',linestyle=linetype)") 
      diagrams.append("ax41.plot(t,sim_res['bioreactor.culture.q[1]'],color='r',linestyle=linetype)") 
      diagrams.append("ax41.legend(['D','mu'])")      
      
      diagrams.append("ax12.plot(sim_res['bioreactor.c[2]'], sim_res['bioreactor.culture.q[1]'], 'b*')")   
      diagrams.append("ax12.plot(sim_res['bioreactor.c[2]'][-1], sim_res['bioreactor.culture.q[1]'][-1], 'r*')") 
      diagrams.append("ax12.set_xlim([0,1])")    
        
   elif plotType =='PhasePlane':
      plt.figure()
      ax1 = plt.subplot(1,1,1)
        
      ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('S [g/L]')
      ax1.set_xlabel('X [g/L]')  

      # List of commands to be executed by simu() after a simulation  
      diagrams.clear()
      diagrams.append("ax1.plot(sim_res['bioreactor.c[1]'], sim_res['bioreactor.c[2]'],color='b',linestyle=linetype)")
             
   else:
      print("Plot window type not correct")

def cstrProdMax(model):
   """Calculate from the model maximal chemostat productivity FX_max"""
        
   X_max = model.get('bioreactor.culture.Y')*model.get('feedtank.c_in[2]')
   mu_max = model.get('bioreactor.culture.Y')*model.get('bioreactor.culture.qSmax')
   V_nom = model.get('bioreactor.V_0')
   FX_max = mu_max*X_max*V_nom    
    
   return FX_max[0]

# Define describtions partly coded here and partly taken from the FMU
def describe(name, decimals=3):
   """Look up description of culture, media, as well as parameters and variables in the model code"""
   
   if name == 'culture':
      print('Simplified text book model - only substrate S and cell concentration X')      
 
   elif name in ['broth', 'liquidphase', 'media']: 
      """Describe medium used"""
      X = model.get('liquidphase.X')[0] 
      X_description = model.get_variable_description('liquidphase.X') 
      X_mw = model.get('liquidphase.mw[1]')[0]
         
      S = model.get('liquidphase.S')[0] 
      S_description = model.get_variable_description('liquidphase.S')
      S_mw = model.get('liquidphase.mw[2]')[0]
         
      print()
      print('Reactor broth substances included in the model')
      print()
      print(X_description, '    index = ', X, 'molecular weight = ', X_mw, 'Da')
      print(S_description, 'index = ', S, 'molecular weight = ', S_mw, 'Da')
  
   elif name in ['parts']:
      describe_parts(component_list_minimum)
      
   elif name in ['cstrProdMax']:
      print(cstrProdMax.__doc__,':',cstrProdMax(model), '[ g/h ]')
      
   else:
      describe_general(name, decimals)

#------------------------------------------------------------------------------------------------------------------
#  General code 
FMU_explore = 'FMU-explore ver 0.9.1'
#------------------------------------------------------------------------------------------------------------------

# Define function par() for parameter update
def par(parDict=parDict, parLocation=parLocation, *x, **x_kwarg):
   """ Set parameter values if available in the predefined dictionaryt parDict. """
   x_kwarg.update(*x)
   x_temp = {}
   for key in x_kwarg.keys():
      if key in parDict.keys():
         x_temp.update({key: x_kwarg[key]})
      else:
         print('Error:', key, '- seems not an accessible parameter - check the spelling')
   parDict.update(x_temp)

# Define function init() for initial values update
def init(parDict=parDict, *x, **x_kwarg):
   """ Set initial values and the name should contain string '_0' to be accepted.
       The function can handle general parameter string location names if entered as a dictionary. """
   x_kwarg.update(*x)
   x_init={}
   for key in x_kwarg.keys():
      if '_0' in key: 
         x_init.update({key: x_kwarg[key]})
      else:
         print('Error:', key, '- seems not an initial value, use par() instead - check the spelling')
   parDict.update(x_init)
   
# Define function disp() for display of initial values and parameters
def dict_reverser(d):
   seen = set()
   return {v: k for k, v in d.items() if v not in seen or seen.add(v)}
   
def disp(name='', decimals=3, mode='short'):
   """ Display intial values and parameters in the model that include "name" and is in parLocation list.
       Note, it does not take the value from the dictionary par but from the model. """
   global parLocation, model
   
   if mode in ['short']:
      k = 0
      for Location in parLocation.values():
         if name in Location:
            if type(model.get(Location)[0]) != np.bool_:
               print(dict_reverser(parLocation)[Location] , ':', np.round(model.get(Location)[0],decimals))
            else:
               print(dict_reverser(parLocation)[Location] , ':', model.get(Location)[0])               
         else:
            k = k+1
      if k == len(parLocation):
         for parName in parDict.keys():
            if name in parName:
               if type(model.get(Location)[0]) != np.bool_:
                  print(parName,':', np.round(model.get(parLocation[parName])[0],decimals))
               else: 
                  print(parName,':', model.get(parLocation[parName])[0])
   if mode in ['long','location']:
      k = 0
      for Location in parLocation.values():
         if name in Location:
            if type(model.get(Location)[0]) != np.bool_:       
               print(Location,':', dict_reverser(parLocation)[Location] , ':', np.round(model.get(Location)[0],decimals))
         else:
            k = k+1
      if k == len(parLocation):
         for parName in parDict.keys():
            if name in parName:
               if type(model.get(Location)[0]) != np.bool_:
                  print(parLocation[parName], ':', dict_reverser(parLocation)[Location], ':', parName,':', 
                     np.round(model.get(parLocation[parName])[0],decimals))

# Line types
def setLines(lines=['-','--',':','-.']):
   """Set list of linetypes used in plots"""
   global linecycler
   linecycler = cycle(lines)

# Show plots from sim_res, just that
def show(diagrams=diagrams):
   """Show diagrams chosen by newplot()"""
   # Plot pen
   linetype = next(linecycler)    
   # Plot diagrams 
   for command in diagrams: eval(command)

# Simulation
def simu(simulationTimeLocal=simulationTime, mode='Initial', diagrams=diagrams,timeDiscreteStates=timeDiscreteStates):
   """Model loaded and given intial values and parameter before,
      and plot window also setup before."""
    
   # Global variables
   global model, parDict, stateDict, prevFinalTime, simulationTime, sim_res, t

   # Transfer of argument to global variable
   simulationTime = simulationTimeLocal 
   
   # Check parDict
   value_missing = 0
   for key in parDict.keys():
      if parDict[key] in [np.nan, None, '']:
         print('Value missing:', key)
         value_missing =+1
   if value_missing>0: return
         
   # Load model
   if model is None:
      model = load_fmu(fmu_model) 
   model.reset()
      
   # Run simulation
   if mode in ['Initial', 'initial', 'init']:
      # Set parameters and intial state values:
      for key in parDict.keys():
         model.set(parLocation[key],parDict[key])   
      # Simulate
      sim_res = model.simulate(final_time=simulationTime, options=opts)      
   elif mode in ['Continued', 'continued', 'cont']:
      # Set parameters and intial state values:
      for key in parDict.keys():
         model.set(parLocation[key],parDict[key])                
      try: 
         for key in stateDict.keys():
            if not key[-1] == ']':
               model.set(key+'_0', stateDict[key])
            elif key[-3] == '[':
               model.set(key[:-3]+'_0'+key[-3:], stateDict[key]) 
            elif key[-4] == '[':
               model.set(key[:-4]+'_0'+key[-4:], stateDict[key]) 
            elif key[-5] == '[':
               model.set(key[:-5]+'_0'+key[-5:], stateDict[key]) 
            else:
               print('The state vecotr has more than 1000 states')
               break
      except NameError:
         print("Simulation is first done with default mode='init'")
         prevFinalTime = 0
      # Simulate
      sim_res = model.simulate(start_time=prevFinalTime,
                              final_time=prevFinalTime + simulationTime,
                              options=opts)     
   else:
      print("Simulation mode not correct")
    
   # Extract data
   t = sim_res['time']
 
   # Plot diagrams
   linetype = next(linecycler)    
   for command in diagrams: eval(command)
            
   # Store final state values stateDict:
   try: stateDict
   except NameError:
      stateDict = {}
      stateDict = model.get_states_list()
      stateDict.update(timeDiscreteStates)
   for key in list(stateDict.keys()):
      stateDict[key] = model.get(key)[0]        

   # Store time from where simulation will start next time
   prevFinalTime = model.time
   
# Describe model parts of the combined system
def describe_parts(component_list=[]):
   """List all parts of the model""" 
       
   def model_component(variable_name):
      i = 0
      name = ''
      finished = False
      if not variable_name[0] == '_':
         while not finished:
            name = name + variable_name[i]
            if i == len(variable_name)-1:
                finished = True 
            elif variable_name[i+1] in ['.', '(']: 
                finished = True
            else: 
                i=i+1
      if name in ['der', 'temp_1']: name = ''
      return name
    
   variables = list(model.get_model_variables().keys())
        
   for i in range(len(variables)):
      component = model_component(variables[i])
      if (component not in component_list) \
      & (component not in ['','BPL', 'Customer', 'today[1]', 'today[2]', 'today[3]', 'temp_2', 'temp_3']):
         component_list.append(component)
      
   print(sorted(component_list, key=str.casefold))

# Describe parameters and variables in the Modelica code
def describe_general(name, decimals):
  
   if name == 'time':
      description = 'Time'
      unit = 'h'
      print(description,'[',unit,']')
      
   elif name in parLocation.keys():
      description = model.get_variable_description(parLocation[name])
      value = model.get(parLocation[name])[0]
      try:
         unit = model.get_variable_unit(parLocation[name])
      except FMUException:
         unit =''
      if unit =='':
         if type(value) != np.bool_:
            print(description, ':', np.round(value, decimals))
         else:
            print(description, ':', value)            
      else:
        print(description, ':', np.round(value, decimals), '[',unit,']')
                  
   else:
      description = model.get_variable_description(name)
      value = model.get(name)[0]
      try:
         unit = model.get_variable_unit(name)
      except FMUException:
         unit =''
      if unit =='':
         if type(value) != np.bool_:
            print(description, ':', np.round(value, decimals))
         else:
            print(description, ':', value)     
      else:
         print(description, ':', np.round(value, decimals), '[',unit,']')
         
# Describe framework
def BPL_info():
   print()
   print('Model for bioreactor has been setup. Key commands:')
   print(' - par()       - change of parameters and initial values')
   print(' - init()      - change initial values only')
   print(' - simu()      - simulate and plot')
   print(' - newplot()   - make a new plot')
   print(' - show()      - show plot from previous simulation')
   print(' - disp()      - display parameters and initial values from the last simulation')
   print(' - describe()  - describe culture, broth, parameters, variables with values / units')
   print()
   print('Note that both disp() and describe() takes values from the last simulation')
   print()
   print('Brief information about a command by help(), eg help(simu)') 
   print('Key system information is listed with the command system_info()')

def system_info():
   """Print system information"""
   FMU_type = model.__class__.__name__
   print()
   print('System information')
   print(' -OS:', platform.system())
   print(' -Python:', platform.python_version())
   print(' -PyFMI:', version('pyfmi'))
   print(' -FMU by:', model.get_generation_tool())
   print(' -FMI:', model.get_version())
   print(' -Type:', FMU_type)
   print(' -Name:', model.get_name())
   print(' -Generated:', model.get_generation_date_and_time())
   print(' -Description:', model.get('BPL.version')[0])  
   print(' -Interaction:', FMU_explore)

#------------------------------------------------------------------------------------------------------------------
#  Startup
#------------------------------------------------------------------------------------------------------------------

BPL_info()
