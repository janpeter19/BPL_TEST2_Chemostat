# setup data TEST2_Chemostat_fmpy 
# Author: Jan Peter Axelsson
#------------------------------------------------------------------------------------------------------------------
# 2026-09-08 - Created
# 2026-09-12 - Move calculations around stateValue etc to the initialization of fmy:_explore_fmpy ver 1.1.8
# 2026-09-18 - Decrease the framework to what is necessary and move matlotlib to the other setup-file
#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------
#  Framework
#------------------------------------------------------------------------------------------------------------------

# Setup framework
import platform
import locale
from fmpy import simulate_fmu
from fmpy import read_model_description

# Set the environment - for Linux a JSON-file in the FMU is read
if platform.system() == 'Linux': locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

#------------------------------------------------------------------------------------------------------------------
#  Setup application FMU
#------------------------------------------------------------------------------------------------------------------

# Provde the right FMU and load for different platforms in user dialogue:
if platform.system() == 'Windows':
   print('Windows - run FMU pre-compiled JModelica 2.14')
   fmu_model ='BPL_TEST2_Chemostat_windows_jm_cs.fmu' 
   model_description = read_model_description(fmu_model)         
   flag_vendor = 'JM'
   flag_type = 'CS' 
elif platform.system() == 'Linux': 
   flag_vendor = 'OM'
   flag_type = 'ME'
   if flag_vendor in ['OM','om']:
      print('Linux - run FMU pre-compiled OpenModelica') 
      if flag_type in ['CS','cs']:         
         fmu_model ='BPL_TEST2_Chemostat_linux_om_cs.fmu'    
         model_description = read_model_description(fmu_model)  
      if flag_type in ['ME','me']:         
         fmu_model ='BPL_TEST2_Chemostat_linux_om_me.fmu'    
         model_description = read_model_description(fmu_model)  
   else:    
      print('There is no FMU for this platform')

# Provide various opts-profiles
if flag_type in ['CS', 'cs']:
   opts_std = {'NCP': 500}
elif flag_type in ['ME', 'me']:
   opts_std = {'NCP': 500}
else:    
   print('There is no FMU for this platform')
  
# Provide various MSL and BPL versions
if flag_vendor in ['JM', 'jm']:
   constants = [v for v in model_description.modelVariables if v.causality == 'local'] 
   MSL_usage = [x[1] for x in [(constants[k].name, constants[k].start) for k in range(len(constants))] if 'MSL.usage' in x[0]][0]   
   MSL_version = [x[1] for x in [(constants[k].name, constants[k].start) for k in range(len(constants))] if 'MSL.version' in x[0]][0]
   BPL_version = [x[1] for x in [(constants[k].name, constants[k].start) for k in range(len(constants))] if 'BPL.version' in x[0]][0] 
elif flag_vendor in ['OM', 'om']:
   MSL_usage = '4.1.0 - used components: RealInput, RealOutput, CombiTimeTable, Types' 
   MSL_version = '4.1.0'
   BPL_version = 'Bioprocess Library version 2.3.2' 
else:    
   print('There is no FMU for this platform')

# Simulation time
simulationTime = 60.0

# Dictionary of time discrete states
timeDiscreteStates = {} 

# Define a minimal compoent list of the model as a starting point for describe('parts')
component_list_minimum = ['bioreactor', 'bioreactor.culture']

# Provide process diagram on disk
fmu_process_diagram ='BPL_TEST2_Chemostat_process_diagram_om.png'

#------------------------------------------------------------------------------------------------------------------
#  Specific application constructs: stateValue, parValue, parLocation, parCheck, diagrams. ax
#------------------------------------------------------------------------------------------------------------------
    
# Create dictionaries parValue and parLocation
parValue = {}
parValue['V_start'] = 1.0
parValue['VX_start'] = 1.0
parValue['VS_start'] = 30.0

parValue['Y'] = 0.5
parValue['qSmax'] = 0.75
parValue['Ks'] = 0.1

parValue['S_in'] = 30.0
parValue['feedtank.V_start'] = 100.0

parValue['t0'] = 0.0
parValue['F0'] = 0.0
parValue['t1'] = 10.0
parValue['F1'] = 0.20
parValue['t2'] = 999.0
parValue['F2'] = 0.20
parValue['t3'] = 1000.0
parValue['F3'] = 0.20

parLocation = {}
parLocation['V_start'] = 'bioreactor.V_start'
parLocation['VX_start'] = 'bioreactor.m_start[1]' 
parLocation['VS_start'] = 'bioreactor.m_start[2]' 

parLocation['Y'] = 'bioreactor.culture.Y'
parLocation['qSmax'] = 'bioreactor.culture.qSmax'
parLocation['Ks'] = 'bioreactor.culture.Ks'

parLocation['S_in'] = 'feedtank.c_in[2]'
parLocation['feedtank.V_start'] = 'feedtank.V_start'
parLocation['t0'] = 'schemePumps.table[1,1]'
parLocation['F0'] = 'schemePumps.table[1,2]'
parLocation['t1'] = 'schemePumps.table[2,1]'
parLocation['F1'] = 'schemePumps.table[2,2]'
parLocation['t2'] = 'schemePumps.table[3,1]'
parLocation['F2'] = 'schemePumps.table[3,2]'
parLocation['t3'] = 'schemePumps.table[4,1]'
parLocation['F3'] = 'schemePumps.table[4,2]'

# Extra only for describe()
keyVariables = []
parLocation['mu'] = 'bioreactor.culture.mu'; keyVariables.append(parLocation['mu'])
keyVariables.append(parLocation['S_in'])

# Parameter value check 
parCheck = []
parCheck.append("parValue['Y'] > 0")
parCheck.append("parValue['qSmax'] > 0")
parCheck.append("parValue['Ks'] > 0")
parCheck.append("parValue['V_start'] > 0")
parCheck.append("parValue['VX_start'] >= 0")
parCheck.append("parValue['t0'] < parValue['t1']")
parCheck.append("parValue['t1'] < parValue['t2']")
parCheck.append("parValue['t2'] < parValue['t3']")

# Create list of diagrams to be plotted by simu()
diagrams = []

# Create an empty list axes to be defined in newplot() and plotted by simu() or show()
ax = []

# Create list of pens for the diagrams
lines = ['-','--',':','-.']

#------------------------------------------------------------------------------------------------------------------
#  Specific application constructs: external function
#------------------------------------------------------------------------------------------------------------------

# Define maximal performance criteria
def cstrProdMax():
   """Calculate from the model maximal chemostat productivity FX_max"""        
   X_max = model_get('bioreactor.culture.Y')*model_get('feedtank.c_in[2]')        
   mu_max = model_get('bioreactor.culture.Y')*model_get('bioreactor.culture.qSmax')
   V_nom = model_get('bioreactor.V_start')
   FX_max = mu_max*X_max*V_nom      
   return FX_max[0]

