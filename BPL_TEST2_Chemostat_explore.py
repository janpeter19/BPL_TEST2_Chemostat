# setup application functions BPL_TEST2_Chemostat, dependent on previous import of functions from fmu_explore 
# Author: Jan Peter Axelsson
#------------------------------------------------------------------------------------------------------------------
# 2026-08-25 - Created
#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------
#  Specific application functions: newplot(), describe()
#------------------------------------------------------------------------------------------------------------------

# Define standard diagrams
def newplot(title='Chemostat cultivation', plotType='TimeSeries'):
   """ Standard plot window, two possibilities:
         diagram = 'TimeSeries' default
         diagram = 'PhasePlane' 
       and plot main title
         title = '' """
       
   # Reset pens
   resetPen()
    
   # Plot diagram
   if plotType == 'TimeSeries':

      plt.figure()
      ax1 = plt.subplot(4,1,1)
      ax2 = plt.subplot(4,1,2)       
      ax3 = plt.subplot(4,1,3)
      ax4 = plt.subplot(4,1,4)
      
      ax.clear()
      ax.append(ax1)
      ax.append(ax2)
      ax.append(ax3)
      ax.append(ax4)

      ax[0].set_title(title)
      ax[0].grid()
      ax[0].set_ylabel('S [g/L]')
    
      ax[1].grid()
      ax[1].set_ylabel('X [g/L]')
 
      ax[2].grid()
      ax[2].set_ylabel('F*X [g/h]') 
 
      ax[3].grid()
      ax[3].set_ylabel('D=F/V, mu [1/h]')
      ax[3].set_xlabel('Time [h]') 
      
      # List of commands to be executed by simu() after a simulation  
      diagrams.clear()
      diagrams.append("ax[0].plot(t,sim_res['bioreactor.c[2]'],color='b',linestyle=linetype)")
      diagrams.append("ax[1].plot(t,sim_res['bioreactor.c[1]'],color='b',linestyle=linetype)")   
      diagrams.append("ax[2].plot(t,sim_res['bioreactor.inlet[1].F']*sim_res['bioreactor.c[1]'],color='b',linestyle=linetype)") 
#      diagrams.append("ax[2].plot([0, simulationTime], [cstrProdMax(model), cstrProdMax(model)], color='r',linestyle=linetype)")
      diagrams.append("ax[2].plot([0, simulationTime], [external_function(), external_function()], color='r',linestyle=linetype)")
#      diagrams.append("ax[2].plot([0, simulationTime], color='r',linestyle=linetype)")
      diagrams.append("ax[2].legend(['FX','FX_max'])")   
      diagrams.append("ax[3].plot(t,sim_res['D'],color='b',linestyle=linetype)") 
      diagrams.append("ax[3].plot(t,sim_res['bioreactor.culture.q[1]'],color='r',linestyle=linetype)") 
      diagrams.append("ax[3].legend(['D','mu'])")    
   
   elif plotType == 'TimeSeries2':

      plt.figure()
      ax1 = plt.subplot(5,1,1)
      ax2 = plt.subplot(5,1,2)       
      ax3 = plt.subplot(5,1,3)
      ax4 = plt.subplot(5,1,4)
      ax5 = plt.subplot(5,1,5)
      
      ax.clear()
      ax.append(ax1)
      ax.append(ax2)
      ax.append(ax3)
      ax.append(ax4)
      ax.append(ax5)

      ax[0].set_title(title)
      ax[0].grid()
      ax[0].set_ylabel('S [g/L]')
      
      ax[1].grid()
      ax[1].set_ylabel('S [g/L]')
          
      ax[2].grid()
      ax[2].set_ylabel('X [g/L]')
 
      ax[3].grid()
      ax[3].set_ylabel('F*X [g/h]') 
 
      ax[4].grid()
      ax[4].set_ylabel('D=F/V, mu [1/h]')
      ax[4].set_xlabel('Time [h]') 
      
      # List of commands to be executed by simu() after a simulation  
      diagrams.clear()
      diagrams.append("ax[0].plot(t,sim_res['bioreactor.c[2]'],color='b',linestyle=linetype)")
      diagrams.append("ax[1].plot(t,sim_res['bioreactor.c[2]'],color='b',linestyle=linetype)")
      diagrams.append("ax[1].set_ylim([0,1])")
      diagrams.append("ax[2].plot(t,sim_res['bioreactor.c[1]'],color='b',linestyle=linetype)")   
      diagrams.append("ax[3].plot(t,sim_res['bioreactor.inlet[1].F']*sim_res['bioreactor.c[1]'],color='b',linestyle=linetype)") 
      diagrams.append("ax[3].plot([0, simulationTime], [cstrProdMax(model), cstrProdMax(model)], color='r',linestyle=linetype)")
      diagrams.append("ax[3].legend(['FX','FX_max'])")    
      diagrams.append("ax[4].plot(t,sim_res['D'],color='b',linestyle=linetype)") 
      diagrams.append("ax[4].plot(t,sim_res['bioreactor.culture.q[1]'],color='r',linestyle=linetype)") 
      diagrams.append("ax[4].legend(['D','mu'])")    

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
      
   elif name in ['MSL']:
      describe_MSL()
      
   elif name in ['cstrProdMax']:
      print(cstrProdMax.__doc__ ,':',cstrProdMax(model), '[ g/h ]')
      
   else:
      describe_general(name, decimals)

#------------------------------------------------------------------------------------------------------------------
#  Startup
#------------------------------------------------------------------------------------------------------------------

FMU_explore_info()
