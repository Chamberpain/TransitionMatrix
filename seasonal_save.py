#import libraries: 

import numpy as np
import pickle
from scipy.linalg import inv

def least_square_fit(data, trend, parameters, period, fill_val):
    
    """
    least_square_fit(data, trend, parameters, period, fill_val)
    
        Function to compute the unweighted least square fit of temporal data 
        
        Parameters 
        ----------
        data : numpy 2D array of temporal data
               e.g. print(data.shape) => (133,) 
        trend : type of function least square is fitting. Options for this variables include: 
               a) trend = 'linear'
               b) trend = 'exponential'
               c) trend = 'sinusoidal' 
        parameters : number of parameters in model 
               a) if trend = 'linear', then options for parameters includes: 
                    i) parameters = 1 : fits mean 
                    ii) parameters = 2 : fits mean and linear trend 
               b) if trend = 'exponential', then the only option for paramters includes: 
                    i) parameters = 2: fits linear trend and mean offset to a linearized exponential equation 
               c) if trend = 'sinusiodal', then options for paramters includes: 
                    i) parameters = 3 : fits mean, and annual cycle (period = 1 year, 12 months, or 365.25 days)
                    ii) parameters = 4 : fits mean, linear trend, and annual cycle
                    iii) parameters = 5 : fits mean, annual cycle, and semi-annual cycle (period= 1/2 year,6 months,182.625 days)
                    iv) paramters = 6: fits mean, linear trend, annual cycle, and semi-annual cycle 

        period : period of the frequency of the sinusoidal signal 
        
        fill_val : determines what fill value is present in your data set. Options include:
               fill_val = 'NaN' or fill_val = 'mask' or fill_val = 'none'
        
        Returns
        -------
        data_ulsf : unweighted least square fitted data
        
        Libraries necessary to run function
        -----------------------------------
        Numpy : import numpy as np
        
    """
    

    
    #consider for LSF only unmasked or non-nan data: 
    if fill_val == 'mask':
        ind = data.mask
    elif fill_val == 'NaN':
        ind = np.isnan(data)
        
    #set all masked data points to the mean swh of the time series or interpolate the data: 
    if fill_val == 'mask':
        data[ind] = np.ma.mean(data)
    elif fill_val == 'NaN':
        data[ind] = np.nanmean(data)
        
    #initialize time vector: 
    time = np.arange(1,len(data)+1,1)
    
    #set number of parameters and length of data: 
    N, M = parameters, len(data)

    #create model matrix: 
    A = np.zeros((M,N))
    
    #set conditional statements for each case that the lsf will be applied: 
    if trend == 'linear':
        
        #set the two cases for the linear trend parameters: 
        if parameters == 1: 
            #place parameters in model matrix: 
            A[:,0] = 1.0
            #Compute solution to the matrix equation Ax = b:  
            x_data = np.dot( np.dot( inv(np.dot(A.conj().T, A)), A.conj().T), data.conj().T)
            #Compute unweighted least square fit model: 
            hfit = x_data[0]
            
        elif parameters == 2: 
            #place parameters in model matrix: 
            A[:,0] = 1.0
            A[:,1] = time
            #Compute solution to the matrix equation Ax = b:  
            x_data = np.dot( np.dot( inv(np.dot(A.conj().T, A)), A.conj().T), data.conj().T)
            #Compute unweighted least square fit model: 
            hfit = x_data[0] + x_data[1]*time
    
    elif trend == 'exponential':
        
        #set the two cases for the linear trend parameters: 
        if parameters == 2: 
            #place parameters in model matrix: 
            A[:,0] = 1.0
            A[:,1] = time
            #Compute solution to the matrix equation Ax = b:  
            x_data = np.dot( np.dot( inv(np.dot(A.conj().T, A)), A.conj().T), data.conj().T)
            #Compute unweighted least square fit model: 
            hfit = x_data[0] + x_data[1]*time
        
    elif trend == 'sinusoidal':
        
        #set the two cases for the linear trend parameters: 
        if parameters == 3: 
            #place parameters in model matrix: 
            A[:,0] = 1.0
            A[:,1] = np.sin(time*2*np.pi/period)
            A[:,2] = np.cos(time*2*np.pi/period)
            #Compute solution to the matrix equation Ax = b:  
            x_data = np.dot( np.dot( inv(np.dot(A.conj().T, A)), A.conj().T), data.conj().T)
            #Compute unweighted least square fit model: 
            hfit = x_data[0] + x_data[1]*np.sin(time*2*np.pi/period) + x_data[2]*np.cos(time*2*np.pi/period)
            
        elif parameters == 4: 
            #place parameters in model matrix: 
            A[:,0] = 1.0
            A[:,1] = time
            A[:,2] = np.sin(time*2*np.pi/period)
            A[:,3] = np.cos(time*2*np.pi/period)
            #Compute solution to the matrix equation Ax = b:  
            x_data = np.dot( np.dot( inv(np.dot(A.conj().T, A)), A.conj().T), data.conj().T)
            #Compute unweighted least square fit model: 
            hfit = x_data[0] + x_data[1]*time + x_data[2]*np.sin(time*2*np.pi/period) + x_data[3]*np.cos(time*2*np.pi/period)
            
        elif parameters == 5:
            #place parameters in model matrix: 
            A[:,0] = 1.0
            A[:,1] = np.sin(time*2*np.pi/period)
            A[:,2] = np.cos(time*2*np.pi/period)
            A[:,3] = np.sin(time*2*np.pi/(period/2))
            A[:,4] = np.cos(time*2*np.pi/(period/2))
            #Compute solution to the matrix equation Ax = b:  
            x_data = np.dot( np.dot( inv(np.dot(A.conj().T, A)), A.conj().T), data.conj().T)
            #Compute unweighted least square fit model: 
            hfit = x_data[0] + x_data[1]*np.sin(time*2*np.pi/period) + x_data[2]*np.cos(time*2*np.pi/period) + x_data[3]*np.sin(time*2*np.pi/(period/2)) + x_data[4]*np.cos(time*2*np.pi/(period/2))
            
        elif parameters == 6: 
            #place parameters in model matrix: 
            A[:,0] = 1.0
            A[:,1] = time
            A[:,2] = np.sin(time*2*np.pi/period)
            A[:,3] = np.cos(time*2*np.pi/period)
            A[:,4] = np.sin(time*2*np.pi/(period/2))
            A[:,5] = np.cos(time*2*np.pi/(period/2))
            #Compute solution to the matrix equation Ax = b:  
            x_data = np.dot( np.dot( inv(np.dot(A.conj().T, A)), A.conj().T), data.conj().T)
            #Compute unweighted least square fit model: 
            hfit = x_data[0] + x_data[1]*time + x_data[2]*np.sin(time*2*np.pi/period) + x_data[3]*np.cos(time*2*np.pi/period) + x_data[4]*np.sin(time*2*np.pi/(period/2)) + x_data[5]*np.cos(time*2*np.pi/(period/2))
            
            
    return hfit, x_data



variable_list = ['pco2','o2','dic','po4','biomass_p']
for variable in variable_list:
	file_name=variable+'.pkl'
	file = open (file_name,'rb')
	data = pickle.load(file)
	data = np.vstack(data)
	sin_output_holder = np.zeros([data.shape[1],data.shape[2]])
	cos_output_holder = np.zeros([data.shape[1],data.shape[2]])
	for dummy_lat in range(data.shape[1]):
		print('dummy lat = '+str(dummy_lat))
		for dummy_lon in range(data.shape[2]):
			print('dummy lon = '+str(dummy_lon)) 
			time_series = data[:,dummy_lat,dummy_lon]
			_,params = least_square_fit(time_series,'sinusoidal',4,365.25,'NaN')
			sin_output_holder[dummy_lat,dummy_lon]=params.data[2]
			cos_output_holder[dummy_lat,dummy_lon]=params.data[3]
	total = np.sqrt(sin_output_holder**2+cos_output_holder**2)
	np.save('seasonal_'+variable,total)


for variable in variable_list:
	total = np.load('seasonal_'+variable+'.npy')
	total_plot = np.ma.masked_less(total,10**-6)
	total_plot = np.ma.masked_greater(total_plot,total.mean()+7*total.std())
	plt.title('Seasonal Magnitude for '+variable)
	plt.pcolor(total_plot)
	plt.colorbar()
	plt.savefig(variable)
	plt.close()
