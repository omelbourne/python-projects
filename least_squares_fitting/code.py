import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import random
from scipy import optimize
from mpl_toolkits import mplot3d

#Model function for linear
def funcLine(x, a,b):
    return a*x+b

#Model function for weighted linear
def funcWeighted(x, a,b,c):
    return a*x**2+b*np.log(x)+c

#Model function for non-linear
def funcNonlinear(x, a,b):
    return x*np.exp(a)+b*np.exp(x)

#Model function for variable change of non-linear
def funcNonlinear2(x, c,b):
    return x*c+b*np.exp(x)

#Calculating linear least squares with LS formulas
def task1a():
    data = np.loadtxt("linear_LS.txt") #import data
    X = data[:, 0]                                                                                
    Ymeas = data[:, 1]
    sigma_noise = 1 #measurement noise std dev
    Z = Ymeas
    
    H = np.column_stack((X, np.ones(len(X))))   #matrix H
    HT = np.transpose(H)                        #transpose of H
    theta_hat = np.linalg.inv(HT@H)@HT@Z        #least squares estimation  
    Cth = sigma_noise**2*np.linalg.inv(HT@H)    #covariance matrix
    uncertainties = (np.diagonal(Cth))**0.5     #uncertainties
   
    #plotting graph
    x = np.linspace(np.min(X),np.max(X),100)    
    y = funcLine(x, theta_hat[0],theta_hat[1])
    y1 = funcLine(x, theta_hat[0]+uncertainties[0],theta_hat[1]+uncertainties[1])
    y2 = funcLine(x, theta_hat[0]-uncertainties[0],theta_hat[1]-uncertainties[1])
    
    fig, ax = plt.subplots(figsize=(5, 3), dpi = 150)
    ax.errorbar(X, Ymeas, yerr=1, fmt='b.', label= 'Data points')
    ax.plot(x, y, 'r-', label='Linear fit')
    ax.plot(x, y1, alpha=0)
    ax.plot(x, y2, alpha=0)
    ax.fill_between(x, y1, y2, facecolor="gray", alpha=0.4, label='Uncertainty')
    ax.set_title('a = %f ± %f and b = %f ± %f' %(theta_hat[0], uncertainties[0], theta_hat[1],uncertainties[1]), size = 10)
    ax.set_xlabel('X values')
    ax.set_ylabel('Y values')
    ax.legend(loc=(0.05, 0.65))
    ax.tick_params(direction="in")

#Built in fitting tool to solve linear LS
def task1b():
    data = np.loadtxt("linear_LS.txt")
    X = data[:, 0]
    Ymeas = data[:, 1]
    sigma_noise = np.ones(len(X)) #std dev of noise for each point
    
    #using scipy curve_fit
    constantsLine, uncertainties = sc.optimize.curve_fit (funcLine, X, Ymeas, sigma=sigma_noise, absolute_sigma=True)
    uncertainties = (np.diagonal(uncertainties))**0.5
    
    #plotting graph
    X1 = np.linspace(X.min(),X.max(),100)
    Y1 = funcLine(X1, *constantsLine)
    
    fig, ax = plt.subplots(figsize=(5, 3), dpi = 150)
    ax.plot(X, Ymeas, 'rx', label = 'Data points')
    ax.plot(X1, Y1, 'b-', label='Linear approximation')
    ax.set_title('A = %f ± %f and B = %f ± %f' %(constantsLine[0], uncertainties[0], constantsLine[1],uncertainties[1]), size = 10)
    ax.set_xlabel('X values')
    ax.set_ylabel('Y values')
    ax.legend(loc=(0.05, 0.75))
    ax.tick_params(direction="in")

#different numbers of data points
def task1c():
    Nruns = 1000 #number of runs
    avalues = np.empty([Nruns, 9]) #values for a
    bvalues = np.empty([Nruns, 9]) #values for b 
    auncertainties = np.empty([Nruns, 9]) #uncertainties for a
    buncertainties = np.empty([Nruns, 9]) #uncertainties for b
    
    for j in range(Nruns):     
        for i in range(9): #looping over different amounts of data points
            data = np.loadtxt("linear_LS.txt")
            
            values = random.sample(range(10),8-i) #selecting random sample
            data = np.delete(data,values,0) #removing data points
            
            X = data[:, 0]
            Ymeas = data[:, 1]
            sigma_noise = 1
            Z = Ymeas
            
            #solving linear LS as before
            H = np.column_stack((X, np.ones(len(X))))
            HT = np.transpose(H) 
            theta_hat = np.linalg.inv(HT@H)@HT@Z 
            Cth = sigma_noise**2*np.linalg.inv(HT@H)
            uncertainties = (np.diagonal(Cth))**0.5
            
            #updating arrays
            avalues[j,i] = theta_hat[0]
            bvalues[j,i] = theta_hat[1]
            auncertainties[j,i] = uncertainties[0]
            buncertainties[j,i] = uncertainties[1]
    
    #determining mean values
    meana = np.mean(avalues, axis = 0)
    meanb = np.mean(bvalues, axis = 0)
    meanuna = np.mean(auncertainties, axis = 0)
    meanunb = np.mean(buncertainties, axis = 0)
    
    #plotting graph
    fig, ax = plt.subplots(figsize=(6, 4), dpi = 150)
    ax.plot(np.arange(2, 11, 1), meanuna, 'b--', label ='Uncertainty in A')
    ax.plot(np.arange(2, 11, 1), meanunb, 'r--', label ='Uncertainty in B')
    ax.plot(np.arange(2, 11, 1), meana, 'g-', label ='A')
    ax.plot(np.arange(2, 11, 1), meanb, 'y-', label ='B')
    ax.set_xlabel('No. of data points (randomly selected from full set)')
    ax.set_ylabel('Mean (%d runs)' %Nruns)
    ax.legend(loc=(0.55, 0.6))
    ax.tick_params(direction="in")
            
 
#Calculating weighted linear least squares
def task2(weighted):
    data = np.loadtxt("weighted_LS.txt")
    
    X = data[:, 0]
    Ymeas = data[:, 1]
    Z = Ymeas
    label=''
    
    #adjusting noise covariance matrix depending on if weighted or not
    if weighted == 1:
        weighting = [0.1**-2 for i in range(5)]
        weighting.extend([10**-2 for i in range(5,10)])
        weighting = np.diag(weighting)
        label = '(weighted)'
    else:
        weighting = [1 for i in range(len(X))]
        weighting = np.diag(weighting)
    
    #solving weighted LS equations
    H = np.column_stack((X**2, np.log(X), np.ones(len(X))))
    HT = np.transpose(H) 
    theta_hat = np.linalg.inv(HT@weighting@H)@HT@weighting@Z 
    Cth = np.linalg.inv(HT@weighting@H)
    uncertainties = (np.diagonal(Cth))**0.5
    
    #plotting graph
    x = np.linspace(np.min(X),np.max(X),100)
    y = funcWeighted(x, theta_hat[0],theta_hat[1],theta_hat[2])
    y1 = funcWeighted(x, theta_hat[0]+uncertainties[0],theta_hat[1]+uncertainties[1],theta_hat[2]+uncertainties[2])
    y2 = funcWeighted(x, theta_hat[0]-uncertainties[0],theta_hat[1]-uncertainties[1],theta_hat[2]-uncertainties[2])

    fig, ax = plt.subplots(figsize=(5, 3), dpi = 150)
    ax.plot(X, Ymeas, 'bx', label ='Data points')
    ax.plot(x, y, 'r-', label ='Linear fit %s' %label)
    ax.plot(x, y1, alpha=0)
    ax.plot(x, y2, alpha=0)
    ax.fill_between(x, y1, y2, facecolor="gray", alpha=0.4, label='Uncertainty')
    ax.set_title('A = %f ± %f , B = %f ± %f and C = %f ± %f' %(theta_hat[0], uncertainties[0], theta_hat[1],uncertainties[1], theta_hat[2],uncertainties[2]), size=9)
    ax.set_xlabel('X values')
    ax.set_ylabel('Y values')
    ax.legend(loc=(0.15, 0.65))
    ax.tick_params(direction="in")

#unweighted task2
def task2a():
    task2(0)

#weighted task2    
def task2b():
    task2(1)

#Calculating non-linear LS
def task3b():
    data = np.loadtxt("nonlinear_LS.txt")
    
    X = data[:, 0]
    Ymeas = data[:, 1]
    Z = Ymeas
    
    theta = np.array([[3],[1]]) #initial solution guess values
    kappa = 0.01 #step parameter
    Niter = 0 #count for number of iterations
    sigma_noise = 1000 #std dev of measurement noise
    contest = 1 #convergence test value
    alist=[3]
    blist=[1]
    
    while contest > 10**-6 or 10**-15 > contest: #loop whilst criteria met
        Niter += 1
        Htheta = X*np.exp(theta[0])+theta[1]*np.exp(X) #Htheta matrix
        
        J = np.column_stack((X*np.exp(theta[0]), np.exp(X))) #jacobian matrix
        JT = np.transpose(J) #transpose of jacobian matrix
        
        #determining change in theta, and then updating theta
        deltatheta = kappa*(np.linalg.inv(JT@J)@JT@(np.array([Z - Htheta]).T))
        theta = theta + deltatheta
        
        contest = abs(sum(deltatheta)) #updating convergence test value
        
        alist.extend(theta[0])
        blist.extend(theta[1])
        
    Cth = sigma_noise**2*np.linalg.inv(JT@J)
    uncertainties = (np.diagonal(Cth))**0.5
    
    #plotting graphs
    x = np.linspace(np.min(X),np.max(X),100)
    y = funcNonlinear(x, theta[0],theta[1])
    y1 = funcNonlinear(x, theta[0]+uncertainties[0],theta[1]+uncertainties[1])
    y2 = funcNonlinear(x, theta[0]-uncertainties[0],theta[1]-uncertainties[1])    
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,figsize=(10.5, 3), dpi = 150)
    
    ax1.plot(np.linspace(1,Niter+1,Niter+1), alist, 'b-', label ='A')
    ax1.plot(np.linspace(1,Niter+1,Niter+1), blist, 'r-', label ='B')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Magnitude of value')
    ax1.legend(loc=(0.15, 0.65))
    ax1.tick_params(direction="in")
    fig.suptitle('A = %f ± %f and B = %f ± %f' %(theta[0], uncertainties[0], theta[1],uncertainties[1]))

    ax2.plot(X, Ymeas, 'bx', label='Data points')
    ax2.plot(x, y, 'r-', label='Non-linear fitting')
    ax2.plot(x, y1, alpha=0)
    ax2.plot(x, y2, alpha=0)
    ax2.fill_between(x, y1, y2, facecolor="gray", alpha=0.4, label='Uncertainty')
    ax2.set_xlabel('X values')
    ax2.set_ylabel('Y values')
    ax2.legend(loc=(0.42, 0.12))
    ax2.tick_params(direction="in")
    
    print(Niter)

#Changing variable to use linear LS for model used in non-linear case    
def task3blinear():
    data = np.loadtxt("nonlinear_LS.txt")
    
    X = data[:, 0]
    Ymeas = data[:, 1]
    Z = Ymeas
    sigma_noise = 1000
    
    H = np.column_stack((X, np.exp(X)))
    HT = np.transpose(H) 
    theta_hat = np.linalg.inv(HT@H)@HT@Z 
    
    Cth = sigma_noise**2*np.linalg.inv(HT@H)
    uncertainties = (np.diagonal(Cth))**0.5
    
    x = np.linspace(np.min(X),np.max(X),100)
    y = funcNonlinear2(x, theta_hat[0],theta_hat[1]) #function with new variable
    
    #plotting graphs
    fig, ax = plt.subplots(figsize=(6, 4), dpi = 150)
    ax.plot(X, Ymeas, 'bx', label='Data points')
    ax.plot(x, y, 'r-', label='Linear fitting')
    ax.set_xlabel('X values')
    ax.set_ylabel('Y values')
    ax.legend(loc=(0.05, 0.8))
    ax.tick_params(direction="in")
    ax.set_title('C = %f ± %f and B = %f ± %f' %(theta_hat[0], uncertainties[0], theta_hat[1],uncertainties[1]))
    
    print('A = %f ± %f' %(np.log(theta_hat[0]), uncertainties[0]/theta_hat[0]))    
   
#looping over different intial guesses and plotting results in 3D    
def task3c():
    data = np.loadtxt("nonlinear_LS.txt")
    
    X = data[:, 0]
    Ymeas = data[:, 1]
    Z = Ymeas
    
    kappa = 0.01
    sigma_noise = 1000
    
    #inital guess values
    initial0 = 2
    initial1 = -10
    
    interval = 1 #interval between guess values
    npoints = 20 #amount of each guess value
    
    #creating arrays for 3D plot
    Xdata = np.linspace(initial0, initial0 + (npoints-1)*interval, npoints)
    Ydata = np.linspace(initial1, initial1 + (npoints-1)*interval, npoints)
    Xdata, Ydata = np.meshgrid(Xdata, Ydata)
    Zdata = np.empty([npoints, npoints])
    
    #looping over both initial guess values and performing non-linear LS as before
    for j in range(npoints):
        for i in range(npoints):
            Niter = 0
            theta = np.array([[initial0 + j*interval],[initial1 + i*interval]])
            contest = 1
            while contest > 10**-12 or 10**-15 > contest:
                Niter += 1
                Htheta = X*np.exp(theta[0])+theta[1]*np.exp(X)
                
                J = np.column_stack((X*np.exp(theta[0]), np.exp(X)))
                JT = np.transpose(J)
                
                deltatheta = kappa*(np.linalg.inv(JT@J)@JT@(np.array([Z - Htheta]).T))
                theta = theta + deltatheta
                contest = abs(sum(deltatheta))
                
            Cth = sigma_noise**2*np.linalg.inv(JT@J)
            uncertainties = (np.diagonal(Cth))**0.5
            
            Zdata[i,j] = Niter #updating number of iterations in relevant place
        print(100*(j+1)/npoints) #progress counter (%)
    
    #plotting graph
    fig = plt.subplots(figsize=(6, 4), dpi = 150)
    ax = plt.axes(projection='3d')
    ax.view_init(azim=285)
    ax.plot_surface(Xdata, Ydata, Zdata, rstride=1, cstride=1, cmap='plasma', edgecolor='none')
    ax.set_xlabel('Initial guess for A')
    ax.set_ylabel('Initial guess for B')
    ax.set_zlabel('Number of iterations')
    plt.tight_layout()

#non-linear using scipy curve_fit
def task3d():
    data = np.loadtxt("nonlinear_LS.txt")
    X = data[:, 0]
    Ymeas = data[:, 1]
    sigma_noise = 1000*np.ones(len(X)) #std dev of noise for each point
    
    constantsLine, uncertainties = sc.optimize.curve_fit (funcNonlinear, X, Ymeas, sigma=sigma_noise, absolute_sigma=True)
    uncertainties = (np.diagonal(uncertainties))**0.5
    
    X1 = np.linspace(X.min(),X.max(),100)
    Y1 = funcNonlinear(X1, *constantsLine)
    
    #plotting graph
    fig, ax = plt.subplots(figsize=(6, 4), dpi = 150)
    ax.plot(X, Ymeas, 'rx', label = 'Data points')
    ax.plot(X1, Y1, 'b-', label='Linear approximation')
    ax.set_title('A = %f ± %f and B = %f ± %f' %(constantsLine[0], uncertainties[0], constantsLine[1],uncertainties[1]), size = 10)
    ax.set_xlabel('X values')
    ax.set_ylabel('Y values')
    ax.legend(loc=(0.05, 0.75))
    ax.tick_params(direction="in")
