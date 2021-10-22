import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class string:
    
    def __init__(self, N, L=1, masses=None, springs=None):
        '''
        Create a vibrating string of length 1m, made of N harmonic oscillators
        with masses and springs values

        Arguments:
        ----------
        N       [int]          : number of oscillators
        L       [float]        : length of the string 
        masses  [list of float]: masses of each oscillator (should have N numbers)
        springs [list of float]: springs values (should have N+1 numbers)
        '''

        # Default values
        if masses is None:
            masses = np.array([1.0]*N)
        if springs is None:
            springs = np.array([1.0]*(N+1))
        
        # Sanity checks
        if len(masses) != N:
            raise NameError(f'Masses should have {N} elements, while it has {len(masses)}.')

        if len(springs) != N+1:
            raise NameError(f'Springs should have {N} elements, while it has {len(springs)}.')

        # Basic variable
        self.N  = N
        self.L  = L
        self.Ms = masses
        self.Ks = springs

        # Computed variables
        self.dL = L/N
        self.Xs = np.array([(i+1)*self.dL for i in range(self.N)])

    
    def motion_equations(self, q, t, p):
        '''
        Return the differential equation of motion to be feed
        into scipy.integrate.odeint().
        
        Arguments:
        ----------
        q : vector of variables describing the system at a given time.
            numpy array of float with 2*N numbers (xi=positions, yi=velocities):
                 y = [x1, y1, x2, y2, ... xn, yn]
        t [float]:  time
        p [list of float]: vector of the parameters, being (mi, ki) and
                           one more k for the list oscillator.
                              p = [m1, k1, m1, k1, ... mn, kn, kn+1]
        '''
        
        # Sanity checks
        if len(q)%2 != 0:
            raise NameError('System state should contain an even number of coordinate.')
        if len(q)/2 != self.N:
            raise NameError(f'System state should contain 2N number, not {len(q)}.')
        
        # Get values
        xs, ys = q[::2], q[1::2]
        ms, ks = p[::2], p[1::2]
        
        # Create system derivate f (twice Ndof: positions & speeds)
        f = [0] * 2*self.N
        
        # First discrete oscillator
        f[0] = ys[0]
        f[1] = -ks[0]/ms[0] * xs[0] + ks[1]/ms[1] * (xs[1]-xs[0])
        
        # Middle oscillator
        for iDof in range(1, self.N-1):
            iEq      = 2*iDof
            f[iEq]   = ys[iDof] 
            f[iEq+1] = - ks[iDof]/ms[iDof] * (xs[iDof] - xs[iDof-1]) + ks[iDof]/ms[iDof] * (xs[iDof+1] - xs[iDof]) 
    
        # Last ocillator
        f[-2] = ys[-1]
        f[-1] = -ks[-2]/ms[-1] * (xs[-1] - xs[-2]) - ks[-1]/ms[-1] * xs[-1]
            
        return f
        
        
    def solved_motion(self, Ts, Y0s, V0s):
        '''
        Return the motion of the N oscillators, which is the positions
        and the velocities of the N oscillators, at all times.

        Arguments:
        ----------
        Ts : numpy array of times at which the
             oscillators state are computed.
        Y0s: initial positions (array of N elements)
        V0s: initial velocities (array of N elements).
        
        Return:
        -------
        Ys: positions at each time of each oscillators, array of shape (len(Ts), N)
        Vs: velocities at each time of each oscillators, array of shape (len(Ts), N)        
        '''
        
        def pack(As, Bs):
            '''
            Pack values of lists As and Bs to return 
               Cs = [As[0], Bs[0], As[1], Bs[1], ...]
            '''
            Cs = []
            for a, b in zip(As, Bs):
                Cs.append(a)
                Cs.append(b)
            return Cs
        
        # Packing everything for the resolution
        p  = pack(self.Ms, self.Ks)
        q0 = pack(Y0s, V0s)

        # Solving the equations
        res = odeint(self.motion_equations, q0, Ts, args=(p,))

        # Extracting position and velocities
        Ys = res[:, ::2]
        Vs = res[:, 1::2]

        # Return positions and velocities
        return Ys, Vs
        

def plot_string(Xs, Ys):
    '''
    Plot the string at given time.
    Xs, Ys are two arrays of size N
    '''
    plt.figure()
    plt.plot(Xs, Ys, 'o')
    plt.xlabel('X')
    plt.ylabel('Y')
    

def animate_string(Xs, Ys, saveName=''):
    '''
    Animate a string given the positions of the N oscillators
    along the x-axis, and Ys, the N positions along the y-axis
    for n time steps.

    Arguments:
    ---------
    Xs: array of shape (N,)
    Ys: array of shape (n, N)
    '''

    # Initialize the figure
    fig = plt.figure()
    line, = plt.plot([],[], 'o') 

    # Cosmetics
    plt.xlim(Xs[0], Xs[-1])
    plt.ylim(Ys.min(), Ys.max())
    plt.ylabel('Y')
    plt.xlabel('X')

    def init():
        line.set_data([],[])
        return line,

    def animate(i): 
        line.set_data(Xs, Ys[i, :])
        return line,
 
    # Launch the animation
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=1000, blit=True, interval=100, repeat=False)
    plt.tight_layout()

    # Set up formatting for the movie files
    if saveName:
        anim.save('{}.gif'.format(saveName), writer='imagemagick', fps=60)
    
    return anim
        
