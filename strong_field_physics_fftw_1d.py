__doc__ = "This file is an example how to use FFTW implementations of phase space dynamics to study" \
          "strong field physics (ionization due to fs laser pulese)"

from wigner_random_collision_model_fftw_1d import \
    WignerMoyalFTTW1D, WignerRandomCollisionModelFFTW1D, WignerBlochFFTW1D
import numpy as np
import pickle
import pyfftw

# load tools for creating animation
import sys
import matplotlib

if sys.platform == 'darwin':
    # only for MacOS
    matplotlib.use('TKAgg')

import matplotlib.animation
import matplotlib.pyplot as plt

# Load FFTW wisdom if saved before
try:
    with open('fftw_wisdom', 'rb') as f:
        pyfftw.import_wisdom(pickle.load(f))

    print("\nFFTW wisdom has been loaded\n")
except IOError:
    pass

##########################################################################################
#
#   Parameters of quantum systems
#
##########################################################################################
sys_params = dict(
    t=0.,
    dt=0.01,

    X_gridDIM=512,

    # the lattice constant
    X_amplitude=8.,

    # Lattice height
    V0=0.37,

    P_gridDIM=512,
    P_amplitude=10,

    # Temperature
    kT=0.03,

    # Decay constant in the random collision model
    gamma=0.01,

    # frequency of laser field (800nm)
    omega=0.05698,

    # field strength
    F=0.04,

    # The vector potential of laser field (the field will be on for 8 periods of laser field)
    A=lambda self, t: -self.F/self.omega * np.sin(self.omega * t) * np.sin(self.omega * t / 16.)**2,

    ##########################################################################################
    #
    # Specify system's hamiltonian
    #
    ##########################################################################################

    # the kinetic energy
    K=lambda self, p, t: 0.5*(p + self.A(t))**2,

    # derivative of the kinetic energy to calculate Ehrenfest
    diff_K=lambda self, p, t: (p + self.A(t)),

    # Mathieu-type periodic system
    V=lambda self, x: -self.V0 * (1 + np.cos(2*np.pi*x/self.X_amplitude)),

    # the derivative of the potential to calculate Ehrenfest
    diff_V=lambda self, x: self.V0 * 2*np.pi/self.X_amplitude * np.sin(2*np.pi*x/self.X_amplitude),

    ##########################################################################################
    #
    # FFTW parameters
    #
    ##########################################################################################
    ffw_flags=('FFTW_MEASURE',),
    fftw_threads=10,
)


class VisualizeDynamicsPhaseSpace:
    """
    Class to visualize dynamics in phase space.
    """
    def __init__(self, fig, sys_params):
        """
        Initialize all propagators and frame
        :param fig: matplotlib figure object
        :param sys_params: dictionary of parameters to initialize quantum system
        """
        self.fig = fig
        self.sys_params = sys_params

        #  Initialize systems
        self.set_quantum_sys()

        #################################################################
        #
        # Initialize plotting facility
        #
        #################################################################

        ax = fig.add_subplot(211)

        ax.set_title(
            'Wigner function evolution $W(x,p,t)$\nwith $\\gamma^{-1} = $ %.2f and $kT = $ %.2f (a.u.)'
            % (1. / self.quant_sys.gamma, self.quant_sys.kT)
        )
        extent = [self.quant_sys.X.min(), self.quant_sys.X.max(), self.quant_sys.P.min(), self.quant_sys.P.max()]

        # import utility to visualize the wigner function
        from wigner_normalize import WignerNormalize, WignerSymLogNorm

        # generate empty plot
        self.img = ax.imshow(
            [[]],
            extent=extent,
            origin='lower',
            aspect=1,
            cmap='seismic',
            norm=WignerSymLogNorm(linthresh=1e-10, vmin=-0.2, vmax=0.2),
            #norm=WignerNormalize(vmin=-0.01, vmax=1.)
        )

        self.fig.colorbar(self.img)

        ax.set_xlabel('$x$ (a.u.)')
        ax.set_ylabel('$p$ (a.u.)')

        ax = fig.add_subplot(212)
        A0 = self.quant_sys.F/self.quant_sys.omega
        self.laser_filed_plot, = ax.plot([0., 8*2*np.pi/self.quant_sys.omega], [-A0, A0])
        ax.set_xlabel('time (a.u.)')
        ax.set_ylabel('Vector potential $A(t)$ (a.u.)')

    def set_quantum_sys(self):
        """
        Initialize quantum propagator
        :param self:
        :return:
        """
        # Create propagator
        #self.quant_sys = WignerRandomCollisionModelFFTW1D(**self.sys_params)
        self.quant_sys = WignerMoyalFTTW1D(**self.sys_params)


        # List to save times
        self.times = [self.quant_sys.t]

        # set randomised initial condition
        #self.quant_sys.set_wignerfunction(self.quant_sys.scaled_gibbs_state)
        self.quant_sys.set_wignerfunction(
            WignerBlochFFTW1D(**self.sys_params).get_gibbs_state()
        )

    def empty_frame(self):
        """
        Make empty frame and reinitialize quantum system
        :param self:
        :return: image object
        """
        self.img.set_array([[]])
        self.laser_filed_plot.set_data([], [])
        return self.img, self.laser_filed_plot

    def __call__(self, frame_num):
        """
        Draw a new frame
        :param frame_num: current frame number
        :return: image objects
        """
        # propagate the wigner function
        self.img.set_array(self.quant_sys.propagate(100))

        self.times.append(self.quant_sys.t)

        t = np.array(self.times)
        self.laser_filed_plot.set_data(t, self.quant_sys.A(t))

        return self.img, self.laser_filed_plot


fig = plt.gcf()
visualizer = VisualizeDynamicsPhaseSpace(fig, sys_params)
animation = matplotlib.animation.FuncAnimation(
    fig, visualizer, frames=np.arange(100), init_func=visualizer.empty_frame, repeat=True, blit=True
)

plt.show()

# Set up formatting for the movie files
writer = matplotlib.animation.writers['mencoder'](fps=5, metadata=dict(artist='Denys Bondar'))

# Save animation into the file
animation.save('strong_field_physics.mp4', writer=writer)

# extract the reference to quantum system
quant_sys = visualizer.quant_sys

#################################################################
#
# Plot the Ehrenfest theorems after the animation is over
#
#################################################################

# generate time step grid
dt = quant_sys.dt
times = dt * np.arange(len(quant_sys.X_average)) + dt

plt.subplot(131)
plt.title("Ehrenfest 1")
plt.plot(times, np.gradient(quant_sys.X_average, dt), 'r-', label='$d\\langle x \\rangle/dt$')
plt.plot(times, quant_sys.X_average_RHS, 'b--', label='$\\langle p + \\gamma x \\rangle$')

plt.legend(loc='upper left')
plt.xlabel('time $t$ (a.u.)')

plt.subplot(132)
plt.title("Ehrenfest 2")

plt.plot(times, np.gradient(quant_sys.P_average, dt), 'r-', label='$d\\langle p \\rangle/dt$')
plt.plot(
    times, quant_sys.P_average_RHS, 'b--',
    label='$\\langle -\\partial V/\\partial x  + \\gamma p \\rangle$'
)

plt.legend(loc='upper left')
plt.xlabel('time $t$ (a.u.)')

plt.subplot(133)
plt.title('Hamiltonian')
plt.plot(times, quant_sys.hamiltonian_average)
plt.xlabel('time $t$ (a.u.)')

plt.show()

#################################################################
#
# Save FFTW wisdom
#
#################################################################
with open('fftw_wisdom', 'wb') as f:
    pickle.dump(pyfftw.export_wisdom(), f)
