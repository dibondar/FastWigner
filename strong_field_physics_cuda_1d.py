__doc__ = "This file is an example how to use FFTW implementations of phase space dynamics to study" \
          "strong field physics (ionization due to fs laser pulese)"

from wigner_bloch_cuda_1d import WignerMoyalCUDA1D, WignerBlochCUDA1D
import numpy as np

# load tools for creating animation
import sys
import matplotlib

if sys.platform == 'darwin':
    # only for MacOS
    matplotlib.use('TKAgg')

import matplotlib.animation
import matplotlib.pyplot as plt

##########################################################################################
#
#   Parameters of quantum systems
#
##########################################################################################
sys_params = dict(
    t=0.,
    dt=0.01,

    X_gridDIM=512,

    # the lattice constant is 2 * X_amplitude
    X_amplitude=4.,

    # Lattice height
    V0=0.37,

    P_gridDIM=1024,
    P_amplitude=15.,

    # Temperature in atomic units
    kT=0.03,

    # Decay constant in the random collision model
    _gamma=0.01,

    # frequency of laser field (800nm)
    omega=0.05698,

    # field strength
    F=0, #0.06,

    functions="""
    // # The vector potential of laser field (the field will be on for 8 periods of laser field)
    __device__ double A(double t)
    {{
        return -F / omega * sin(omega * t) * pow(sin(omega * t / 16.), 2);
    }}
    """,

    abs_boundary_x_theta="exp(-dt * 0.0001 * Theta * Theta)",
    #abs_boundary_lambda_p="exp(-dt * 0.05 * Lambda * Lambda)",
    #abs_boundary_x_p="pow(abs(sin(0.5 * M_PI * (P + P_amplitude) / P_amplitude)), 2*dt*0.03)",

    # The same as C code
    A=lambda self, t: -self.F/self.omega * np.sin(self.omega * t) * np.sin(self.omega * t / 16.)**2,

    ##########################################################################################
    #
    # Specify system's hamiltonian
    #
    ##########################################################################################

    # the kinetic energy
    K="0.5 * pow(P + A(t), 2)",

    # derivative of the kinetic energy to calculate Ehrenfest
    diff_K="P + A(t)",

    # Mathieu-type periodic system
    V = "-V0 * (1. + cos(M_PI * (X + X_amplitude) / X_amplitude))",

    # the derivative of the potential to calculate Ehrenfest
    diff_V="V0 * M_PI / X_amplitude * sin(M_PI * (X + X_amplitude) / X_amplitude)",
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
            % (1. / self.quant_sys._gamma, self.quant_sys.kT)
        )
        extent = [self.quant_sys.X.min(), self.quant_sys.X.max(), self.quant_sys.P.min(), self.quant_sys.P.max()]

        # import utility to visualize the wigner function
        from wigner_normalize import WignerNormalize, WignerSymLogNorm

        # generate empty plot
        self.img = ax.imshow(
            [[]],
            extent=extent,
            origin='lower',
            aspect=0.2,
            cmap='seismic',
            norm=WignerSymLogNorm(linthresh=1e-10, vmin=-0.3, vmax=0.3),
            #norm=WignerNormalize(vmin=-0.01, vmax=0.1)
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
        self.quant_sys = WignerMoyalCUDA1D(**self.sys_params)

        # List to save times
        self.times = [self.quant_sys.t]

        # set the Gibbs state as initial condition
        self.quant_sys.set_wignerfunction(
            WignerBlochCUDA1D(**self.sys_params).get_gibbs_state()
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
        self.img.set_array(self.quant_sys.propagate(500).get())

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
#writer = matplotlib.animation.writers['mencoder'](fps=5, metadata=dict(artist='Denys Bondar'))

# Save animation into the file
#animation.save('strong_field_physics.mp4', writer=writer)

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
# Plot HHG spectra as FFT(<P>)
#
#################################################################

N = len(quant_sys.P_average)

# the windowed fft of the evolution
# to remove the spectral leaking. For details see
# rhttp://docs.scipy.org/doc/scipy/reference/tutorial/fftpack.html
from scipy import fftpack
from scipy.signal import blackman

fft_P_average = fftpack.fft(blackman(N) * np.array(quant_sys.P_average))
spectrum = np.abs(fftpack.fftshift(fft_P_average))**2
omegas = fftpack.fftshift(fftpack.fftfreq(N, quant_sys.dt/(2*np.pi))) / quant_sys.omega

plt.semilogy(omegas, spectrum)
plt.ylabel('spectrum FFT($\\langle p \\rangle$)')
plt.xlabel('frequency / $\\omega$')
plt.xlim([0, 100.])
#plt.ylim([1e-14, 1e3])

plt.show()