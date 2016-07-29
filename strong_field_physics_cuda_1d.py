__doc__ = "This file is an example how to use FFTW implementations of phase space dynamics to study" \
          "strong field physics (ionization due to fs laser pulese)"

from rho_bloch_cuda_1d import RhoBlochCUDA1D
import numpy as np

# load tools for creating animation
import sys
import matplotlib
import h5py

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

    X_gridDIM=2*1024,

    # the lattice constant is 2 * X_amplitude
    X_amplitude=150.,

    # Temperature in atomic units
    #kT=0.001,
    #kT = 0.001,

    # Decay constant in the random collision model
    _gamma=0.01,

    # frequency of laser field (800nm)
    omega=0.05698,

    # field strength
    F=0.04,

    # ionization potential
    Ip=0.59,

    functions="""
    // # The laser field (the field will be on for 8 periods of laser field)
    __device__ double E(double t)
    {{
        return -F * sin(omega * t) * pow(sin(omega * t / 16.), 2);
    }}
    """,

    abs_boundary_x="pow("
                   "    abs(sin(0.5 * M_PI * (X + X_amplitude) / X_amplitude)"
                   "    * sin(0.5 * M_PI * (X_prime + X_amplitude) / X_amplitude))"
                   ", dt * 0.05)",

    # The same as C code
    E=lambda self, t: -self.F * np.sin(self.omega * t) * np.sin(self.omega * t / 16.)**2,

    ##########################################################################################
    #
    # Specify system's hamiltonian
    #
    ##########################################################################################

    # the kinetic energy
    K="0.5 * P * P",

    # derivative of the kinetic energy to calculate Ehrenfest
    diff_K="P",

    # the soft core Coulomb potential for Ar
    V = "-1. / sqrt(X * X + 1.37) + X * E(t)",

    # the derivative of the potential to calculate Ehrenfest
    diff_V="X / pow(X * X + 1.37, 1.5) + E(t)",
)


class VisualizeDynamicsPhaseSpace:
    """
    Class to visualize dynamics in phase space.
    """

    def __init__(self, fig, sys_params, file_results):
        """
        Initialize all propagators and frame
        :param fig: matplotlib figure object
        :param sys_params: dictionary of parameters to initialize quantum system
        :param file_results: HDF5 file to save results
        """
        self.fig = fig
        self.sys_params = sys_params
        self.file_results = file_results

        #  Initialize systems
        self.set_quantum_sys()

        #################################################################
        #
        # Save quantum system's parameters into the HDF5 file
        #
        #################################################################

        self.settings_grp = self.file_results.create_group("settings")

        for key, val in sys_params.items():
            try:
                self.settings_grp[key] = val
            except TypeError:
                pass

        self.settings_grp["X_wigner"] = self.quant_sys.X_wigner
        self.settings_grp["P_wigner"] = self.quant_sys.P_wigner

        # Create group where each frame will be saved
        self.frames_grp = self.file_results.create_group("frames")

        #################################################################
        #
        # Initialize plotting facility
        #
        #################################################################

        ax = fig.add_subplot(211)

        #ax.set_title(
        #    'Wigner function evolution $W(x,p,t)$\nwith $\\gamma^{-1} = $ %.2f and $kT = $ %.2f (a.u.)'
        #    % (1. / self.quant_sys._gamma, self.quant_sys.kT)
        #)
        ax.set_title("Wigner function")

        extent = [
            self.quant_sys.X_wigner.min(), self.quant_sys.X_wigner.max(),
            self.quant_sys.P_wigner.min(), self.quant_sys.P_wigner.max()
        ]

        # import utility to visualize the wigner function
        from wigner_normalize import WignerNormalize, WignerSymLogNorm

        # generate empty plot
        self.img = ax.imshow(
            [[]],
            extent=extent,
            origin='lower',
            aspect=4,
            cmap='bwr',
            #norm=WignerSymLogNorm(linthresh=1e-4, vmin=-0.3, vmax=0.3),
            norm=WignerNormalize(vmin=-0.001, vmax=0.001)
        )

        ax.set_xlim([-25, 25])
        ax.set_ylim([-3, 3])

        self.fig.colorbar(self.img)

        ax.set_xlabel('$x$ (a.u.)')
        ax.set_ylabel('$p$ (a.u.)')

        ax = fig.add_subplot(212)
        F = self.quant_sys.F
        self.laser_filed_plot, = ax.plot([0., self.T_final], [-F, F])
        ax.set_xlabel('time (a.u.)')
        ax.set_ylabel('Laser field $E(t)$ (a.u.)')

    def set_quantum_sys(self):
        """
        Initialize quantum propagator
        :param self:
        :return:
        """
        # Create propagator
        self.quant_sys = RhoBlochCUDA1D(**self.sys_params)

        # Constant specifying the duration of simulation

        # final propagation time
        self.T_final = 8 * 2 * np.pi / self.quant_sys.omega

        # Number of steps before plotting
        self.num_iteration = 100

        # Number of frames
        self.num_frames = int(np.ceil(self.T_final / self.quant_sys.dt / self.num_iteration))

        self.current_frame_num = 0

        # List to save times
        self.times = [self.quant_sys.t]

        # set the Gibbs state as initial condition
        self.quant_sys.get_ground_state(abs_tol_purity=1e-11)

        # save the initial wigner function
        r = self.quant_sys.get_wignerfunction()
        self.quant_sys.wigner_initial = r.real

        assert r is not self.quant_sys.wigner_initial

        print("Purity: 1 - %.1e" % (1 - self.quant_sys.get_purity()))

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
        # propagate
        self.quant_sys.propagate(self.num_iteration)

        # prepare goup where simulations for the current frame will be saved
        frame_grp = self.frames_grp.create_group(str(self.current_frame_num))

        # Get the Wigner function
        wigner = self.quant_sys.wigner_current.get().real
        self.img.set_array(wigner)

        frame_grp["wigner"] = wigner
        frame_grp["t"] = self.quant_sys.t

        # Extract the diagonal of the density matrix
        frame_grp["prob"] = self.quant_sys.rho.get().diagonal()

        #print("Purity: 1 - %.1e" % (1 - self.quant_sys.get_purity()))

        print("Frame : %d / %d" % (self.current_frame_num, self.num_frames))
        self.current_frame_num += 1

        self.times.append(self.quant_sys.t)

        t = np.array(self.times)
        self.laser_filed_plot.set_data(t, self.quant_sys.E(t))

        return self.img, self.laser_filed_plot

with h5py.File('strong_field_physics.hdf5', 'w') as file_results:
    fig = plt.gcf()
    visualizer = VisualizeDynamicsPhaseSpace(fig, sys_params, file_results)
    animation = matplotlib.animation.FuncAnimation(
        fig, visualizer, frames=min(881, visualizer.num_frames),
        init_func=visualizer.empty_frame, blit=True, repeat=True
    )

    #plt.show()

    # Set up formatting for the movie files
    writer = matplotlib.animation.writers['mencoder'](fps=20, metadata=dict(artist='Denys Bondar'), bitrate=-1)

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

    """
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

    # obtain the dipole
    J = np.array(quant_sys.P_average)

    fft_J = fftpack.fft(blackman(N) * J)
    #fft_J = fftpack.fft(J)
    spectrum = np.abs(fftpack.fftshift(fft_J))**2
    omegas = fftpack.fftshift(fftpack.fftfreq(N, quant_sys.dt/(2*np.pi))) / quant_sys.omega


    spectrum /= spectrum.max()

    plt.semilogy(omegas, spectrum)
    plt.ylabel('spectrum FFT($\\langle p \\rangle$)')
    plt.xlabel('frequency / $\\omega$')
    plt.xlim([0, 100.])
    plt.ylim([1e-20, 1.])

    plt.show()
    """

    #################################################################
    #
    # Saving Ehrenfest theorem results into HDF5 file
    #
    #################################################################

    ehrenfest_grp = file_results.create_group("ehrenfest")
    ehrenfest_grp["X_average"] = quant_sys.X_average
    ehrenfest_grp["P_average"] = quant_sys.P_average
    ehrenfest_grp["X_average_RHS"] = quant_sys.X_average_RHS
    ehrenfest_grp["P_average_RHS"] = quant_sys.P_average_RHS
    ehrenfest_grp["hamiltonian_average"] = quant_sys.hamiltonian_average
    ehrenfest_grp["wigner_time"] = quant_sys.wigner_time