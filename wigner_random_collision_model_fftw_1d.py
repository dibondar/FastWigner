from wigner_moyal_fftw_1d import WignerMoyalFTTW1D
from wigner_bloch_fftw_1d import WignerBlochFFTW1D
import numpy as np


class WignerRandomCollisionModelFFTW1D(WignerMoyalFTTW1D):
    """
    Second-order split operator propagator for the random collision model of open system dynamics.
    In paricular, the Wigner function obeys the following equation

        dW / dt = {{H, W}} + gamma (W_0 - W)

    where {{ , }} is the Moyal bracket, W_0 is a Gibbs-Boltzmann state, and
    gamma characterize the collision rate (i.e., 1/gamma is dephasing time).
    """
    def __init__(self, **kwargs):
        """
        In addition to kwagrs of WignerMoyalFTTW1D.__init__ this constructor accepts:

            gamma - decay rate
        """
        # Initialize the parent class
        WignerMoyalFTTW1D.__init__(self, **kwargs)

        # Make sure gamma was specified
        try:
            self.gamma
        except AttributeError:
            raise AttributeError("Collision rate (gamma) was not specified")

        # Pre-calculate the constants needed for the dissipator propagation
        self.w_coeff = np.exp(-self.gamma * 0.5 * self.dt)

        # Calculate the Gibbs state (W_0)
        self.scaled_gibbs_state = WignerBlochFFTW1D(**kwargs).get_gibbs_state()

        # Then scale it correspondingly
        self.scaled_gibbs_state *= 1 - self.w_coeff

        if self.isEhrenfest:
            # if the Ehrenfest theorems are to be calculated,
            # then calculate some statistics for the Gibbs state

            # calculate the coordinate density
            density_coord = self.scaled_gibbs_state.sum(axis=0)
            density_coord /= density_coord.sum()

            self.X_average_gibbs = np.dot(density_coord, self.X.reshape(-1))

            # calculate the momentum density
            density_momentum = self.scaled_gibbs_state.sum(axis=1)
            density_momentum /= density_momentum.sum()

            self.P_average_gibbs = np.dot(density_momentum, self.P.reshape(-1))

    def single_step_propagation(self):
        """
        Overload the method WignerMoyalFTTW1D.single_step_propagation.
        Perform single step propagation. The final Wigner function is not normalized.
        :return: self.wignerfunction
        """

        # Dissipation requites to updated W as
        #   W = W0 * (1 - exp(-gamma*0.5*dt)) + exp(-gamma*0.5*dt) * W,
        self.wignerfunction *= self.w_coeff
        self.wignerfunction += self.scaled_gibbs_state

        # follow the unitary evolution
        WignerMoyalFTTW1D.single_step_propagation(self)

        # Dissipation requites to updated W as
        #   W = W0 * (1 - exp(-gamma*0.5*dt)) + exp(-gamma*0.5*dt) * W,
        self.wignerfunction *= self.w_coeff
        self.wignerfunction += self.scaled_gibbs_state

        return self.wignerfunction

    def get_Ehrenfest(self, t):
        """
        Overload the method WignerMoyalFTTW1D.get_Ehrenfest.
        Update the Ehrenfest theorems to account for open system interaction.
        """
        if self.isEhrenfest:

            # Calculate the Ehrenfest theorems for unitary dynamics
            result = WignerMoyalFTTW1D.get_Ehrenfest(self, t)

            # Amend the first Ehrenfest theorem
            self.X_average_RHS[-1] += self.gamma * (self.X_average_gibbs - self.X_average[-1])

            # Amend the second Ehrenfest theorem
            self.P_average_RHS[-1] += self.gamma * (self.P_average_gibbs - self.P_average[-1])

            return result

##########################################################################################
#
# Example
#
##########################################################################################

if __name__ == '__main__':

    print(WignerRandomCollisionModelFFTW1D.__doc__)

    # load tools for creating animation
    import sys
    import matplotlib

    if sys.platform == 'darwin':
        # only for MacOS
        matplotlib.use('TKAgg')

    import matplotlib.animation
    import matplotlib.pyplot as plt

    class VisualizeDynamicsPhaseSpace:
        """
        Class to visualize the random collision dynamics in phase space.
        """
        def __init__(self, fig):
            """
            Initialize all propagators and frame
            :param fig: matplotlib figure object
            """
            #  Initialize systems
            self.set_quantum_sys()

            #################################################################
            #
            # Initialize plotting facility
            #
            #################################################################

            self.fig = fig

            ax = fig.add_subplot(111)

            ax.set_title(
                'Wigner function evolution $W(x,p,t)$\nwith $\\gamma^{-1} = $ %.2f and $kT = $ %.2f (a.u.)'
                % (1. / self.quant_sys.gamma, self.quant_sys.kT)
            )
            extent = [self.quant_sys.X.min(), self.quant_sys.X.max(), self.quant_sys.P.min(), self.quant_sys.P.max()]

            # import utility to visualize the wigner function
            from wigner_normalize import WignerNormalize

            # generate empty plot
            self.img = ax.imshow(
                [[]],
                extent=extent,
                origin='lower',
                cmap='seismic',
                norm=WignerNormalize(vmin=-0.01, vmax=0.1)
            )

            self.fig.colorbar(self.img)

            ax.set_xlabel('$x$ (a.u.)')
            ax.set_ylabel('$p$ (a.u.)')

        def set_quantum_sys(self):
            """
            Initialize quantum propagator
            :param self:
            :return:
            """
            # Create propagator
            self.quant_sys = WignerRandomCollisionModelFFTW1D(
                t=0,
                dt=0.01,
                X_gridDIM=512,
                X_amplitude=10.,
                P_gridDIM=512,
                P_amplitude=10,

                # randomized collision rate
                gamma=np.random.uniform(0.1, 1.),

                # randomized temperature
                kT=np.random.uniform(0.01, 1.),

                # randomized parameter
                omega_square=np.random.uniform(0.5, 3),

                # parameter controlling the width of the initial wigner function
                sigma=np.random.uniform(0.5, 4.),

                # kinetic energy part of the hamiltonian
                K=lambda _, p: 0.5 * p ** 2,

                # potential energy part of the hamiltonian
                V=lambda self, x: 0.5 * self.omega_square * x ** 2,

                # these functions are used for evaluating the Ehrenfest theorems
                diff_K=lambda _, p: p,
                diff_V=lambda self, x: self.omega_square * x
            )

            # set randomised initial condition
            self.quant_sys.set_wignerfunction(
                lambda self, x, p: np.exp(
                    # randomized position
                    -self.sigma * (x + np.random.uniform(-3., 3.)) ** 2
                    # randomized initial velocity
                    -(1. / self.sigma) * (p + np.random.uniform(-3., 3.)) ** 2
                )
            )

        def empty_frame(self):
            """
            Make empty frame and reinitialize quantum system
            :param self:
            :return: image object
            """
            self.img.set_array([[]])
            return self.img,

        def __call__(self, frame_num):
            """
            Draw a new frame
            :param frame_num: current frame number
            :return: image objects
            """
            # propagate the wigner function
            self.img.set_array(self.quant_sys.propagate(20))
            return self.img,


    fig = plt.gcf()
    visualizer = VisualizeDynamicsPhaseSpace(fig)
    animation = matplotlib.animation.FuncAnimation(
        fig, visualizer, frames=np.arange(100), init_func=visualizer.empty_frame, repeat=True, blit=True
    )

    plt.show()

    # Set up formatting for the movie files
    # writer = matplotlib.animation.writers['mencoder'](fps=5, metadata=dict(artist='Denys Bondar'))

    # Save animation into the file
    # animation.save('harmonic_oscilator_random_collision.mp4', writer=writer)

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
    plt.title("The first Ehrenfest theorem verification")

    plt.plot(times, np.gradient(quant_sys.X_average, dt), 'r-', label='$d\\langle x \\rangle/dt$')
    plt.plot(times, quant_sys.X_average_RHS, 'b--', label='$\\langle p + \\gamma x \\rangle$')

    plt.legend()
    plt.xlabel('time $t$ (a.u.)')

    plt.subplot(132)
    plt.title("The second Ehrenfest theorem verification")

    plt.plot(times, np.gradient(quant_sys.P_average, dt), 'r-', label='$d\\langle p \\rangle/dt$')
    plt.plot(
        times, quant_sys.P_average_RHS, 'b--',
        label='$\\langle -\\partial V/\\partial x  + \\gamma p \\rangle$'
    )

    plt.legend()
    plt.xlabel('time $t$ (a.u.)')

    plt.subplot(133)
    plt.title('Hamiltonian')
    plt.semilogy(times, quant_sys.hamiltonian_average)
    plt.xlabel('time $t$ (a.u.)')

    plt.show()