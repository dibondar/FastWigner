from wigner_moyal_cuda_1d import WignerMoyalCUDA1D

import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np


class WignerBlochCUDA1D(WignerMoyalCUDA1D):
    """
    Find the Wigner function of the Maxwell-Gibbs canonical state [rho = exp(-H/kT)]
    by second-order split-operator propagation of the Bloch equation in phase space using CUDA.
    The Hamiltonian should be of the form H = K(p) + V(x).

    This implementation is based on the algorithm described in
        D. I Bondar, A. G. Campos, R. Cabrera, H. A. Rabitz, arXiv:1602.07288
    """
    def __init__(self, **kwargs):
        """
        In addition to kwagrs of WignerMoyalCUDA1D.__init__ this constructor accepts:

        kT - the temperature for the Gibbs state [rho = exp(-H/kT)]
        dbeta (optional) -  inverse temperature increments for the split-operator propagation
        t_initial (optional) - if the Hamiltonian is time dependent, then the the Gibbs state will be calculated
            for the hamiltonian at t_initial (default value of zero).
        """
        if 't_initial' not in kwargs:
            kwargs.update(t_initial=0.)
            print("Warning: Initial time (t_initial) was not specified. So the default value was used t_initial = 0.")

        try:
            self.kT = kwargs['kT']
        except KeyError:
            raise AttributeError("Temperature (kT) was not specified")

        if self.kT > 0:
            try:
                self.dbeta = kwargs['dbeta']
            except KeyError:
                # if dbeta is not defined, just choose some value
                self.dbeta = 0.01

            # get number of dbeta steps to reach the desired Gibbs state
            self.num_beta_steps = 1. / (self.kT*self.dbeta)

            if round(self.num_beta_steps) <> self.num_beta_steps:
                # Changing self.dbeta so that num_beta_steps is an exact integer
                self.num_beta_steps = round(self.num_beta_steps)
                self.dbeta = 1. / (self.kT*self.num_beta_steps)

            self.num_beta_steps = int(self.num_beta_steps)

        else:
            raise NotImplemented("The calculation of the ground state Wigner function has not been implemented")

        # Save the inverse temperature increment also dbeta takes the form of dt
        kwargs.update(dbeta=self.dbeta, dt=self.dbeta)

        # Initialize parent class
        WignerMoyalCUDA1D.__init__(self, **kwargs)

        # Make sure the Ehrenfest theorems are not calculated
        self.isEhrenfest = False

    def get_V_min(self):
        """
        Return the potential energy minimum in the x theta space
        """
        # allocate memory
        v_theta_x = gpuarray.zeros((self.Theta.size, self.X.size), np.float64)

        fill_V = SourceModule(
            self.fill_extended_V.format(cuda_consts=self.cuda_consts, V=self.V),
        ).get_function("Kernel")

        # fill array with values of the potential energy
        fill_V(v_theta_x, **self.expV_mapper_params)

        return gpuarray.min(v_theta_x).get()

    def get_K_min(self):
        """
        Return the kinetic energy minimum in the lambda p space
        """
        # allocate memory
        k_p_lambda = gpuarray.zeros((self.P.size, self.Lambda.size), np.float64)

        fill_K = SourceModule(
            self.fill_extended_K.format(cuda_consts=self.cuda_consts, K=self.K),
        ).get_function("Kernel")

        # fill array with values of the kinetic energy
        fill_K(k_p_lambda, **self.expK_mapper_params)

        return gpuarray.min(k_p_lambda).get()

    def get_gibbs_state(self):
        """
        Calculate the Boltzmann-Gibbs state and save it in self.wignerfunction
        :return: GPUArray with Boltzmann-Gibbs state
        """
        # Set the initial state and propagate
        self.set_wignerfunction(1. / np.prod(self.wignerfunction.shape))
        return self.propagate(self.num_beta_steps)

    expK_cuda_source = """
    /////////////////////////////////////////////////////////////////////////////
    //
    // Overloading WignerMoyalCUDA1D.expK_cuda_source
    //
    /////////////////////////////////////////////////////////////////////////////

    // CUDA code to define the action of the kinetic energy exponent
    // onto the wigner function in P Lambda representation

    #include<pycuda-complex.hpp>
    #include<math.h>
    #define _USE_MATH_DEFINES

    typedef pycuda::complex<double> cuda_complex;

    {cuda_consts}

    // Kinetic energy
    __device__ double K(double P, double t)
    {{
        return ({K});
    }}

    __global__ void Kernel(cuda_complex *Z, double t)
    {{
        const size_t i = blockIdx.y;
        const size_t j = threadIdx.x + blockDim.x*blockIdx.x;
        const size_t indexTotal = threadIdx.x +
            blockDim.x * blockIdx.x  + blockIdx.y * (blockDim.x + 1) * gridDim.x;

        const double Lambda = dLambda * j;
        const double P = dP * (i - 0.5 * P_gridDIM);

        const double phase = -dbeta * (
            K(P + 0.5 * Lambda, t_initial) + K(P - 0.5 * Lambda, t_initial) - K_min
        );

        Z[indexTotal] *= exp(phase);
    }}
    """

    fill_extended_K = """
    /////////////////////////////////////////////////////////////////////////////
    //
    // set Z = K(P + 0.5 * Lambda, t_initial) + K(P - 0.5 * Lambda, t_initial)
    //
    /////////////////////////////////////////////////////////////////////////////

    #include<math.h>
    #define _USE_MATH_DEFINES

    {cuda_consts}

    // Kinetic energy
    __device__ double K(double P, double t)
    {{
        return ({K});
    }}

    __global__ void Kernel(double *Z)
    {{
        const size_t i = blockIdx.y;
        const size_t j = threadIdx.x + blockDim.x*blockIdx.x;
        const size_t indexTotal = threadIdx.x +
            blockDim.x * blockIdx.x  + blockIdx.y * (blockDim.x + 1) * gridDim.x;

        const double Lambda = dLambda * j;
        const double P = dP * (i - 0.5 * P_gridDIM);

        Z[indexTotal] = K(P + 0.5 * Lambda, t_initial) + K(P - 0.5 * Lambda, t_initial);
    }}
    """

    expV_cuda_source = """
    /////////////////////////////////////////////////////////////////////////////
    //
    // Overloading WignerMoyalCUDA1D.expV_cuda_source
    //
    /////////////////////////////////////////////////////////////////////////////

    // CUDA code to define the action of the potential energy exponent
    // onto the wigner function in Theta X representation

    #include<pycuda-complex.hpp>
    #include<math.h>
    #define _USE_MATH_DEFINES

    typedef pycuda::complex<double> cuda_complex;

    {cuda_consts}

    // Absorbing boundary condition
    __device__ double abs_boundary(double X)
    {{
        return ({abs_boundary});
    }}

    // Potential energy
    __device__ double V(double X, double t)
    {{
        return ({V});
    }}

    __global__ void Kernel(cuda_complex *Z, double t)
    {{
        const size_t i = blockIdx.y;
        const size_t j = threadIdx.x + blockDim.x*blockIdx.x;
        const size_t indexTotal = threadIdx.x + blockDim.x * blockIdx.x  + blockIdx.y * blockDim.x * gridDim.x;

        const double X = dX * (j - 0.5 * X_gridDIM);
        const double Theta = dTheta * i;

        const double X_minus = X - 0.5 * Theta;
        const double X_plus = X + 0.5 * Theta;

        const double phase = -0.5 * dbeta * (V(X_minus, t_initial) + V(X_plus, t_initial) - V_min);

        Z[indexTotal] *= exp(phase) * abs_boundary(X_minus) * abs_boundary(X_plus);
    }}
    """

    fill_extended_V = """
    /////////////////////////////////////////////////////////////////////////////
    //
    // set Z = V( X - 0.5 * Theta, t_initial) + V( X + 0.5 * Theta, t_initial)
    //
    /////////////////////////////////////////////////////////////////////////////

    #include<math.h>
    #define _USE_MATH_DEFINES

    {cuda_consts}

    // Potential energy
    __device__ double V(double X, double t)
    {{
        return ({V});
    }}

    __global__ void Kernel(double *Z)
    {{
        const size_t i = blockIdx.y;
        const size_t j = threadIdx.x + blockDim.x*blockIdx.x;
        const size_t indexTotal = threadIdx.x + blockDim.x * blockIdx.x  + blockIdx.y * blockDim.x * gridDim.x;

        const double X = dX * (j - 0.5 * X_gridDIM);
        const double Theta = dTheta * i;

        const double X_minus = X - 0.5 * Theta;
        const double X_plus = X + 0.5 * Theta;

        Z[indexTotal] = V(X_minus, t_initial) + V(X_plus, t_initial);
    }}
    """

##########################################################################################
#
# Example
#
##########################################################################################

if __name__ == '__main__':

    print(WignerBlochCUDA1D.__doc__)

    import matplotlib.pyplot as plt

    # parameters for the quantum system
    params = dict(
        t=0.,
        dt=0.01,
        X_gridDIM=1024,
        X_amplitude=10.,
        P_gridDIM=512,
        P_amplitude=10.,

        # Temperature of the initial state
        kT=np.random.uniform(0.1, 1.),

        # randomized parameter
        omega=np.random.uniform(1., 2.),

        # kinetic energy part of the hamiltonian
        K="0.5 * P * P",

        # potential energy part of the hamiltonian
        V="0.5 * omega * omega * X * X",

        # Hamiltonian
        H=lambda self, x, p: 0.5*(p**2 + self.omega**2 * x**2),

        # Exact analytical expression for the harmonic oscillator Gibbs state
        get_exact_gibbs=lambda self: np.tanh(0.5 * self.omega / self.kT) / np.pi * np.exp(
            -2. * np.tanh(0.5 * self.omega / self.kT) * self.H(self.X, self.P) / self.omega
        )
    )

    print("Calculating the Gibbs state...")
    gibbs_state = WignerBlochCUDA1D(**params).get_gibbs_state()

    print("Check that the obtained Gibbs state is stationary under the Wigner-Moyal propagation...")
    propagator = WignerMoyalCUDA1D(**params)
    final_state = propagator.set_wignerfunction(gibbs_state).propagate(3000).get()

    gibbs_state = gibbs_state.get()

    exact_gibbs = propagator.get_exact_gibbs()
    print(
        "\nIninity norm between analytical and numerical Gibbs states = %.2e ." %
        (np.linalg.norm(exact_gibbs.reshape(-1) - gibbs_state.reshape(-1), np.inf) * propagator.dX * propagator.dP)
    )

    ##########################################################################################
    #
    #   Plot the results
    #
    ##########################################################################################

    from wigner_normalize import WignerSymLogNorm

    # save common plotting parameters
    plot_params = dict(
        origin='lower',
        extent=[propagator.X.min(), propagator.X.max(), propagator.P.min(), propagator.P.max()],
        cmap='seismic',
        # make a logarithmic color plot (see, e.g., http://matplotlib.org/users/colormapnorms.html)
        norm=WignerSymLogNorm(linthresh=1e-14, vmin=-0.01, vmax=0.1)
    )
    plt.subplot(131)

    plt.title("The Gibbs state (initial state)")
    plt.imshow(gibbs_state, **plot_params)
    plt.colorbar()
    plt.xlabel('$x$ (a.u.)')
    plt.ylabel('$p$ (a.u.)')

    plt.subplot(132)

    plt.title("The exact Gibbs state")
    plt.imshow(exact_gibbs, **plot_params)
    plt.colorbar()
    plt.xlabel('$x$ (a.u.)')
    plt.ylabel('$p$ (a.u.)')

    plt.subplot(133)

    plt.title("The Gibbs state after propagation")
    plt.imshow(final_state, **plot_params)
    plt.colorbar()
    plt.xlabel('$x$ (a.u.)')
    plt.ylabel('$p$ (a.u.)')

    plt.show()
