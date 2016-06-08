from wigner_moyal_cuda_1d import WignerMoyalCUDA1D

import pycuda.gpuarray as gpuarray
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

        try:
            self.dbeta = kwargs['dbeta']
            # remove dbeta from kwargs so that it does not enter into self.cuda_consts
            del kwargs['dbeta']
        except KeyError:
            # if dbeta is not defined, just choose some value
            self.dbeta = 0.01

        if 'dt' not in kwargs:
            # Save the inverse temperature increment as dt
            kwargs.update(dt=self.dbeta)

        if self.kT > 0:
            # get number of dbeta steps to reach the desired Gibbs state
            self.num_beta_steps = 1. / (self.kT*self.dbeta)

            if round(self.num_beta_steps) <> self.num_beta_steps:
                # Changing self.dbeta so that num_beta_steps is an exact integer
                self.num_beta_steps = round(self.num_beta_steps)
                self.dbeta = 1. / (self.kT*self.num_beta_steps)

            self.num_beta_steps = int(self.num_beta_steps)

        self.dbeta = np.float64(self.dbeta)

        # Initialize parent class
        WignerMoyalCUDA1D.__init__(self, **kwargs)

        # Save the minimums of the potential (V) and kinetic (K) energy
        self.cuda_consts += "    const double V_min = %.15e;\n" % self.get_V_min()
        self.cuda_consts += "    const double K_min = %.15e;\n" % self.get_K_min()

        print("\n================================ Compiling Bloch expK and expV ================================\n")

        bloch_expK_expV_compiled = SourceModule(
            self.bloch_expK_expV_cuda_source.format(
                cuda_consts=self.cuda_consts, K=self.K, V=self.V,
                abs_boundary_lambda_p=self.abs_boundary_lambda_p, abs_boundary_x_theta=self.abs_boundary_x_theta
            )
        )

        self.bloch_expK_bulk = bloch_expK_expV_compiled.get_function("bloch_expK_bulk")
        self.bloch_expK_boundary = bloch_expK_expV_compiled.get_function("bloch_expK_boundary")

        self.bloch_expV_bulk = bloch_expK_expV_compiled.get_function("bloch_expV_bulk")
        self.bloch_expV_boundary = bloch_expK_expV_compiled.get_function("bloch_expV_boundary")

    def get_V_min(self):
        """
        Return the potential energy minimum in the x theta space
        """
        # allocate memory
        v_theta_x = gpuarray.zeros((self.Theta.size, self.X.size), np.float64)

        # fill array with values of the potential energy
        fill_compiled = SourceModule(
            self.fill_V_K.format(cuda_consts=self.cuda_consts, K=self.K, V=self.V)
        )
        fill_compiled.get_function("fill_V_bulk")(v_theta_x, **self.V_bulk_mapper_params)
        fill_compiled.get_function("fill_V_boundary")(v_theta_x, **self.V_boundary_mapper_params)

        return gpuarray.min(v_theta_x).get()

    def get_K_min(self):
        """
        Return the kinetic energy minimum in the lambda p space
        """
        # allocate memory
        k_p_lambda = gpuarray.zeros((self.P.size, self.Lambda.size), np.float64)

        # fill array with values of the kinetic energy
        fill_compiled = SourceModule(
            self.fill_V_K.format(cuda_consts=self.cuda_consts, K=self.K, V=self.V)
        )
        fill_compiled.get_function("fill_K_bulk")(k_p_lambda, **self.K_bulk_mapper_params)
        fill_compiled.get_function("fill_K_boundary")(k_p_lambda, **self.K_boundary_mapper_params)

        return gpuarray.min(k_p_lambda).get()

    def get_gibbs_state(self):
        """
        Calculate the Boltzmann-Gibbs state and save it in self.wignerfunction
        :return: GPUArray with Boltzmann-Gibbs state
        """
        # Set the initial state and propagate
        self.set_wignerfunction(1. / np.prod(self.wignerfunction.shape))

        for _ in xrange(self.num_beta_steps):
            # advance by one time step
            self.bloch_single_step_propagation(self.dbeta)

            # normalization
            #self.wignerfunction /= gpuarray.sum(self.wignerfunction).get() * self.dXdP

        print("Purity %.6f; Uncertanty %.3f" % (self.get_purity(), self.get_sigma_x_sigma_p()))

        return self.wignerfunction

    def get_ground_state(self, dbeta=0.5):

        # Set the initial state and propagate
        self.set_wignerfunction(1. / np.prod(self.wignerfunction.shape))

        # Initialize varaibles
        previous_energy = current_energy = np.inf
        dbeta = np.float64(dbeta)
        previous_purity = current_purity = 0.

        # Allocate memory for extra copy of the Wigner function
        previous_wigner_function = self.wignerfunction.copy()

        #while current_purity < (1. - 1e-7):
        while True:
            # advance by one time step
            self.bloch_single_step_propagation(dbeta)

            try:
                # Check whether the state cooled
                current_energy = self.get_average(self.hamiltonian)
                assert current_energy < previous_energy

                # Purity cannot be larger than one
                #current_purity = self.get_purity()
                #assert current_purity <= 1.

                # Verify the uncertainty principle
                assert self.get_sigma_x_sigma_p() >= 0.5

                # the current state seems to be physical, so we accept it
                previous_energy = current_energy
                previous_purity = current_purity

                print current_purity, current_energy, dbeta

                # make a copy of the current state
                gpuarray._memcpy_discontig(previous_wigner_function, self.wignerfunction)

            except AssertionError:
                # the current state is unphysical,
                # revert the propagation
                gpuarray._memcpy_discontig(self.wignerfunction, previous_wigner_function)

                # and half the step size
                dbeta *= 0.5

                #print dbeta

                # restore the original settings
                current_energy = previous_energy
                current_purity = previous_purity

        return self.wignerfunction

    def bloch_single_step_propagation(self, dbeta):
        """
        Perform a single step propagation with respect to the inverse temperature via the Bloch equation.
        The final Wigner function is not normalized.
        :param dbeta: (float) the inverse temperature step size
        :return: self.wignerfunction
        """
        self.p2theta_transform()
        self.bloch_expV_bulk(self.wigner_theta_x, dbeta, **self.V_bulk_mapper_params)
        self.bloch_expV_boundary(self.wigner_theta_x, dbeta, **self.V_boundary_mapper_params)
        self.theta2p_transform()

        self.x2lambda_transform()
        self.bloch_expK_bulk(self.wigner_p_lambda, dbeta, **self.K_bulk_mapper_params)
        self.bloch_expK_boundary(self.wigner_p_lambda, dbeta, **self.K_boundary_mapper_params)
        self.lambda2x_transform()

        self.p2theta_transform()
        self.bloch_expV_bulk(self.wigner_theta_x, dbeta, **self.V_bulk_mapper_params)
        self.bloch_expV_boundary(self.wigner_theta_x, dbeta, **self.V_boundary_mapper_params)
        self.theta2p_transform()

        # normalize
        self.wignerfunction /= gpuarray.sum(self.wignerfunction).get() * self.dXdP

        return self.wignerfunction

    bloch_expK_expV_cuda_source = """
    /////////////////////////////////////////////////////////////////////////////
    //
    // This code closely follows WignerMoyalCUDA1D.expK_cuda_source
    //
    /////////////////////////////////////////////////////////////////////////////

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

    // Potential energy
    __device__ double V(double X, double t)
    {{
        return ({V});
    }}

    ////////////////////////////////////////////////////////////////////////////
    //
    // CUDA code to define the action of the kinetic energy exponent
    // onto the wigner function in P Lambda representation
    //
    ////////////////////////////////////////////////////////////////////////////

    __global__ void bloch_expK_bulk(cuda_complex *Z, double dbeta)
    {{
        const size_t i = blockIdx.y;
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t indexTotal = j  + i * (X_gridDIM / 2 + 1);

        const double Lambda = dLambda * j;
        const double P = dP * (i - 0.5 * P_gridDIM);

        const double phase = -0.5 * dbeta * (
            K(P + 0.5 * Lambda, t_initial) + K(P - 0.5 * Lambda, t_initial) - K_min
        );

        Z[indexTotal] *= exp(phase) * ({abs_boundary_lambda_p});
    }}

    __global__ void bloch_expK_boundary(cuda_complex *Z, double dbeta)
    {{
        const size_t i = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t j = X_gridDIM / 2;
        const size_t indexTotal = j  + i * (X_gridDIM / 2 + 1);

        const double Lambda = dLambda * j;
        const double P = dP * (i - 0.5 * P_gridDIM);

        const double phase = -0.5 * dbeta * (
            K(P + 0.5 * Lambda, t_initial) + K(P - 0.5 * Lambda, t_initial) - K_min
        );

        Z[indexTotal] *= exp(phase) * ({abs_boundary_lambda_p});
    }}

    ////////////////////////////////////////////////////////////////////////////
    //
    // CUDA code to define the action of the potential energy exponent
    // onto the wigner function in Theta X representation
    //
    ////////////////////////////////////////////////////////////////////////////

    __global__ void bloch_expV_bulk(cuda_complex *Z, double dbeta)
    {{
        const size_t i = blockIdx.y;
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t indexTotal = j + i * X_gridDIM;

        const double X = dX * (j - 0.5 * X_gridDIM);
        const double Theta = dTheta * i;

        const double phase = -0.25 * dbeta * (
            V(X - 0.5 * Theta, t_initial) + V(X + 0.5 * Theta, t_initial) - V_min
        );

        Z[indexTotal] *= exp(phase) * ({abs_boundary_x_theta});
    }}

    __global__ void bloch_expV_boundary(cuda_complex *Z, double dbeta)
    {{
        const size_t i = P_gridDIM / 2;
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t indexTotal = j + i * X_gridDIM;

        const double X = dX * (j - 0.5 * X_gridDIM);
        const double Theta = dTheta * i;

        const double phase = -0.25 * dbeta * (
            V(X - 0.5 * Theta, t_initial) + V(X + 0.5 * Theta, t_initial) - V_min
        );

        Z[indexTotal] *= exp(phase) * ({abs_boundary_x_theta});
    }}
    """

    fill_V_K = """
    #include<math.h>
    #define _USE_MATH_DEFINES

    {cuda_consts}

    // Potential energy
    __device__ double V(double X, double t)
    {{
        return ({V});
    }}

    // Kinetic energy
    __device__ double K(double P, double t)
    {{
        return ({K});
    }}

    /////////////////////////////////////////////////////////////////////////////
    //
    // set Z = V( X - 0.5 * Theta, t_initial) + V( X + 0.5 * Theta, t_initial)
    //
    /////////////////////////////////////////////////////////////////////////////

    __global__ void fill_V_bulk(double *Z)
    {{
        const size_t i = blockIdx.y;
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t indexTotal = j + i * X_gridDIM;

        const double X = dX * (j - 0.5 * X_gridDIM);
        const double Theta = dTheta * i;

        const double X_minus = X - 0.5 * Theta;
        const double X_plus = X + 0.5 * Theta;

        Z[indexTotal] = V(X_minus, t_initial) + V(X_plus, t_initial);
    }}

    __global__ void fill_V_boundary(double *Z)
    {{
        const size_t i = P_gridDIM / 2;
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t indexTotal = j + i * X_gridDIM;

        const double X = dX * (j - 0.5 * X_gridDIM);
        const double Theta = dTheta * i;

        const double X_minus = X - 0.5 * Theta;
        const double X_plus = X + 0.5 * Theta;

        Z[indexTotal] = V(X_minus, t_initial) + V(X_plus, t_initial);
    }}

    /////////////////////////////////////////////////////////////////////////////
    //
    // set Z = K(P + 0.5 * Lambda, t_initial) + K(P - 0.5 * Lambda, t_initial)
    //
    /////////////////////////////////////////////////////////////////////////////

    __global__ void fill_K_bulk(double *Z)
    {{
        const size_t i = blockIdx.y;
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t indexTotal = j  + i * (X_gridDIM / 2 + 1);

        const double Lambda = dLambda * j;
        const double P = dP * (i - 0.5 * P_gridDIM);

        Z[indexTotal] = K(P + 0.5 * Lambda, t_initial) + K(P - 0.5 * Lambda, t_initial);
    }}

    __global__ void fill_K_boundary(double *Z)
    {{
        const size_t i = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t j = X_gridDIM / 2;
        const size_t indexTotal = j  + i * (X_gridDIM / 2 + 1);

        const double Lambda = dLambda * j;
        const double P = dP * (i - 0.5 * P_gridDIM);

        Z[indexTotal] = K(P + 0.5 * Lambda, t_initial) + K(P - 0.5 * Lambda, t_initial);
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
