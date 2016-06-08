import numpy as np
from scipy import fftpack # Tools for fourier transform
from scipy import linalg # Linear algebra for dense matrix
from band_structure_1d import MUBQBandStructure


class MUBGibbsBand(MUBQBandStructure):
    """
    Calculate the Gibbs satte for band structure for quantum Hamiltonian, H(x,p) = K(p) + V(x),
    using mutually unbiased bases (MUB).
    """
    def get_gibbs(self, K, n, kT):
        """
        Return the Gibbs state rho = exp(-H/kT) by diagonalizing the Hamiltonian
        :param n: number of bands to include
        :param kT: temperature
        :return:
        """
        rho_gibbs = np.zeros((self.X_gridDIM, self.X_gridDIM), dtype=np.complex)

        # Save the band structure
        self.band_structure = []

        for k_val in K:
            vals, vecs = linalg.eigh(self.get_hamiltonian(k_val))

            # extract only first n eigenvectors
            vecs = vecs[:,:n]
            vals = vals[:n]

            # Save eigenvalues
            self.band_structure.append(vals)

            # rho[a,b] = sum over c of vecs[a,c] * conj(vecs[b,c]) * exp(-E[c]/kT)
            rho_gibbs += np.einsum('ac,bc,c', vecs, vecs.conj(), np.exp(-vals / kT))

        # normalize the obtained gibbs state
        rho_gibbs /= rho_gibbs.trace()
        return rho_gibbs

    def get_gibbs_bloch(self, kT, dbeta=0.01):
        """
        Get Gibbs state by solving the Bloch equation
        :param kT: (float) temperature
        :return:
        """
        # get number of dbeta steps to reach the desired Gibbs state
        num_beta_steps = 1. / (kT * dbeta)

        if round(num_beta_steps) <> num_beta_steps:
            # Changing dbeta so that num_beta_steps is an exact integer
            num_beta_steps = round(num_beta_steps)
            dbeta = 1. / (kT * num_beta_steps)

        num_beta_steps = int(num_beta_steps)

        # precalcuate the exponenets
        expV = np.exp(-0.25 * dbeta * (
            self.V(self.X_range[:, np.newaxis]) + self.V(self.X_range[np.newaxis, :])
        ))
        expK = np.exp(-0.5 * dbeta * (
            self.K(self.P_range[:, np.newaxis]) + self.K(self.P_range[np.newaxis, :])
        ))

        # initiate the Gibbs state
        rho_gibb = np.eye(self.X_gridDIM, dtype=np.complex)

        # propagate the state in beta
        for _ in xrange(num_beta_steps):
            rho_gibb *= expV
            rho_gibb = fftpack.fft2(rho_gibb, overwrite_x=True)
            rho_gibb *= expK
            rho_gibb = fftpack.ifft2(rho_gibb, overwrite_x=True)
            rho_gibb *= expV
            rho_gibb /= rho_gibb.trace()

        return rho_gibb

    def get_band_structure(self):
        return np.array(self.band_structure).T

    def get_energy(self, rho):
        """
        Calculate the expectation value of the hamiltonian over the denisty matrix rho
        :param rho: (numpy.array) denisty matrix
        :return: float
        """
        # normalize the state
        rho /= rho.trace()

        # expectation value of the potential energy
        average_V = np.dot(rho.diagonal().real, self.V(self.X_range))

        # going into the momentum representation
        rho_p = fftpack.fft2(rho)
        rho_p /= rho_p.trace()

        # expectation value of the kinetic energy
        average_K = np.dot(rho_p.diagonal().real, self.K(self.P_range))

        return average_K + average_V

##########################################################################################
#
# Example
#
##########################################################################################

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    print(MUBGibbsBand.__doc__)

    sys_params = dict(
        X_gridDIM=256,

        # the lattice constant is 2 * X_amplitude
        X_amplitude=4.,

        # Lattice height
        V0=0.37,

        # the kinetic energy
        K=lambda self, p: 0.5 * p ** 2,

        # Mathieu-type periodic system
        V=lambda self, x: -self.V0 * (1 + np.cos(np.pi * (x + self.X_amplitude) / self.X_amplitude))
    )

    def is_physicial(rho):
        """
        check for physicallity of the density matrix rho
        """
        p = linalg.eigvalsh(rho)
        if not np.allclose(p[p < 0], 0) and not np.allclose(rho.diagonal().imag, 0):
            print("WARNING: Obtained Gibbs denisty matrix i not a positively defined matrix")

    # initialize the system
    qsys = MUBGibbsBand(**sys_params)

    # range of bloch vectors to compute the band structure
    k_ampl = np.pi / qsys.X_amplitude
    K = np.linspace(-0.5 * k_ampl, 0.5 * k_ampl, 200)

    plt.subplot(121)

    gibbs = qsys.get_gibbs(K, 5, 0.05)
    is_physicial(gibbs)
    print("Energy of the Gibbs state via MUB: %f (a.u.)" % qsys.get_energy(gibbs))

    gibbs_bloch = qsys.get_gibbs_bloch(0.05)
    is_physicial(gibbs_bloch)
    print("Energy of the Gibbs state via Bloch: %f (a.u.)" % qsys.get_energy(gibbs_bloch))

    #plt.title("Gibbs state")
    #plt.imshow(np.real(gibbs), origin='lower')
    #plt.colorbar()
    plt.plot(qsys.X_range, gibbs.diagonal().real,label='Gibbs')
    plt.plot(qsys.X_range, gibbs_bloch.diagonal().real, label='Gibbs via Bloch')
    plt.legend()

    #plt.subplot(132)
    #plt.title("Gibbs bloch")
    #plt.imshow(np.real(gibbs_bloch), origin='lower')
    #plt.colorbar()

    plt.subplot(122)
    for epsilon in qsys.get_band_structure():
        plt.plot(K, epsilon)

    print("Mininum energy: %f (a.u.)" % qsys.get_band_structure().min())

    plt.title("Reproduction of Fig. 1 from M. Wu et al. Phys. Rev A 91, 043839 (2015)")
    plt.xlabel("$k$ (a.u.)")
    plt.ylabel('$\\varepsilon(k)$ (a.u.)')

    plt.show()
