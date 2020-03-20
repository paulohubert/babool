import numpy as np
import scipy as sp
from scipy.stats import geom, poisson, beta
from joblib import delayed, Parallel

class BoolFunction:
    '''
    Boolean Function learning - uses a simulated annealing procedure to learn boolean functions from data.
    '''
    
    def __init__(self, ap0 = 1, bp0 = 1, ap1 = 1, bp1 = 1, theta = 3, pgeomk = None, pgeomm = 0.5):
        '''
        ap0 - parameter for Beta prior for p0
        bp0 - parameter for Beta prior for p0
        ap1 - parameter for Beta prior for p1
        bp1 - parameter for Beta prior for p1
        theta - parameter for Poisson prior for the size of the marker k
        pgeomk - parameter for Geometric prior for the size of the marker k. If None, adopts Poisson prior
        pgeomm - parameter for Geometric prior for the number of markers m
        '''
        self.ap0 = ap0
        self.bp0 = bp0
        self.ap1 = ap1
        self.bp1 = bp1
        self.theta = theta
        self.pgeomk = pgeomk
        self.pgeomm = pgeomm
        
        self.X = None
        self.y = None
        self.N = None       
        self.psi = None
    
    def __apply_f(self, X, f):
        '''
        Applies f to the current data matrix, returns number of affected individuals and number of positive and affected individuals
        '''
        
        fX = np.zeros(shape = self.N)
        for phi in f:
            fX = fX + np.prod(X[:,phi], axis = 1)
            fX[fX > 1] = 1

        # Number of positive individuals
        npos = np.sum(fX)

        # Number of positive affected individuals
        nposaff = np.sum(fX * self.y)
        
        return npos, nposaff
        
    def prior_p(self, p0,p1):
        '''
        Joint log-prior for pi0 and pi1. Uses two independent betas
        '''
        return beta.logpdf(x = p0, a = self.ap0, b = self.bp0) + sp.stats.beta.logpdf(x = p1, a = self.ap1, b = self.bp1)

    def prior_m(self, m):
        '''
        Log-prior for m, the number of conjunction terms
        '''
        if m <= 0:
            return -np.inf
        elif self.pgeomm is None:
            return 0
        else:
            return (m-1)*np.log(1-self.pgeomm) + np.log(self.pgeomm)
    
    def prior_k(self, k):
        '''
        Log-prior for k, the number of atomic terms in the marker.
        Poisson 
        '''
        if k <= 0:
            return -np.inf
        else:
            if self.pgeomk is None:
                if self.theta is not None:
                    # Poisson for k - 1, since P(k = 0) = 0
                    return poisson.logpmf(k - 1, self.theta)
                else:
                    return 0
            else:
                return geom.logpmf(k-1, self.pgeomk)

    def prior_f(self, f):
        '''
        Log-prior for the function f
        f is a list of lists, each element is a conjunction term
        '''
        m = len(f)
        lp = self.prior_m(m)
        for s in f:
            lp += self.prior_k(len(s))

        return lp

    def loglike(self, p0, p1, f):
        '''
        Log-likelihood
        '''

        if self.X is None or self.y is None:
            raise ValueError("No data provided!")
        
        npos, nposaff = self.__apply_f(self.X, f)
        
        if npos == 0 or npos == self.N:
            # Does not accept markers such that none or all individuals are marked
            return -np.inf

        # To avoid division by 0
        p0 = max(p0, 1e-10)
        p1 = min(p1, 1-1e-10)

        # Calculating odds
        o1 = p1 / (1-p1)
        o0 = p0 / (1-p0)
        o10 = (1-p1)/(1-p0)

        return(self.N*np.log(1-p0) + nposaff*np.log(o1/o0) + self.Naff*np.log(o0) + npos*np.log(o10))   

    def post(self, p0, p1, f):
        '''
        Posterior kernel
        '''
        
        if self.X is None or self.y is None:
            raise ValueError("No data provided!")
            

        return self.loglike(p0,p1,f) + self.prior_f(f) + self.prior_p(p0, p1)
        
    
    def __optim_chain(self, phicurr, nsteps = 1000):
        '''
        Generates samples from the MCMC for the single marker boolean function.
        
        @args
        phicurr - starting point, in the form of a list of numpy boolean vectors
        nsteps - total number of steps to run
        
        @return
        opt - maximum point for the model's posterior
        '''
        
        if self.X is None or self.y is None:
            raise ValueError("No data provided!")
            
        nattempt = [0,0,0,0,0,0,0]
        naccept = [0,0,0,0,0,0,0]

        # Number of times each individual is selected
        nsel = np.zeros(len(self.iaff))

        # Number of markers
        mcurr = len(phicurr)

        # Cooling factor
        coolfactor = 0.9

        # Initial modification factor for acceptance probability
        cool = 1000

        # Converting phi to on-set representation
        oncurr = list(np.where(phicurr[0] == 1)[0])
        oncurr.sort()

        # Keeps the list of markers
        markers = [oncurr]
        # Sizes of each marker
        sizes = [len(oncurr)]

        # Fills in the list of markers
        for i in range(1, len(phicurr)):
            oncurr = list(np.where(phicurr[i] == 1)[0])
            oncurr.sort()
            markers.append(oncurr)
            sizes.append(len(oncurr))

        params = []    
        
        # Starting values for p0 and p1
        p0curr = np.random.uniform()
        p1curr = np.random.uniform()

        # Starting value for posterior
        pcurr = self.post(p0curr, p1curr, markers)

        # Move probabilities
        moveprobs = [.05,.05,.4,.4,.05,.04,.01]
        
        for i in range(nsteps):
            
            npos, nposaff = self.__apply_f(self.X, markers)            

            movetype = int(np.random.choice(range(7), size = 1, p = moveprobs))
            nattempt[movetype] += 1

            if movetype == 0:

                ### Move 0: sample p0
                a0pos = self.ap0 + (self.Naff - nposaff)
                b0pos = self.bp0 + (self.N - npos) - (self.Naff - nposaff)

                p0curr = a0pos / (a0pos + b0pos)
                
                pcurr = self.post(p0curr, p1curr, markers)

            elif movetype == 1:

                ### Move 1: sample p1
                a1pos = self.ap1 + nposaff
                b1pos = self.bp1 + npos - nposaff

                p1curr = a1pos / (a1pos + b1pos)

                pcurr = self.post(p0curr, p1curr, markers)

            elif movetype == 2:
                ### Move 2: flip one bit off

                # Select marker
                mcand = markers.copy()
                # Probability proportional to marker size
                probs = [s / sum(sizes) for s in sizes]
                im = np.random.choice(len(markers), p = probs)
                mflip = mcand[im].copy()
                rm = np.random.choice(mflip, size = 1)[0]
                mcand.remove(mflip)
                mflip.remove(rm)
                mflip.sort()
                mcand.append(mflip)

                pcand = self.post(p0curr, p1curr, mcand)

                if not np.isinf(pcand):
                    alpha = min(0, (pcand - pcurr))
                    if np.log(np.random.uniform()) < alpha:
                        pcurr = pcand
                        markers = mcand
                        sizes = [len(f) for f in markers]
                        naccept[movetype] += 1                         


            elif movetype == 3:
                ### Move 3: flip one bit on

                # Select marker with probability inversely proportional to size
                probs = [1/s for s in sizes]
                probs = [p/sum(probs) for p in probs]
                mcand = markers.copy()
                im = np.random.choice(len(markers), p = probs)
                mflip = mcand[im].copy()
                sset = [i for i in range(self.p) if i not in mflip]
                ap = np.random.choice(sset, size = 1)[0]
                mcand.remove(mflip)
                mflip.append(ap)
                mflip.sort()
                mcand.append(mflip)
                pcand = self.post(p0curr, p1curr, mcand)

                if not np.isinf(pcand):
                    alpha = min(0, (pcand - pcurr))
                    if np.log(np.random.uniform()) < alpha:
                        pcurr = pcand
                        markers = mcand
                        sizes = [len(f) for f in markers]
                        naccept[movetype] += 1   


            elif movetype == 4:
                ### Move 4: add new affected individual

                # To select it: keep vector with number of times each positive individual have been 
                # previously selected. Draws with prob inverse to this number.

                # Keeps a probability of not selecting anyone (skipping move)
                Z = sum([ 1/ (n+1) for n in nsel])
                probs = [(1/(n+1))/Z for n in nsel]

                # Selects the index of a positive individual
                u = np.random.choice(range(len(nsel)), size = 1, p = probs)[0]
                isel = self.iaff[u]

                # Updates number of selections
                nsel[u] += 1

                # Obtains its onset and adds it to the candidate point
                mcand = markers.copy()
                mcand.append(list(np.where(self.X[isel,:] == 1)[0]))

                # Posterior
                pcand = self.post(p0curr, p1curr, mcand)

                # Acceptance probability
                alpha = min(0,(cool) + (pcand - pcurr))

                if not np.isinf(alpha):
                    if np.log(np.random.uniform()) < alpha:
                        pcurr = pcand
                        markers = mcand
                        mcurr += 1
                        sizes = [len(f) for f in markers]
                        naccept[movetype] += 1

                cool = cool * coolfactor

            elif movetype == 5:

                ### Move 5: replace pair of markers by their intersection

                # Only if m > 1
                if mcurr > 1:
                    # Selects two of the current markers with probability proportional to the sizes
                    sel = np.random.choice(range(mcurr), size = 2, replace = False)
                    m1 = markers[sel[0]].copy()
                    m2 = markers[sel[1]].copy()
                    md = list(set(m1).intersection(set(m2)))

                    if len(md) > 0:
                        mcand = markers.copy()
                        mcand.remove(m1)
                        mcand.remove(m2)
                        md.sort()
                        mcand.append(md)

                        # Posterior
                        pcand = self.post(p0curr, p1curr, mcand)                    

                        # Acceptance probability
                        alpha = min(0, (pcand - pcurr))

                        if not np.isinf(alpha):
                            if np.log(np.random.uniform()) < alpha:
                                mcurr += -1
                                pcurr = pcand
                                markers = mcand
                                sizes = [len(f) for f in markers]
                                naccept[movetype] += 1

                    cool = cool * coolfactor

            elif movetype == 6:

                ### Move 6: remove random marker
                

                # Only if m > 1
                if mcurr > 1:

                    # Selects one marker with prob proportional to size
                    sel = np.random.choice(range(mcurr), size = 1, replace = False)
                    md = markers[sel[0]].copy()

                    mcand = markers.copy()
                    mcand.remove(md)

                    # Posterior
                    pcand = self.post(p0curr, p1curr, mcand)                    

                    # Acceptance probability
                    alpha = min(0, (pcand - pcurr))

                    if not np.isinf(alpha):
                        if np.log(np.random.uniform()) < alpha:
                            mcurr += -1
                            pcurr = pcand
                            markers = mcand
                            sizes = [len(f) for f in markers]
                            naccept[movetype] += 1

                cool = cool * coolfactor  

            params.append([markers, p0curr, p1curr])

        return [params[-1], self.post(p0curr, p1curr, markers)]
    
    def fit(self, X, y, nchains = 4, njobs = -1, nsteps = 1000, nstart = 1):
        '''
        Runs nchains parallel algorithms, with parameter njobs for the Parallel library.
        
        @args
        X - numpy array with explanatory variables
        y - numpy array with response variable
        nchains - number of parallel runs
        njobs - number of parallel jobs
        nsteps - number of steps to run the stochastic search
        nstart - number of conjunction terms in each initial point
        
        @returns
        params - list with results for each parallel run
        
        '''
    
        self.X = X
        self.y = y
        self.N = len(self.y)
        self.Naff = sum(self.y)
        self.iaff = list(np.where(self.y==1)[0])
        self.p = self.X.shape[1]
        
        startlist = []
        
        for i in range(nchains):
            startlist.append(list(self.X[np.random.choice(range(self.N), nstart, replace = False), :]))
        
        markers = Parallel(n_jobs=njobs)(delayed(self.__optim_chain)(start, nsteps) for start in startlist)
        
        # Optimal point
        pmax = -np.inf
        marker_max = None
        for m in markers:
            if m[1] > pmax:
                pmax = m[1]
                marker_max = m[0][0]
                p0_max = m[0][1]
                p1_max = m[0][2]
                
        self.pmax = pmax
        self.psi = marker_max
        self.p0 = p0_max
        self.p1 = p1_max
        
        return markers
    
    def predict(self, X, binary = False):
        '''
        Function to generate predictions based on current model
        
        @args
        X - binary explanatory matrix
        binary - if true, returns binary predictions; otherwise returns probability values
        '''
        
        if self.psi is None:
            raise ValueError("Fit model first.")
        # Apply f to the new data
        fX = np.zeros(shape = X.shape[0])
        for phi in self.psi:
            fX = fX + np.prod(X[:,phi], axis = 1)

        fX[fX > 1] = 1
        
        if not binary:
            fX[fX == 1] = self.p1
            fX[fX == 0] = self.p0
        
        return(fX)
        
        
        
        
        
        
        
        
        
        

class MultiMarker:
    '''
    MultiMarker class - uses MCMC to obtain samples for a single marker (implicant) boolean function for the binary classification problem with binary explanatory variables.
    '''
    def __init__(self, ap0 = 1, bp0 = 1, ap1 = 1, bp1 = 1, theta = 3, pgeomk = None):
        '''
        ap0 - parameter for Beta prior for p0
        bp0 - parameter for Beta prior for p0
        ap1 - parameter for Beta prior for p1
        bp1 - parameter for Beta prior for p1
        theta - parameter for Poisson prior for the size of the marker k
        pgeomk - parameter for Geometric prior for the size of the marker k. If None, adopts Poisson prior.
        '''
        self.ap0 = ap0
        self.bp0 = bp0
        self.ap1 = ap1
        self.bp1 = bp1
        self.theta = theta
        self.pgeomk = pgeomk
        
        self.X = None
        self.y = None
        self.N = None
    
    def __apply_f(self, phi):
        '''
        Applies f to the current data matrix, returns number of affected individuals and number of positive and affected individuals
        '''
        # Number of positive individuals
        npos = np.sum(np.prod(self.X[:,phi], axis = 1))

        # Number of positive affected individuals
        nposaff = np.sum(np.prod(self.X[self.y==1][:,phi], axis = 1))
        
        return npos, nposaff
    
    def prior_p(self, p0,p1):
        '''
        Joint log-prior for pi0 and pi1. Uses two independent betas
        '''
        return beta.logpdf(x = p0, a = self.ap0, b = self.bp0) + sp.stats.beta.logpdf(x = p1, a = self.ap1, b = self.bp1)

    def prior_k(self, k):
        '''
        Log-prior for k, the number of atomic terms in the marker.
        Poisson 
        '''
        if k <= 0:
            return -np.inf
        else:
            if self.pgeomk is None:
                if self.theta is not None:
                    # Poisson for k - 1, since P(k = 0) = 0
                    return poisson.logpmf(k - 1, self.theta)
                else:
                    return 0
            else:
                return geom.logpmf(k-1, self.pgeomk)

    def prior_phi(self, phi):
        '''
        Log-prior for the set phi. For completeness reasons, uniform on the entire power set.
        '''
        return 0

    def loglike(self, p0, p1, phi):
        '''
        Log-likelihood
        '''

        if self.X is None or self.y is None:
            raise ValueError("No data provided!")
        
        npos, nposaff = self.__apply_f(phi)
        
        if npos == 0 or npos == self.N:
            # Does not accept markers such that none or all individuals are marked
            return -np.inf

        # To avoid division by 0
        p0 = max(p0, 1e-10)
        p1 = min(p1, 1-1e-10)

        # Calculating odds
        o1 = p1 / (1-p1)
        o0 = p0 / (1-p0)
        o10 = (1-p1)/(1-p0)

        return(self.N*np.log(1-p0) + nposaff*np.log(o1/o0) + self.Naff*np.log(o0) + npos*np.log(o10))   

    def post(self, p0, p1, phi):
        '''
        Posterior kernel
        '''
        
        if self.X is None or self.y is None:
            raise ValueError("No data provided!")
            

        return self.loglike(p0,p1,phi) + self.prior_phi(phi) + self.prior_k(len(phi)) + self.prior_p(p0, p1)
        
    
    def __run_chain(self, phicurr, nsteps = 1000, nwarm = 500):
        '''
        Generates samples from the MCMC for the single marker boolean function.
        
        @args
        phicurr - starting point, in the form of a numpy boolean vector of shape (p) (number of variables)
        nsteps - total number of MCMC steps
        nwarm  - warmup
        
        @return
        chain - list where each element is itself a list containing the marker, and values for p0 and p1
        '''
        
        if self.X is None or self.y is None:
            raise ValueError("No data provided!")
        
        # On-set of current point
        oncurr = np.where(phicurr == 1)[0]
        oncurr.sort()
        
        # Off-set of current point
        offcurr = np.where(phicurr == 0)[0]
        offcurr.sort()
        
        # Uniformly selects starting points for p0 and p1
        p0curr = np.random.uniform()
        p1curr = np.random.uniform()

        # Current posterior value
        pcurr = self.post(p0curr, p1curr, oncurr)
        
        # Number of positive and affected + positive individuals with current phi
        npos, nposaff = self.__apply_f(oncurr)

        chain =[]
        naccepted = [0,0,0,0]
        for i in range(nsteps):

            ### Move 1: sample p0, p1
            
            # Calculates parameter of conditional Beta posterior for p0
            a0post = self.ap0 + (self.Naff - nposaff)
            b0post = self.bp0 + (self.N - npos) - (self.Naff - nposaff)
            p0curr = np.random.beta(a = a0post, b = b0post, size = 1)
            
            # Posterior value
            pcurr = self.post(p0curr, p1curr, oncurr)
                        
            # Calculates parameter of conditional Beta posterior for p1
            a1post = self.ap1 + nposaff
            b1post = self.bp1 + npos - nposaff
            p1curr = np.random.beta(a = a1post, b = b1post, size = 1)

            # Posterior value
            pcurr = self.post(p0curr, p1curr, oncurr)
            
            ### Move 2: turn off an on bit
            if len(oncurr) > 0:
                
                # Selects position to remove from on-set
                pos = np.random.choice(oncurr, 1)
                oncand = np.delete(oncurr, np.where(oncurr == pos))

                # Posterior value
                pcand = self.post(p0curr, p1curr, oncand)

                if not np.isinf(pcand):
                    u = pcand - pcurr
                    if np.log(np.random.uniform()) < u:
                        oncurr = oncand
                        pcurr = pcand
                        phicurr[pos] = 0
                        offcurr = np.append(offcurr, pos)
                        naccepted[2] += 1
                        npos, nposaff = self.__apply_f(oncurr)

            ### Move 3: turn on an off bit
            if len(offcurr) > 0:
                
                # Selects position to include in the on-set
                pos = np.random.choice(offcurr, 1)
                oncand = np.append(oncurr, pos)

                pcand = self.post(p0curr, p1curr, oncand)

                if not np.isinf(pcand):
                    u = pcand - pcurr
                    if np.log(np.random.uniform()) < u:
                        oncurr = oncand
                        pcurr = pcand
                        phicurr[pos] = 1
                        offcurr = np.delete(offcurr, np.where(offcurr == pos))
                        naccepted[3] += 1
                        npos, nposaff = self.__apply_f(oncurr)

            oncurr.sort()
            offcurr.sort()

            chain.append([oncurr, p0curr, p1curr, pcurr])

        return chain[nwarm:]    
    
    def fit(self, X, y, nchains = 4, njobs = -1, nsteps = 1000, nwarm = 500):
        '''
        Runs nchains parallel chains, with parameter njobs for the Parallel library.
        
        @args
        X - numpy array with explanatory variables
        y - numpy array with response variable
        nchains - number of parallel chains
        njobs - number of parallel jobs
        nsteps - number of steps for each chain
        nwarm - warmup steps for each chain
                
        @returns
        chains - list of samples from the chain
        
        '''
    
        self.X = X
        self.y = y
        self.N = len(self.y)
        self.Naff = sum(self.y)
        
        # Starting point is the X row corresponding to an affected individual
        onset = np.where(self.y==1)[0]
        #phicurr_list = self.X[onset[np.random.choice(range(0, len(onset)), nchains, replace = False)],:]
        phicurr_list = self.X[np.random.choice(range(0, self.N), nchains, replace = False), :]
        
        markers = Parallel(n_jobs=njobs)(delayed(self.__run_chain)(start, nsteps, nwarm) for start in phicurr_list)
        
        return markers