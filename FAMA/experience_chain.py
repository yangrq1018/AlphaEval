import numpy as np
from sklearn.cluster import KMeans

class ExperienceChainSet:
    """Experience chain set for factor analysis."""

    def __init__(self):
        self.expressions: list[str] = []
        self.data: np.ndarray = np.empty((0, 0))
        self.ics: np.ndarray = np.empty((0,))
        self.chains: list['ExperienceChain'] = []
        
    
    def insert(self, chain: 'ExperienceChain') -> None:
        """
        Insert a new ExperienceChain into the set.
        
        Parameters
        ----------
        chain : ExperienceChain
            The ExperienceChain to insert.
        """
        self.chains.append(chain)
        self.expressions.extend(chain.expressions)
        print(f"Inserting chain with {len(chain.expressions)} expressions, data shape: {chain.data.shape}, ICs shape: {chain.ics.shape}")
        # self.data's shape is (0,0) and chain.data's shape is (n_samples, n_features)
        self.data = np.column_stack((self.data, chain.data)) if self.data.size else chain.data
        self.ics = np.concatenate((self.ics, chain.ics)) if self.ics.size else chain.ics
        

    def warmup(self, expressions: list[str], data: np.ndarray, ics: np.ndarray, kmeans: KMeans, k: int = 5):
        """
        Initialize the ExperienceChainSet with expressions, data, and ICs.
        
        Parameters
        ----------
        expressions : list[str]
            List of factor expressions.
        data : np.ndarray
            Data array of shape (n_samples, n_features).
        ics : np.ndarray
            Information coefficients array of shape (n_features).
        """
        self.expressions = expressions
        print(f"Warmup with {len(expressions)} expressions, data shape: {data.shape}, ICs shape: {ics.shape}")
        self.data = data
        self.ics = ics
        self.chains: list[ExperienceChain] = []
        self.kmeans = kmeans
        # Initialize chains based on the clustering
        for i in range(k):
            cluster_indices = np.where(self.kmeans.labels_ == i)[0]
            if len(cluster_indices) > 0:
                chain_expressions = [expressions[idx] for idx in cluster_indices]
                chain_data = data[:, cluster_indices]
                chain_ics = ics[cluster_indices]
                self.chains.append(ExperienceChain(chain_expressions, chain_data, chain_ics))
                
    def match(self, expression: str, data: np.ndarray) -> 'ExperienceChain':
        """
        Match the expression to the ExperienceChain.
        
        Parameters
        ----------
        expression : str
            The factor expression to match.
        
        Returns
        -------
        ExperienceChain
            The matched ExperienceChain.
        """
        corr = 0
        best_chain = None
        for chain in self.chains:
            if expression in chain.expressions:
                return chain
            # calculate correlation with existing data
            correlations = [np.abs(np.corrcoef(chain.data[:, i], data)[0][1]) for i in range(chain.data.shape[1])]
            max_corr = np.max(correlations)
            if max_corr > corr:
                corr = max_corr
                best_chain = chain
        return best_chain
    
class ExperienceChain:
    """Experience chain for factor analysis."""
    
    def __init__(self, expressions: list[str], data: np.ndarray, ics: np.ndarray):
        """
        Initialize the ExperienceChain with expressions, data, and ICs.
        Parameters
        ----------
        expressions : list[str]
            List of factor expressions.
        data : np.ndarray
            Data array of shape (n_samples, n_features).
        ics : np.ndarray
            Information coefficients array of shape (n_features).
        """
        self.expressions = expressions
        self.data = data
        self.ics = ics
        self.chain: list[int] = np.argsort(self.ics).tolist()
        
    def __str__(self) -> str:
        res = '"'+self.expressions[self.chain[0]]+'"'
        for i in range(1, len(self.chain)):
            res += ' -> "' + self.expressions[self.chain[i]] + '"'
        return res
            
    def insert(self, expr: str, data: np.ndarray, ic: float) -> None:
        """
        Insert a new expression into the chain based on its IC.
        
        Parameters
        ----------
        expr : str
            The factor expression to insert.
        ic : float
            The information coefficient of the expression.
        data : np.ndarray
            The data associated with the expression.
        """
        if ic > np.max(self.ics):
            # calculate correlation with existing data
            correlations = [np.abs(np.corrcoef(self.data[:,i], data)[0][1])for i in range(self.data.shape[1])]
            # insert at the next behind the most correlated expression
            max_corr_index = np.argmax(correlations)
            # delete the expressions behind the most correlated one
            index_in_chain = self.chain.index(max_corr_index)
            self.chain = self.chain[:index_in_chain + 1]
            self.chain.append(len(self.expressions))
            self.expressions.append(expr)
            # update the data and ICs
            self.data = np.column_stack((self.data, data))
            self.ics = np.append(self.ics, ic)
            
            
            