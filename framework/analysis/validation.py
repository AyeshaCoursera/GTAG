"""Database validation"""

import requests
import time
import pandas as pd
from typing import Dict, List


class DatabaseValidator:
    """Validate gene findings against external databases."""
    
    def __init__(self, rate_limit: float = 0.3):
        """
        Initialize database validator.
        
        Parameters:
        -----------
        rate_limit : float
            Seconds between API calls
        """
        self.rate_limit = rate_limit
        
    def query_ensembl(self, gene_symbols: List[str], limit: int = 10) -> pd.DataFrame:
        """
        Query Ensembl REST API for gene information.
        
        Parameters:
        -----------
        gene_symbols : list
            List of gene symbols
        limit : int
            Maximum number of genes to query
            
        Returns:
        --------
        pd.DataFrame: Gene annotations
        """
        results = []
        for gene in gene_symbols[:limit]:
            try:
                url = f"https://rest.ensembl.org/lookup/symbol/homo_sapiens/{gene}"
                headers = {"Content-Type": "application/json"}
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    results.append({
                        'gene_symbol': gene,
                        'ensembl_id': data.get('id'),
                        'chromosome': data.get('seq_region_name'),
                        'biotype': data.get('biotype'),
                        'status': 'Found'
                    })
                else:
                    results.append({'gene_symbol': gene, 'status': 'Not found'})
                
                time.sleep(self.rate_limit)
                
            except Exception as e:
                results.append({'gene_symbol': gene, 'status': f'Error: {str(e)}'})
        
        return pd.DataFrame(results)
    
    def query_uniprot(self, gene_symbols: List[str], limit: int = 8) -> pd.DataFrame:
        """Query UniProt for protein information."""
        results = []
        
        for gene in gene_symbols[:limit]:
            try:
                url = "https://rest.uniprot.org/uniprotkb/search"
                params = {
                    "query": f"gene:{gene} AND organism_id:9606",
                    "fields": "accession,gene_names,protein_name",
                    "format": "json",
                    "size": 1
                }
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('results'):
                        result = data['results'][0]
                        protein_name = result.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', 'N/A')
                        results.append({
                            'gene': gene,
                            'uniprot_id': result.get('primaryAccession'),
                            'protein_name': protein_name[:50],
                            'status': 'Found'
                        })
                    else:
                        results.append({'gene': gene, 'status': 'Not found'})
                
                time.sleep(self.rate_limit)
                
            except Exception as e:
                results.append({'gene': gene, 'status': f'Error: {str(e)}'})
        
        return pd.DataFrame(results)
    
    def query_string_db(self, gene_symbols: List[str]) -> pd.DataFrame:
        """Query STRING database for protein interactions."""
        test_genes = ['ACTN2', 'ANKRD1', 'LAMA2', 'CD36', 'KLK3', 'ABCA3']
        
        try:
            url = "https://string-db.org/api/json/network"
            params = {
                "identifiers": "%0d".join(test_genes),
                "species": 9606,
                "required_score": 400,
                "caller_identity": "GTEx_research"
            }
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                for item in data:
                    results.append({
                        'gene1': item.get('preferredName_A', ''),
                        'gene2': item.get('preferredName_B', ''),
                        'score': item.get('score', 0)
                    })
                return pd.DataFrame(results)
                
        except Exception as e:
            print(f"STRING query error: {e}")
        
        return pd.DataFrame()
