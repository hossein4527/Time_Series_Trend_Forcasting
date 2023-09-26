import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
te = importr("RTransferEntropy")

class Net_moduls(object):
    def __init__(self):
        self.bw = 1

    def find_pearson_threshold(self, df, threshold):
        
        result_radius = []
        result_diameter = []
        result_densities = []
        result_number_of_edges = []
        result_average_clustering = []
        result_avg_degree_centrality = []
        result_avg_betweenness_centrality = []
        result_avg_closeness_centrality = []
        result_average_shortest_path_length = []
        
        trained_trhesholds = np.linspace(0, 1, 100)
        
        for thr in trained_trhesholds:
            
            trained_G = self.construct_network(df, 'pearson', thr)
            try:
                result_radius.append(nx.radius(trained_G))
            except :
                result_radius.append(np.nan)

            try:
                result_diameter.append(nx.diameter(trained_G))
            except :
                result_diameter.append(np.nan)

            try:
                result_densities.append(nx.density(trained_G))
            except :
                result_densities.append(np.nan)

            try:
                result_number_of_edges.append(nx.number_of_edges(trained_G))
            except :
                result_number_of_edges.append(np.nan)

            try:
                result_average_clustering.append(nx.average_clustering(trained_G))
            except :
                result_average_clustering.append(np.nan)

            try:
                result_avg_degree_centrality.append(np.mean(list(nx.degree_centrality(trained_G).values())))
            except :
                result_avg_degree_centrality.append(np.nan)

            try:
                result_avg_betweenness_centrality.append(np.mean(list(nx.betweenness_centrality(trained_G).values())))
            except :
                result_avg_betweenness_centrality.append(np.nan)

            try:
                result_avg_closeness_centrality.append(np.mean(list(nx.closeness_centrality(trained_G).values())))
            except :
                result_avg_closeness_centrality.append(np.nan)

            try:
                result_average_shortest_path_length.append(nx.average_shortest_path_length(trained_G))
            except :
                result_average_shortest_path_length.append(np.nan)
                
        return trained_trhesholds[np.where(np.array(result_number_of_edges) >= threshold)[0][-1]]

    def threshold_calc(df , method):
        if method == 'pearson':
            df_corr = df.corr(method ='pearson')
            avg_corr = []
            num_above_avg_corr = []
            for i in range(len(df.columns)):    
                corr_average = np.mean(df_corr[df_corr.columns[i]].values)
                avg_corr.append(corr_average)

                # market_corrs = df_corr[df_corr.columns[i]]
                # number_up_corr = len(market_corrs[market_corrs>corr_average])
                # num_above_avg_corr.append(number_up_corr)
        
            out1_avg_tot_corr = np.mean(avg_corr)
            return out1_avg_tot_corr
    

    def construct_network(self, df, method, threshold):
        if method == 'pearson':
            df_corr = df.corr(method ='pearson')       
        
            df_cc = df_corr
            # apply corr coeff threshold and create new df
            list_symbols = df_cc.columns.to_list()
            list_from = []
            list_to = []
            list_corr_coeff = []
            for i , sym_from in enumerate(list_symbols):
                for sym_to in list_symbols:
                    if sym_from != sym_to:
                        corr_coef = df_cc.loc[sym_from, sym_to]
                        if abs(corr_coef) > threshold:
                            list_from.append(sym_from)
                            list_to.append(sym_to)
                            list_corr_coeff.append(corr_coef)
                        
            # create df for constructing graph
            df_graph = pd.DataFrame({'from':list_from, 'to':list_to, 
                                'corr coeff':list_corr_coeff})
        
            G = nx.from_pandas_edgelist(df_graph, 'from', 'to')
            return G
    
        if method == 'entropy':
            te_list = []
            for c1 in range(df.columns.size):
                for c2 in range(df.columns.size):
                    te_list.append(te.calc_te(robjects.FloatVector(df[df.columns[c1]].values), robjects.FloatVector(df[df.columns[c2]].values))[0])
            te_list = np.array(te_list).reshape((df.columns.size, df.columns.size))
        
            df_corr = pd.DataFrame(te_list, columns=df.columns).set_index(df.columns)
        

            # compute correlation coeff matrix
            df_cc = df_corr


            # apply corr coeff threshold and create new df
            list_symbols = df_cc.columns.to_list()
            list_from = []
            list_to = []
            list_corr_coeff = []
            for i , sym_from in enumerate(list_symbols):
                for sym_to in list_symbols:
                    if sym_from != sym_to:
                        corr_coef = df_cc.loc[sym_from, sym_to]
                        if abs(corr_coef) > threshold:
                            list_from.append(sym_from)
                            list_to.append(sym_to) 
                            list_corr_coeff.append(corr_coef)

            # create df for constructing graph
            df_graph = pd.DataFrame({'from':list_from, 'to':list_to, 
                                    'corr coeff':list_corr_coeff})

            G_ent = nx.from_pandas_edgelist(df_graph, 'from', 'to')
            return G_ent
    
    def finding_ent_thr_per(self, df):
        te_list = []
        for c1 in range(df.columns.size):
            for c2 in range(df.columns.size):
                te_list.append(te.calc_te(robjects.FloatVector(df[df.columns[c1]].values), robjects.FloatVector(df[df.columns[c2]].values))[0])
        te_list = np.array(te_list).reshape((df.columns.size, df.columns.size))

        df_corr = pd.DataFrame(te_list, columns=df.columns).set_index(df.columns)
        suggested_thresholds = np.linspace(0, max(df_corr.max().values), 100)
        return suggested_thresholds

    def find_entropy_threshold(self, df, threshold):
        
        result_radius = []
        result_diameter = []
        result_densities = []
        result_number_of_edges = []
        result_average_clustering = []
        result_avg_degree_centrality = []
        result_avg_betweenness_centrality = []
        result_avg_closeness_centrality = []
        result_average_shortest_path_length = []
        
        trained_trhesholds = self.finding_ent_thr_per(df)
        
        for thr in trained_trhesholds:
            
            trained_G = self.construct_network(df, 'entropy', thr)
            try:
                result_radius.append(nx.radius(trained_G))
            except :
                result_radius.append(np.nan)

            try:
                result_diameter.append(nx.diameter(trained_G))
            except :
                result_diameter.append(np.nan)

            try:
                result_densities.append(nx.density(trained_G))
            except :
                result_densities.append(np.nan)

            try:
                result_number_of_edges.append(nx.number_of_edges(trained_G))
            except :
                result_number_of_edges.append(np.nan)

            try:
                result_average_clustering.append(nx.average_clustering(trained_G))
            except :
                result_average_clustering.append(np.nan)

            try:
                result_avg_degree_centrality.append(np.mean(list(nx.degree_centrality(trained_G).values())))
            except :
                result_avg_degree_centrality.append(np.nan)

            try:
                result_avg_betweenness_centrality.append(np.mean(list(nx.betweenness_centrality(trained_G).values())))
            except :
                result_avg_betweenness_centrality.append(np.nan)

            try:
                result_avg_closeness_centrality.append(np.mean(list(nx.closeness_centrality(trained_G).values())))
            except :
                result_avg_closeness_centrality.append(np.nan)

            try:
                result_average_shortest_path_length.append(nx.average_shortest_path_length(trained_G))
            except :
                result_average_shortest_path_length.append(np.nan)
                
        return trained_trhesholds[np.where(np.array(result_number_of_edges) >= threshold)[0][-1]]
    
    def df_reg(df, reg_num):
        return df[df['Regime_Code_ox']==reg_num].drop('Regime_Code',axis=1).drop('Regime_Code_ox',axis=1)


    def get_network_features(self, network):
        
        radius = []
        diameter = []
        densities = []
        number_of_edges = []
        average_clustering = []
        avg_degree_centrality = []
        avg_betweenness_centrality = []
        avg_closeness_centrality = []
        average_shortest_path_length = []
        
        col_names = ['radius', 'diameter', 'densities', 'number_of_edges', 'average_clustering', 'avg_degree_centrality',\
                'avg_betweenness_centrality', 'avg_closeness_centrality', 'average_shortest_path_length']

        try:
            radius.append(nx.radius(network))
        except :
            radius.append(np.nan)

        try:
            diameter.append(nx.diameter(network))
        except :
            diameter.append(np.nan)

        try:
            densities.append(nx.density(network))
        except :
            densities.append(np.nan)

        try:
            number_of_edges.append(nx.number_of_edges(network))
        except :
            number_of_edges.append(np.nan)

        try:
            average_clustering.append(nx.average_clustering(network))
        except :
            average_clustering.append(np.nan)

        try:
            avg_degree_centrality.append(np.mean(list(nx.degree_centrality(network).values())))
        except :
            avg_degree_centrality.append(np.nan)

        try:
            avg_betweenness_centrality.append(np.mean(list(nx.betweenness_centrality(network).values())))
        except :
            avg_betweenness_centrality.append(np.nan)

        try:
            avg_closeness_centrality.append(np.mean(list(nx.closeness_centrality(network).values())))
        except :
            avg_closeness_centrality.append(np.nan)

        try:
            average_shortest_path_length.append(nx.average_shortest_path_length(network))
        except :
            average_shortest_path_length.append(np.nan)
        
        df = pd.DataFrame(zip(radius, diameter, densities, number_of_edges, average_clustering, avg_degree_centrality, \
                            avg_betweenness_centrality, avg_closeness_centrality, average_shortest_path_length), columns=col_names)
        return df

    def price_return(p):
        pr = []
        for i in range(1, len(p)):
            pr.append(np.log(p[i])-np.log(p[i-1]))
        return pr


    def get_regimes_DataFrames(self, df, regimes_column):
        try:
            df['selected_regimes'] = regimes_column
        except:
            print('Please Make Sure Your Regime Column is the Same Size with the input DF')
        
        regs = np.unique(regimes_column)
        num_of_regime = len(regs)
        regimes_df = []

        for i in range(len(regs)):
            regimes_df.append(df[df['selected_regimes']==regs[i]].drop('selected_regimes', axis=1))
       
        print('Number of columns data in our DataFrame is = '+str(regimes_df[0].columns.size))

        return regimes_df


    def get_network_features_from_df(self,regimes_df, method ,extra_threshold, input_target_length=False):
        if input_target_length:
            target_length = input_target_length
        else:
            target_length = min([len(regimes_df[i]) for i in range(len(regimes_df))])

        cc = [regimes_df[i]['index'].sum() for i in range(len(regimes_df))]
        arr = np.argsort(cc)
        regimes_df = np.array(regimes_df)[arr]

        if method == 'pearson':

            for i , dfs in enumerate([regimes_df[k][-target_length:] for k in range(len(regimes_df))]):
                if i == 0: 
                    fixed_thr =  self.find_pearson_threshold(dfs , threshold = np.log(regimes_df[0].columns.size)*regimes_df[0].columns.size*extra_threshold)
                trained_G = self.construct_network(dfs, 'pearson',fixed_thr)
                dfs_net_features_df = self.get_network_features(trained_G)
                if i == 0:
                    net_features_df = dfs_net_features_df
                net_features_df.loc[i] = dfs_net_features_df.values[0]
            print('Selected Threshold is: '+ str(np.round(fixed_thr, 3)))
            return net_features_df

        if method == 'entropy':
            for i , dfs in enumerate([regimes_df[0][-target_length:], regimes_df[1][-target_length:]]):
                if i == 0: 
                    fixed_thr =  self.find_entropy_threshold(dfs , threshold = np.log(regimes_df[0].columns.size)*regimes_df[0].columns.size*extra_threshold)
                trained_G = self.construct_network(dfs, 'entropy',fixed_thr)
                dfs_net_features_df = self.get_network_features(trained_G)
                if i == 0:
                    net_features_df = dfs_net_features_df
                net_features_df.loc[i] = dfs_net_features_df.values[0]
            print('Selected Threshold is: '+ str(np.round(fixed_thr, 3)))
            return net_features_df

    def get_network_features_no_regimes(self, df, regimes_column,target_window, target_length, extra_threshold=1):
        regimes_percentage = []
        regs = np.unique(regimes_column)
        df['selected_regimes'] = regimes_column

        for wind in range(1, target_length):
            regimes_percentage.append([])
            for i , dfs in enumerate([df[-target_window-wind:-wind]]):
                if wind == 1:
                    fixed_thr =  self.find_pearson_threshold(dfs.drop('selected_regimes', axis=1), threshold = np.log(df.columns.size) * df.columns.size*extra_threshold )
                trained_G = self.construct_network(dfs, 'pearson',fixed_thr)
                dfs_net_features_df = self.get_network_features(trained_G)
                if wind == 1:
                    net_features_df = dfs_net_features_df
                net_features_df.loc[wind] = dfs_net_features_df.values[0]
                tt = df.loc[dfs.index]['selected_regimes'].values
                
                for re in range(len(regs)):
                    regimes_percentage[-1].append(tt[tt==regs[re]].size)
        return [net_features_df[::-1], np.array(regimes_percentage).T[::-1]]
        




                    
