import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.linalg.graphmatrix import adjacency_matrix
import math
from visibility_graph import visibility_graph

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
te = importr("RTransferEntropy")

from nxviz.plots import CircosPlot

class VG_Modules_Model1(object):
    def __init__(self):
        self.bw = 1
        
    #Usual Model One Prediction functions
    def most_similar_node(self,time_series, T=20, tp=30):
        G = visibility_graph(time_series)
        adj_matrix = np.array(adjacency_matrix(G).todense())
        E = G.number_of_edges()
        k = adj_matrix.sum(1)
        P = adj_matrix/k
        n = k.shape[0]
        s_srw = np.zeros((n,n))
        pi = np.diag(np.ones(n))
        for t in range(T):
            pi = P@pi
            s_srw += (k*pi+k.T*pi.T)/(2*E)
        if tp != 0:
            triup = np.triu(s_srw,k=-tp)
            triup[triup != 0] = 1
            s_srw = np.tril(s_srw,k=-1) * triup
        return np.argmax(s_srw,axis=1)

        
    def ts_forcast_model1(self,time_serie):
        start_point = 31
        forcast_ts = np.zeros_like(time_serie)
        forcast_ts[:start_point] = time_serie[:start_point]
        msn = self.most_similar_node(time_serie,T=20,tp=30)
        for i in range(start_point-1,time_serie.shape[0]-1):
            dy = time_serie[i]-time_serie[msn[i]]
            y = time_serie[i]+dy/(i-msn[i])
            yhat = (i-msn[i])/(i-msn[i]+1)*y+(1/(i-msn[i]+1))*time_serie[i]
            forcast_ts[i+1]=yhat
        return forcast_ts
    
    #Prediction Model One Based on 2 Regimed Data
    def most_similar_matrix(self, time_series, T=20, tp=30):
        G = visibility_graph(time_series)
        adj_matrix = np.array(adjacency_matrix(G).todense())
        E = G.number_of_edges()
        k = adj_matrix.sum(1)
        P = adj_matrix/k
        n = k.shape[0]
        s_srw = np.zeros((n,n))
        pi = np.diag(np.ones(n))
        for t in range(T):
            pi = P@pi
            s_srw += (k*pi+k.T*pi.T)/(2*E)
        if tp != 0:
            triup = np.triu(s_srw,k=-tp)
            triup[triup != 0] = 1
            s_srw = np.tril(s_srw,k=-1) * triup
        return s_srw

    def selective_msn_2regimed_Ox(self, srw, df, tp=20):
        indx = []
        rg_3 = []
        ms_reg = []
        for k in range(len(srw)):
            indx.append([])
            rg_3.append([])
            rg_3[-1].append(df.iloc[k]['Ox_2'])
            for i in range(1,tp):
                indx[-1].append(np.where(srw[k] == np.sort(srw[k])[-i])[0][0])
                rg_3[-1].append(df.iloc[indx[-1][-1]]['Ox_2'])
            try:
                ms_reg.append(indx[-1][np.where(rg_3[-1]==rg_3[-1][0])[0][1]])
            except IndexError:
                ms_reg.append(indx[-1][1])
        return ms_reg

    
    def ts_forcast_2Regimed_model1(self, time_serie, df):
        start_point = 31
        forcast_ts = np.zeros_like(time_serie)
        forcast_ts[:start_point] = time_serie[:start_point]
        msns = self.most_similar_matrix(time_serie,T=20,tp=50)
        msn = self.selective_msn_2regimed_Ox(msns, df)
        for i in range(start_point-1,time_serie.shape[0]-1):
            dy = time_serie[i]-time_serie[msn[i]]
            y = time_serie[i]+dy/(i-msn[i])
            yhat = (i-msn[i])/(i-msn[i]+1)*y+(1/(i-msn[i]+1))*time_serie[i]
            forcast_ts[i+1]=yhat
        return forcast_ts
    
    #Prediction Model One Based on 12 Regimed Data
    def selective_msn_12regmied(self, srw, df, tp=20):
        indx = []
        rg_3 = []
        ms_reg = []
        for k in range(len(srw)):
            indx.append([])
            rg_3.append([])
            rg_3[-1].append(df.iloc[k]['12 today/tommorrow'])
            for i in range(1,tp):
                indx[-1].append(np.where(srw[k] == np.sort(srw[k])[-i])[0][0])
                rg_3[-1].append(df.iloc[indx[-1][-1]]['12 today/tommorrow'])
            try:
                ms_reg.append(indx[-1][np.where(rg_3[-1]==rg_3[-1][0])[0][1]])
            except IndexError:
                ms_reg.append(indx[-1][1])
        return ms_reg
    
    def ts_forcast_12Regimed_model1(self, time_serie, df):
        start_point = 31
        forcast_ts = np.zeros_like(time_serie)
        forcast_ts[:start_point] = time_serie[:start_point]
        msns = self.most_similar_matrix(time_serie,T=20,tp=50)
        msn = self.selective_msn_12regmied(msns, df)
        for i in range(start_point-1,time_serie.shape[0]-1):
            dy = time_serie[i]-time_serie[msn[i]]
            y = time_serie[i]+dy/(i-msn[i])
            yhat = (i-msn[i])/(i-msn[i]+1)*y+(1/(i-msn[i]+1))*time_serie[i]
            forcast_ts[i+1]=yhat
        return forcast_ts
    
    def error_bars(self, selected_data, forcast_ts):
        RMSE = math.sqrt(np.square(np.subtract(selected_data,forcast_ts)).mean())
        NRMSE = RMSE/(max(selected_data)-min(selected_data))
        MAPE=np.mean(np.abs((selected_data - forcast_ts)/selected_data))*100
        MAP=np.mean(np.abs((selected_data - forcast_ts)))
        return RMSE, NRMSE, MAPE, MAP

    #Prediction Model One Based on 3 Regimed Data
    def selective_msn_3regmied(self, srw, df, tp=20):
        indx = []
        rg_3 = []
        ms_reg = []
        for k in range(len(srw)):
            indx.append([])
            rg_3.append([])
            rg_3[-1].append(df.iloc[k]['Reg_3'])
            for i in range(1,tp):
                indx[-1].append(np.where(srw[k] == np.sort(srw[k])[-i])[0][0])
                rg_3[-1].append(df.iloc[indx[-1][-1]]['Reg_3'])
            try:
                ms_reg.append(indx[-1][np.where(rg_3[-1]==rg_3[-1][0])[0][1]])
            except IndexError:
                ms_reg.append(indx[-1][1])
        return ms_reg
    
    def ts_forcast_3Regimed_model1(self, time_serie, df):
        start_point = 31
        forcast_ts = np.zeros_like(time_serie)
        forcast_ts[:start_point] = time_serie[:start_point]
        msns = self.most_similar_matrix(time_serie,T=20,tp=50)
        msn = self.selective_msn_3regmied(msns, df)
        for i in range(start_point-1,time_serie.shape[0]-1):
            dy = time_serie[i]-time_serie[msn[i]]
            y = time_serie[i]+dy/(i-msn[i])
            yhat = (i-msn[i])/(i-msn[i]+1)*y+(1/(i-msn[i]+1))*time_serie[i]
            forcast_ts[i+1]=yhat
        return forcast_ts
    
    def error_bars(self, selected_data, forcast_ts):
        RMSE = math.sqrt(np.square(np.subtract(selected_data,forcast_ts)).mean())
        NRMSE = RMSE/(max(selected_data)-min(selected_data))
        MAPE=np.mean(np.abs((selected_data - forcast_ts)/selected_data))*100
        MAP=np.mean(np.abs((selected_data - forcast_ts)))
        return RMSE, NRMSE, MAPE, MAP
    
class VG_Modules_Model2(object):
    def __init__(self):
        self.bw = 1
    
    #Usual Model Two Prediction functions
    def node_similarity(self, time_series, T=20, tp=50):
        G = visibility_graph(time_series)
        adj_matrix = np.array(adjacency_matrix(G).todense())
        E = G.number_of_edges()
        k = adj_matrix.sum(1)
        P = adj_matrix/k
        n = k.shape[0]
        s_srw = np.zeros((n,n))
        pi = np.diag(np.ones(n))
        for t in range(T):
            pi = P@pi
            s_srw += (k*pi+k.T*pi.T)/(2*E)
        if tp != 0:
            triup = np.triu(s_srw,k=-tp)
            triup[triup != 0] = 1
            s_srw = np.tril(s_srw,k=-1) * triup
        return np.argmax(s_srw,axis=1) , s_srw
    
    def ts_forcast_model2_weighted(self,time_serie, tp=30):
        arg , s_srw = self.node_similarity(time_serie,T=20,tp=30)
        sum_s = s_srw.sum(axis=1)
        sum_s[sum_s == 0] = np.inf
        ws_m = (s_srw.T/sum_s).T
        ws = np.zeros((ws_m.shape[0], tp))
        for i in range(ws_m.shape[0]):
            ws[i] = np.roll(ws_m[i], tp-i)[:tp]
        start_point = 31
        forcast_ts = np.zeros_like(time_serie)
        forcast_ts[:start_point] = time_serie[:start_point]
        for i in range(start_point-1,time_serie.shape[0]-1):
            y = time_serie[i]+(time_serie[i]-time_serie[i-tp:i])/(i-np.arange(i-tp,i,1))
            forcast_ts[i+1]=(ws[i]*y).sum()
        
        return forcast_ts
    
    #Prediction Model Two Based on 2 Regimed Data
    def most_similarity_matrix(self,time_series, T=20, tp=50):
        G = visibility_graph(time_series)
        adj_matrix = np.array(adjacency_matrix(G).todense())
        E = G.number_of_edges()
        k = adj_matrix.sum(1)
        P = adj_matrix/k
        n = k.shape[0]
        s_srw = np.zeros((n,n))
        pi = np.diag(np.ones(n))
        for t in range(T):
            pi = P@pi
            s_srw += (k*pi+k.T*pi.T)/(2*E)
        if tp != 0:
            triup = np.triu(s_srw,k=-tp)
            triup[triup != 0] = 1
            s_srw = np.tril(s_srw,k=-1) * triup
        return s_srw
    
    def selective_msn_2Regimed_Ox_model2_weighted(self,srw,df,  tp=20):
        indx = []
        rg_3 = []
        ms_reg = []
        for k in range(len(srw)):
            indx.append([])
            rg_3.append([])
            rg_3[-1].append(df.iloc[k]['Ox_2'])
            for i in range(1,tp):
                indx[-1].append(np.where(srw[k] == np.sort(srw[k])[-i])[0][0])
                rg_3[-1].append(df.iloc[indx[-1][-1]]['Ox_2'])
            try:
                ms_reg.append(indx[-1][np.where(rg_3[-1]==rg_3[-1][0])[0][1]])
            except IndexError:
                ms_reg.append(indx[-1][1])
        return ms_reg
    
    def ts_forcast_2Regimed_model2_weighted(self, time_serie, df):
        start_point = 31
        forcast_ts = np.zeros_like(time_serie)
        forcast_ts[:start_point] = time_serie[:start_point]
        msns = self.most_similarity_matrix(time_serie,T=20,tp=50)
        msn = self.selective_msn_2Regimed_Ox_model2_weighted(msns, df)
        for i in range(start_point-1,time_serie.shape[0]-1):
            dy = time_serie[i]-time_serie[msn[i]]
            y = time_serie[i]+dy/(i-msn[i])
            yhat = (i-msn[i])/(i-msn[i]+1)*y+(1/(i-msn[i]+1))*time_serie[i]
            forcast_ts[i+1]=yhat
        return forcast_ts
    
    #Prediction Model Two Based on 3 Regimed Data
    def most_similarity_matrix(self,time_series, T=20, tp=50):
        G = visibility_graph(time_series)
        adj_matrix = np.array(adjacency_matrix(G).todense())
        E = G.number_of_edges()
        k = adj_matrix.sum(1)
        P = adj_matrix/k
        n = k.shape[0]
        s_srw = np.zeros((n,n))
        pi = np.diag(np.ones(n))
        for t in range(T):
            pi = P@pi
            s_srw += (k*pi+k.T*pi.T)/(2*E)
        if tp != 0:
            triup = np.triu(s_srw,k=-tp)
            triup[triup != 0] = 1
            s_srw = np.tril(s_srw,k=-1) * triup
        return s_srw
    
    def selective_msn_3Regimed_Ox_model2_weighted(self,srw,df,  tp=20):
        indx = []
        rg_3 = []
        ms_reg = []
        for k in range(len(srw)):
            indx.append([])
            rg_3.append([])
            rg_3[-1].append(df.iloc[k]['Reg_3'])
            for i in range(1,tp):
                indx[-1].append(np.where(srw[k] == np.sort(srw[k])[-i])[0][0])
                rg_3[-1].append(df.iloc[indx[-1][-1]]['Reg_3'])
            try:
                ms_reg.append(indx[-1][np.where(rg_3[-1]==rg_3[-1][0])[0][1]])
            except IndexError:
                ms_reg.append(indx[-1][1])
        return ms_reg
    
    def ts_forcast_3Regimed_model2_weighted(self, time_serie, df):
        start_point = 31
        forcast_ts = np.zeros_like(time_serie)
        forcast_ts[:start_point] = time_serie[:start_point]
        msns = self.most_similarity_matrix(time_serie,T=20,tp=50)
        msn = self.selective_msn_3Regimed_Ox_model2_weighted(msns, df)
        for i in range(start_point-1,time_serie.shape[0]-1):
            dy = time_serie[i]-time_serie[msn[i]]
            y = time_serie[i]+dy/(i-msn[i])
            yhat = (i-msn[i])/(i-msn[i]+1)*y+(1/(i-msn[i]+1))*time_serie[i]
            forcast_ts[i+1]=yhat
        return forcast_ts
    
    def ts_forcast_3Regimed_weighted_compact(self, df , time_serie):
        s_srw = self.most_similarity_matrix(time_serie, T=20,tp=50)
        sum_s = s_srw.sum(axis=1)
        sum_s[sum_s == 0] = np.inf
        ws_m = (s_srw.T/sum_s).T
        ws = np.zeros((ws_m.shape[0], tp))
        
        srw = s_srw
        indx = []
        rg_3 = []
        ms_reg = []
        total_y_indexs = []
        total_y_srws = []
        tp=50
        for k in range(len(srw)):
            total_y_indexs.append([])
            total_y_srws.append([])
            indx.append([])
            # indx[-1].append(0)
            rg_3.append([])
            rg_3[-1].append(df.iloc[k]['3_regime_index'])
            for i in range(tp):
                indx[-1].append(np.where(srw[k] == np.sort(srw[k])[::-1][:tp][i])[0][0])
                rg_3[-1].append(df.iloc[indx[-1][-1]]['3_regime_index'])
                
            if rg_3[k][0] != 2:
                same_reg = np.where(rg_3[k]==rg_3[k][0])[0]
                turn_points = np.where(np.array(rg_3[k])==2)[0]
                num_same_regime = len(same_reg)
                num_turn_ponits = len(turn_points)
                indx_same = np.array(indx[k])[list(same_reg[1:]-1)]
                indx_2 = np.array(indx[k])[list(turn_points-1)]
                total_y_indexs[-1].append(np.concatenate((indx_same,indx_2), axis=0))
                srw[k][indx_2] *= num_turn_ponits
                srw[k][indx_same] *= num_same_regime
                ssr = srw[k][total_y_indexs[-1]]
                
            
                total_y_srws[-1].append(ssr[0]/sum(ssr[0]))
                
            if rg_3[k][0] == 2:
                same_reg = np.where(rg_3[k]==rg_3[k][0])[0]
                other_points = np.where(np.array(rg_3[k])!=2)[0]
                num_same_regime = len(same_reg)
                num_turn_ponits = len(other_points)
                indx_same = np.array(indx[k])[list(same_reg[1:]-1)]
                indx_2 = np.array(indx[k])[list(other_points-1)]
                total_y_indexs[-1].append(np.concatenate((indx_same,indx_2), axis=0))
                
                srw[k][indx_2] *= num_turn_ponits
                srw[k][indx_same] *= num_same_regime 
                
                ssr = srw[k][total_y_indexs[-1]]
                # ssr_0 , ssr_1 = np.unique(ssr[0], return_index=True)
                # ssr = ssr_0[np.argsort(ssr_1)]
                # ssr = ssr[ssr!=0]
                # total_y_indexs = total_y_indexs[np.argsort(ssr_1)]
                
                total_y_srws[-1].append(ssr[0]/sum(ssr[0]))
                
                
        start_point = 31
        forcast_ts = np.zeros_like(time_serie)
        forcast_ts[:start_point] = time_serie[:start_point]
        for i in range(start_point-1,time_serie.shape[0]-1):
            y = time_serie[i]+(time_serie[i]-time_serie[total_y_indexs[i][0]])/(i-total_y_indexs[i][0])
            forcast_ts[i+1]=(total_y_srws[i]*y).sum()
            
        return forcast_ts
    
    def error_bars(self, selected_data, forcast_ts):
        RMSE = math.sqrt(np.square(np.subtract(selected_data,forcast_ts)).mean())
        NRMSE = RMSE/(max(selected_data)-min(selected_data))
        MAPE=np.mean(np.abs((selected_data - forcast_ts)/selected_data))*100
        MAP=np.mean(np.abs((selected_data - forcast_ts)))
        return RMSE, NRMSE, MAPE, MAP
    
class VG_Modules_Model3(object):
    def __init__(self):
        self.bw = 1
    
    #Usual Model Two Prediction functions
    def node_similarity(self, time_series, T=20, tp=50):
        G = visibility_graph(time_series)
        adj_matrix = np.array(adjacency_matrix(G).todense())
        E = G.number_of_edges()
        k = adj_matrix.sum(1)
        P = adj_matrix/k
        n = k.shape[0]
        s_srw = np.zeros((n,n))
        pi = np.diag(np.ones(n))
        for t in range(T):
            pi = P@pi
            s_srw += (k*pi+k.T*pi.T)/(2*E)
        if tp != 0:
            triup = np.triu(s_srw,k=-tp)
            triup[triup != 0] = 1
            s_srw = np.tril(s_srw,k=-1) * triup
        return np.argmax(s_srw,axis=1) , s_srw
    
    def ts_forcast_model2_weighted(self,time_serie, tp=30):
        arg , s_srw = self.node_similarity(time_serie,T=20,tp=30)
        sum_s = s_srw.sum(axis=1)
        sum_s[sum_s == 0] = np.inf
        ws_m = (s_srw.T/sum_s).T
        ws = np.zeros((ws_m.shape[0], tp))
        for i in range(ws_m.shape[0]):
            ws[i] = np.roll(ws_m[i], tp-i)[:tp]
        start_point = 31
        forcast_ts = np.zeros_like(time_serie)
        forcast_ts[:start_point] = time_serie[:start_point]
        for i in range(start_point-1,time_serie.shape[0]-1):
            y = time_serie[i]+(time_serie[i]-time_serie[i-tp:i])/(i-np.arange(i-tp,i,1))
            forcast_ts[i+1]=(ws[i]*y).sum()
        
        return forcast_ts
    
    #Prediction Model Two Based on 2 Regimed Data
    def most_similarity_matrix(self,time_series, T=20, tp=50):
        G = visibility_graph(time_series)
        adj_matrix = np.array(adjacency_matrix(G).todense())
        E = G.number_of_edges()
        k = adj_matrix.sum(1)
        P = adj_matrix/k
        n = k.shape[0]
        s_srw = np.zeros((n,n))
        pi = np.diag(np.ones(n))
        for t in range(T):
            pi = P@pi
            s_srw += (k*pi+k.T*pi.T)/(2*E)
        if tp != 0:
            triup = np.triu(s_srw,k=-tp)
            triup[triup != 0] = 1
            s_srw = np.tril(s_srw,k=-1) * triup
        return s_srw
    
    def selective_msn_2Regimed_Ox_model2_weighted(self,srw,df,  tp=20):
        indx = []
        rg_3 = []
        ms_reg = []
        for k in range(len(srw)):
            indx.append([])
            rg_3.append([])
            rg_3[-1].append(df.iloc[k]['Ox_2'])
            for i in range(1,tp):
                indx[-1].append(np.where(srw[k] == np.sort(srw[k])[-i])[0][0])
                rg_3[-1].append(df.iloc[indx[-1][-1]]['Ox_2'])
            try:
                ms_reg.append(indx[-1][np.where(rg_3[-1]==rg_3[-1][0])[0][1]])
            except IndexError:
                ms_reg.append(indx[-1][1])
        return ms_reg
    
    def ts_forcast_2Regimed_model2_weighted(self, time_serie, df):
        start_point = 31
        forcast_ts = np.zeros_like(time_serie)
        forcast_ts[:start_point] = time_serie[:start_point]
        msns = self.most_similarity_matrix(time_serie,T=20,tp=50)
        msn = self.selective_msn_2Regimed_Ox_model2_weighted(msns, df)
        for i in range(start_point-1,time_serie.shape[0]-1):
            dy = time_serie[i]-time_serie[msn[i]]
            y = time_serie[i]+dy/(i-msn[i])
            yhat = (i-msn[i])/(i-msn[i]+1)*y+(1/(i-msn[i]+1))*time_serie[i]
            forcast_ts[i+1]=yhat
        return forcast_ts
    
    #Prediction Model Two Based on 3 Regimed Data
    def most_similarity_matrix(self,time_series, T=20, tp=50):
        G = visibility_graph(time_series)
        adj_matrix = np.array(adjacency_matrix(G).todense())
        E = G.number_of_edges()
        k = adj_matrix.sum(1)
        P = adj_matrix/k
        n = k.shape[0]
        s_srw = np.zeros((n,n))
        pi = np.diag(np.ones(n))
        for t in range(T):
            pi = P@pi
            s_srw += (k*pi+k.T*pi.T)/(2*E)
        if tp != 0:
            triup = np.triu(s_srw,k=-tp)
            triup[triup != 0] = 1
            s_srw = np.tril(s_srw,k=-1) * triup
        return s_srw
    
    def selective_msn_3Regimed_Ox_model2_weighted(self,srw,df,  tp=20):
        indx = []
        rg_3 = []
        ms_reg = []
        for k in range(len(srw)):
            indx.append([])
            rg_3.append([])
            rg_3[-1].append(df.iloc[k]['Reg_3'])
            for i in range(1,tp):
                indx[-1].append(np.where(srw[k] == np.sort(srw[k])[-i])[0][0])
                rg_3[-1].append(df.iloc[indx[-1][-1]]['Reg_3'])
            try:
                ms_reg.append(indx[-1][np.where(rg_3[-1]==rg_3[-1][0])[0][1]])
            except IndexError:
                ms_reg.append(indx[-1][1])
        return ms_reg
    
    def ts_forcast_3Regimed_model2_weighted(self, time_serie, df):
        start_point = 31
        forcast_ts = np.zeros_like(time_serie)
        forcast_ts[:start_point] = time_serie[:start_point]
        msns = self.most_similarity_matrix(time_serie,T=20,tp=50)
        msn = self.selective_msn_3Regimed_Ox_model2_weighted(msns, df)
        for i in range(start_point-1,time_serie.shape[0]-1):
            dy = time_serie[i]-time_serie[msn[i]]
            y = time_serie[i]+dy/(i-msn[i])
            yhat = (i-msn[i])/(i-msn[i]+1)*y+(1/(i-msn[i]+1))*time_serie[i]
            forcast_ts[i+1]=yhat
        return forcast_ts
    
    def ts_forcast_3Regimed_weighted_compact(self, df , time_serie):
        s_srw = self.most_similarity_matrix(time_serie, T=20,tp=50)
        sum_s = s_srw.sum(axis=1)
        sum_s[sum_s == 0] = np.inf
        ws_m = (s_srw.T/sum_s).T
        ws = np.zeros((ws_m.shape[0], tp))
        
        srw = s_srw
        indx = []
        rg_3 = []
        ms_reg = []
        total_y_indexs = []
        total_y_srws = []
        tp=50
        for k in range(len(srw)):
            total_y_indexs.append([])
            total_y_srws.append([])
            indx.append([])
            # indx[-1].append(0)
            rg_3.append([])
            rg_3[-1].append(df.iloc[k]['3_regime_index'])
            for i in range(tp):
                indx[-1].append(np.where(srw[k] == np.sort(srw[k])[::-1][:tp][i])[0][0])
                rg_3[-1].append(df.iloc[indx[-1][-1]]['3_regime_index'])
                
            if rg_3[k][0] != 2:
                same_reg = np.where(rg_3[k]==rg_3[k][0])[0]
                turn_points = np.where(np.array(rg_3[k])==2)[0]
                num_same_regime = len(same_reg)
                num_turn_ponits = len(turn_points)
                indx_same = np.array(indx[k])[list(same_reg[1:]-1)]
                indx_2 = np.array(indx[k])[list(turn_points-1)]
                total_y_indexs[-1].append(np.concatenate((indx_same,indx_2), axis=0))
                srw[k][indx_2] *= num_turn_ponits
                srw[k][indx_same] *= num_same_regime
                ssr = srw[k][total_y_indexs[-1]]
                
            
                total_y_srws[-1].append(ssr[0]/sum(ssr[0]))
                
            if rg_3[k][0] == 2:
                same_reg = np.where(rg_3[k]==rg_3[k][0])[0]
                other_points = np.where(np.array(rg_3[k])!=2)[0]
                num_same_regime = len(same_reg)
                num_turn_ponits = len(other_points)
                indx_same = np.array(indx[k])[list(same_reg[1:]-1)]
                indx_2 = np.array(indx[k])[list(other_points-1)]
                total_y_indexs[-1].append(np.concatenate((indx_same,indx_2), axis=0))
                
                srw[k][indx_2] *= num_turn_ponits
                srw[k][indx_same] *= num_same_regime 
                
                ssr = srw[k][total_y_indexs[-1]]
                # ssr_0 , ssr_1 = np.unique(ssr[0], return_index=True)
                # ssr = ssr_0[np.argsort(ssr_1)]
                # ssr = ssr[ssr!=0]
                # total_y_indexs = total_y_indexs[np.argsort(ssr_1)]
                
                total_y_srws[-1].append(ssr[0]/sum(ssr[0]))
                
                
        start_point = 31
        forcast_ts = np.zeros_like(time_serie)
        forcast_ts[:start_point] = time_serie[:start_point]
        for i in range(start_point-1,time_serie.shape[0]-1):
            y = time_serie[i]+(time_serie[i]-time_serie[total_y_indexs[i][0]])/(i-total_y_indexs[i][0])
            forcast_ts[i+1]=(total_y_srws[i]*y).sum()
            
        return forcast_ts
    
    def error_bars(self, selected_data, forcast_ts):
        RMSE = math.sqrt(np.square(np.subtract(selected_data,forcast_ts)).mean())
        NRMSE = RMSE/(max(selected_data)-min(selected_data))
        MAPE=np.mean(np.abs((selected_data - forcast_ts)/selected_data))*100
        MAP=np.mean(np.abs((selected_data - forcast_ts)))
        return RMSE, NRMSE, MAPE, MAP


class Net_moduls(object):
    def __init__(self):
        self.bw = 1

    def price_return(p):
        pr = []
        for i in range(1, len(p)):
            pr.append(np.log(p[i])-np.log(p[i-1]))
        return pr

    def find_pearson_threshold(self, df, threshold):
        result_number_of_edges = []
        trained_trhesholds = np.linspace(0, 1, 100)
        for thr in trained_trhesholds:
            trained_G = self.construct_network(df, 'pearson', thr)
            try:
                result_number_of_edges.append(nx.number_of_edges(trained_G))
            except :
                result_number_of_edges.append(np.nan)
        return trained_trhesholds[np.where(np.array(result_number_of_edges) >= threshold)[0][-1]]
    
    def find_median_pearson_threshold(self,df):
        df_corr = df.corr(method ='pearson')  
        df_corr_flatten = np.array(df_corr).flatten()
        df_corr_flatten_nonNan = df_corr_flatten[~np.isnan(df_corr_flatten)]
        df_corr_flatten_nonNan.sort()
        median_corr_coef = np.median(df_corr_flatten_nonNan)
        return median_corr_coef

    def find_median_entropy_threshold(self,df):
        te_list = []
        for c1 in range(df.columns.size):
            for c2 in range(df.columns.size):
                te_list.append(te.calc_te(robjects.FloatVector(df[df.columns[c1]].values), robjects.FloatVector(df[df.columns[c2]].values))[0])
        te_list = np.array(te_list).reshape((df.columns.size, df.columns.size))
        df_corr_flatten = np.array(te_list).flatten()
        df_corr_flatten = df_corr_flatten[df_corr_flatten>0]
        df_corr_flatten_nonNan = df_corr_flatten[~np.isnan(df_corr_flatten)]
        df_corr_flatten_nonNan.sort()
        median_corr_coef = np.median(df_corr_flatten_nonNan)
        return median_corr_coef
    
    def find_median_crossed_threshold(self,df):
        cross_offsets = []
        for i in range(len(df.columns)):
            for j in range(len(df.columns)):
                seconds = 5
                fps = 30
                rs = [self.crosscorr(df[df.columns[i]],df[df.columns[j]], lag) for lag in range(-int(seconds*fps),int(seconds*fps+1))]
                offset = np.floor(len(rs)/2)-np.argmax(rs)
                cross_offsets.append(offset)
        cross_offsets = np.array(cross_offsets)
        cross_offsets = cross_offsets[cross_offsets>0]
        df_corr_flatten_nonNan = cross_offsets[~np.isnan(cross_offsets)]
        df_corr_flatten_nonNan.sort()
        df_corr_flatten_nonNan = df_corr_flatten_nonNan/max(df_corr_flatten_nonNan)
        median_corr_coef = np.median(df_corr_flatten_nonNan)
        return median_corr_coef
    
    def find_median_crossed_100_threshold(self,df):
        cross_offsets = []
        for i in range(len(df.columns)):
            for j in range(len(df.columns)):
                seconds = 5
                fps = 5
                rs = [self.crosscorr(df[df.columns[i]],df[df.columns[j]], lag) for lag in range(-int(seconds*fps),int(seconds*fps+1))]
                offset = np.floor(len(rs)/2)-np.argmax(rs)
                cross_offsets.append(offset)
        cross_offsets = np.array(cross_offsets)
        cross_offsets = cross_offsets[cross_offsets>0]
        df_corr_flatten_nonNan = cross_offsets[~np.isnan(cross_offsets)]
        df_corr_flatten_nonNan.sort()
        df_corr_flatten_nonNan = df_corr_flatten_nonNan/max(df_corr_flatten_nonNan)
        median_corr_coef = np.median(df_corr_flatten_nonNan)
        return median_corr_coef
    
    def find_ent_thr_period(self, df):
        te_list = []
        for c1 in range(df.columns.size):
            for c2 in range(df.columns.size):
                te_list.append(te.calc_te(robjects.FloatVector(df[df.columns[c1]].values), robjects.FloatVector(df[df.columns[c2]].values))[0])
        te_list = np.array(te_list).reshape((df.columns.size, df.columns.size))

        df_corr = pd.DataFrame(te_list, columns=df.columns).set_index(df.columns)
        suggested_thresholds = np.linspace(0, max(df_corr.max().values), 100)
        return suggested_thresholds

    def find_entropy_threshold(self, df, threshold):
        result_number_of_edges = []
        trained_trhesholds = self.find_ent_thr_period(df)
        for thr in trained_trhesholds:
            trained_G = self.construct_network(df, 'entropy', thr)
            try:
                result_number_of_edges.append(nx.number_of_edges(trained_G))
            except :
                result_number_of_edges.append(np.nan)
        return trained_trhesholds[np.where(np.array(result_number_of_edges) >= threshold)[0][-1]]
    
    def crosscorr(self,datax, datay, lag=0, wrap=False):
        if wrap:
            shiftedy = datay.shift(lag)
            transfer_entropy_dfshiftedy.iloc[:lag] = datay.iloc[-lag:].values
            return datax.corr(shiftedy)
        else:
            return datax.corr(datay.shift(lag))

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
        
        if method == 'crossed':
            cross_offsets = []
            for i in range(len(df.columns)):
                for j in range(len(df.columns)):
                    seconds = 5
                    fps = 30
                    rs = [self.crosscorr(df[df.columns[i]],df[df.columns[j]], lag) for lag in range(-int(seconds*fps),int(seconds*fps+1))]
                    offset = np.floor(len(rs)/2)-np.argmax(rs)
                    cross_offsets.append(offset)
            cross_offsets = np.array(cross_offsets)
            cross_offsets = cross_offsets/max(cross_offsets)
            crossed_offset_matrix = cross_offsets.reshape((df.shape[1],df.shape[1]))
            crossed_df = pd.DataFrame(crossed_offset_matrix , columns=(df.columns)).set_index(df.columns)
            df_cc = crossed_df


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

            G_crossed = nx.from_pandas_edgelist(df_graph, 'from', 'to')
            return G_crossed
        
        if method == 'crossed_':
            cross_offsets = []
            for i in range(len(df.columns)):
                for j in range(len(df.columns)):
                    seconds = 5
                    fps = 30
                    rs = [self.crosscorr(df[df.columns[i]],df[df.columns[j]], lag) for lag in range(-int(seconds*fps),int(seconds*fps+1))]
                    offset = np.floor(len(rs)/2)-np.argmax(rs)
                    cross_offsets.append(offset)
            cross_offsets = np.array(cross_offsets)
            cross_offsets = cross_offsets/max(cross_offsets)
            crossed_offset_matrix = cross_offsets.reshape((df.shape[1],df.shape[1]))
            crossed_df = pd.DataFrame(crossed_offset_matrix , columns=(df.columns)).set_index(df.columns)
            df_cc = crossed_df


            # apply corr coeff threshold and create new df
            list_symbols = df_cc.columns.to_list()
            list_from = []
            list_to = []
            list_corr_coeff = []
            for i , sym_from in enumerate(list_symbols):
                for sym_to in list_symbols:
                    if sym_from != sym_to:
                        corr_coef = df_cc.loc[sym_from, sym_to]
                        if abs(corr_coef) < threshold:
                            list_from.append(sym_from)
                            list_to.append(sym_to)
                            list_corr_coeff.append(corr_coef)

            # create df for constructing graph
            df_graph = pd.DataFrame({'from':list_from, 'to':list_to, 
                                    'corr coeff':list_corr_coeff})

            G_crossed = nx.from_pandas_edgelist(df_graph, 'from', 'to')
            return G_crossed
        
        if method == 'crossed_100':
            cross_offsets = []
            for i in range(len(df.columns)):
                for j in range(len(df.columns)):
                    seconds = 5
                    fps = 5
                    rs = [self.crosscorr(df[df.columns[i]],df[df.columns[j]], lag) for lag in range(-int(seconds*fps),int(seconds*fps+1))]
                    offset = np.floor(len(rs)/2)-np.argmax(rs)
                    cross_offsets.append(offset)
            cross_offsets = np.array(cross_offsets)
            # cross_offsets = cross_offsets[cross_offsets>0]
            cross_offsets = cross_offsets[~np.isnan(cross_offsets)]
            cross_offsets = cross_offsets/max(cross_offsets)
            crossed_offset_matrix = cross_offsets.reshape((df.shape[1],df.shape[1]))
            crossed_df = pd.DataFrame(crossed_offset_matrix , columns=(df.columns)).set_index(df.columns)
            df_cc = crossed_df


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

            G_crossed = nx.from_pandas_edgelist(df_graph, 'from', 'to')
            return G_crossed
        
        if method == 'crossed_100_':
            cross_offsets = []
            for i in range(len(df.columns)):
                for j in range(len(df.columns)):
                    seconds = 5
                    fps = 5
                    rs = [self.crosscorr(df[df.columns[i]],df[df.columns[j]], lag) for lag in range(-int(seconds*fps),int(seconds*fps+1))]
                    offset = np.floor(len(rs)/2)-np.argmax(rs)
                    cross_offsets.append(offset)

            cross_offsets = np.array(cross_offsets)
            # cross_offsets = cross_offsets[cross_offsets>0]
            cross_offsets = cross_offsets[~np.isnan(cross_offsets)]
            cross_offsets = cross_offsets/max(cross_offsets)
            crossed_offset_matrix = cross_offsets.reshape((df.shape[1],df.shape[1]))
            crossed_df = pd.DataFrame(crossed_offset_matrix , columns=(df.columns)).set_index(df.columns)
            df_cc = crossed_df


            # apply corr coeff threshold and create new df
            list_symbols = df_cc.columns.to_list()
            list_from = []
            list_to = []
            list_corr_coeff = []
            for i , sym_from in enumerate(list_symbols):
                for sym_to in list_symbols:
                    if sym_from != sym_to:
                        corr_coef = df_cc.loc[sym_from, sym_to]
                        if abs(corr_coef) < threshold:
                            list_from.append(sym_from)
                            list_to.append(sym_to) 
                            list_corr_coeff.append(corr_coef)

            # create df for constructing graph
            df_graph = pd.DataFrame({'from':list_from, 'to':list_to, 
                                    'corr coeff':list_corr_coeff})

            G_crossed = nx.from_pandas_edgelist(df_graph, 'from', 'to')
            return G_crossed
    

    def construct_weighted_network(self,df, method, threshold):
        if method == 'pearson':
            
            df_corr = df.corr(method ='pearson')  
            
            G = nx.Graph()
            for i in range(len(df_corr)):
                for j in range(i+1 ,len(df_corr)):
                    corr_coef = df_corr.iloc[i, j]
                    if abs(corr_coef) > threshold:
                        G.add_edge(i , j, weight=df_corr.iloc[i,j])
            return G, df_corr
    
        if method == 'entropy':
            te_list = []
            for c1 in range(df.columns.size):
                for c2 in range(df.columns.size):
                    te_list.append(te.calc_te(robjects.FloatVector(df[df.columns[c1]].values), robjects.FloatVector(df[df.columns[c2]].values))[0])
            te_list = np.array(te_list).reshape((df.columns.size, df.columns.size))
        
            df_corr = pd.DataFrame(te_list, columns=df.columns).set_index(df.columns)

            G = nx.Graph()
            for i in range(len(df_corr)):
                for j in range(i+1 ,len(df_corr)):
                    corr_coef = df_corr.loc[i, j]
                    if abs(corr_coef) > threshold:
                        G.add_edge(i , j, weight=df_corr.iloc[i,j])
            return G
    
    def get_weighted_network_features(network):
    
        radius = []
        diameter = []
        densities = []
        number_of_edges = []
        average_clustering = []
        avg_degree_centrality = []
        avg_betweenness_centrality = []
        avg_closeness_centrality = []
        average_shortest_path_length = []
        average_weighted_degree_dist = []
        modularity = []
        
        col_names = ['radius', 'diameter', 'densities', 'number_of_edges', 'average_clustering', 'avg_degree_centrality',\
                'avg_betweenness_centrality', 'avg_closeness_centrality', 'average_shortest_path_length', 'average_weighted_degree_dist', 'modularity']

        try:
            radius.append(nx.radius(network, weight='weight'))
        except :
            radius.append(np.nan)

        try:
            diameter.append(nx.diameter(network, weight='weight'))
        except :
            diameter.append(np.nan)
        
        total_weight = sum(network[e[0]][e[1]]['weight'] for e in network.edges())
        potential_conection = len(network)*(len(network)-1)
        try:
            densities.append(total_weight/potential_conection)
        except :
            densities.append(np.nan)

        try:
            number_of_edges.append(nx.number_of_edges(network, weight='weight'))
        except :
            number_of_edges.append(np.nan)

        try:
            average_clustering.append(nx.average_clustering(network, weight='weight'))
        except :
            average_clustering.append(np.nan)

        try:
            avg_degree_centrality.append(np.mean(list(nx.degree_centrality(network).values())))
        except :
            avg_degree_centrality.append(np.nan)

        try:
            avg_betweenness_centrality.append(np.mean(list(nx.betweenness_centrality(network, weight='weight').values())))
        except :
            avg_betweenness_centrality.append(np.nan)

        try:
            avg_closeness_centrality.append(np.mean(list(nx.closeness_centrality(network, weight='weight').values())))
        except :
            avg_closeness_centrality.append(np.nan)

        try:
            average_shortest_path_length.append(nx.average_shortest_path_length(network, weight='weight'))
        except :
            average_shortest_path_length.append(np.nan)
            
        try:
            average_weighted_degree_dist.append(np.mean([network.degree(node , weight='weight') for node in network.nodes()]))
        except:
            average_weighted_degree_dist.append(np.nan)
            
        try:
            modularity.append(nx.community.modularity(network, nx.algorithms.community.greedy_modularity_communities(network, "weight"), weight='weight'))
        except:
            modularity.append(np.nan)
            
        df = pd.DataFrame(zip(radius, diameter, densities, number_of_edges, average_clustering, avg_degree_centrality, \
                            avg_betweenness_centrality, avg_closeness_centrality, average_shortest_path_length, average_weighted_degree_dist, modularity), columns=col_names)
        return df  

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

    def get_network_features_from_df(self,regimes_df, method, edges_threshold, input_target_length=False):
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
                    fixed_thr =  self.find_pearson_threshold(dfs , threshold = edges_threshold)
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
                    fixed_thr =  self.find_entropy_threshold(dfs , threshold = edges_threshold)
                trained_G = self.construct_network(dfs, 'entropy',fixed_thr)
                dfs_net_features_df = self.get_network_features(trained_G)
                if i == 0:
                    net_features_df = dfs_net_features_df
                net_features_df.loc[i] = dfs_net_features_df.values[0]
            print('Selected Threshold is: '+ str(np.round(fixed_thr, 3)))
            return net_features_df

    def get_network_features_no_regimes(self, df, regimes_column,target_window, target_length, edges_threshold):
        regimes_percentage = []
        regs = np.unique(regimes_column)
        df['selected_regimes'] = regimes_column

        for wind in range(1, target_length):
            regimes_percentage.append([])
            for i , dfs in enumerate([df[-target_window-wind:-wind]]):
                if wind == 1:
                    fixed_thr =  self.find_pearson_threshold(dfs.drop('selected_regimes', axis=1), threshold = edges_threshold )
                trained_G = self.construct_network(dfs, 'pearson',fixed_thr)
                dfs_net_features_df = self.get_network_features(trained_G)
                if wind == 1:
                    net_features_df = dfs_net_features_df
                net_features_df.loc[wind] = dfs_net_features_df.values[0]
                tt = df.loc[dfs.index]['selected_regimes'].values
                
                for re in range(len(regs)):
                    regimes_percentage[-1].append(tt[tt==regs[re]].size)
        return [net_features_df[::-1], np.array(regimes_percentage).T[::-1]]

    def get_network_features_averaged(self, df,regimes_column,method, edges_threshold,target_window , target_length, p_threshold = False):
        target_win = target_window
        regimes_df = self.get_regimes_DataFrames(df, regimes_column)
        for i , dfs in enumerate(regimes_df):
            for wind in range(1, target_length):
                if i==0 and wind == 1:
                    if p_threshold:
                        fixed_thr = p_threshold
                    else:
                        if method == 'pearson':
                            fixed_thr =  self.find_pearson_threshold(dfs, threshold = edges_threshold)
                        elif method == 'entropy':
                            fixed_thr =  self.find_entropy_threshold(dfs, threshold = edges_threshold)
                        # np.log(df.columns.size) * df.columns.size*
                trained_G = self.construct_network(dfs[-target_win-wind:-wind], method ,fixed_thr)
                dfs_net_features_df = self.get_network_features(trained_G)
                if wind == 1:
                    net_features_df = dfs_net_features_df
                net_features_df.loc[wind] = dfs_net_features_df.values[0]
            if i ==0:
                avg_net_features_df_tot = pd.DataFrame(net_features_df.mean()).T
            else: 
                avg_net_features_df_tot.loc[2*i] = pd.DataFrame(net_features_df.mean()).T.values[0]

            stds = []
            for k in range(len(avg_net_features_df_tot.columns)):
                stds.append(net_features_df[net_features_df.columns[k]].std())
                
            avg_net_features_df_tot.loc[2*i+1] = stds


        avg_net_features_df_tot.index = np.tile(np.array(['Means', 'STDs']),len(regimes_df))
        print('fixed_thr= '+str(fixed_thr))

        return avg_net_features_df_tot
    
    def get_network_features_averaged_median(self, df,regimes_column,method, edges_threshold,target_window , target_length, p_threshold = False):
        target_win = target_window
        regimes_df = self.get_regimes_DataFrames(df, regimes_column)
        regimes_df = regimes_df
        for i , dfs in enumerate(regimes_df):
            for wind in range(1, target_length):
                if i==0 and wind == 1:
                    if p_threshold:
                        fixed_thr = p_threshold
                    else:
                        if method == 'pearson':
                            fixed_thr =  self.find_median_pearson_threshold(dfs[-target_win-wind:-wind])
                        elif method == 'entropy':
                            fixed_thr =  self.find_median_entropy_threshold(dfs[-target_win-wind:-wind])
                        elif method == 'crossed_100':
                            fixed_thr =  self.find_median_crossed_100_threshold(dfs[-target_win-wind:-wind])
                        elif method == 'crossed_100_':
                            fixed_thr =  self.find_median_crossed_100_threshold(dfs[-target_win-wind:-wind])
                        # np.log(df.columns.size) * df.columns.size*
                trained_G = self.construct_network(dfs[-target_win-wind:-wind], method ,fixed_thr)
                dfs_net_features_df = self.get_network_features(trained_G)
                if wind == 1:
                    net_features_df = dfs_net_features_df
                net_features_df.loc[wind] = dfs_net_features_df.values[0]
            if i ==0:
                avg_net_features_df_tot = pd.DataFrame(net_features_df.mean()).T
            else: 
                avg_net_features_df_tot.loc[2*i] = pd.DataFrame(net_features_df.mean()).T.values[0]

            stds = []
            for k in range(len(avg_net_features_df_tot.columns)):
                stds.append(net_features_df[net_features_df.columns[k]].std())
                
            avg_net_features_df_tot.loc[2*i+1] = stds

        avg_net_features_df_tot.index = np.tile(np.array(['Means', 'STDs']),len(regimes_df))
        print('fixed_thr= '+str(fixed_thr))

        return avg_net_features_df_tot

    
    def plot_circos_network(self,network):
        G =  network
        for n, d in G.nodes(data=True):
            if n == 'GDP' or n=='CPI' or n == 'M2' or n=='M1' or n =='Brent Oil':
                G.nodes[n]['class'] = 'one'

            elif n == 'Coin' or n == 'Euro' or n == 'USD' or n == 'Gold' or n == 'HOUSE':
                G.nodes[n]['class'] = 'two'
            else:
                G.nodes[n]['class'] = 'three'
            # G.nodes[n]['class'] = choice(['one', 'two', 'three'])


        fig = plt.figure(figsize=(16,12))

        c = CircosPlot(G, node_color="class", node_order='class', node_labels=True, group_label_color=True)

        return c

        


                    
